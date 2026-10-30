"""Tree search over segment envs: the frontier of saved world states as a task sampler.

The env plays one segment per episode and leaves, in ``trace.info[<info_key>]``,
whether the game continues, the path of the world it saved, and the summary the
policy wrote (``crafter_env`` is the reference: ``continues``, ``state_path``,
``state_sha256``, ``summary``, ``final_goal_reached``, ``achievements``). This
sampler turns those into the next tasks. Each task it emits is a node: a fresh root
seed at depth 0, or a saved world plus the summary that ended the parent segment,
with the summary in the system prompt (the only message the turn window never
evicts). The orchestrator dispatches ``group_size`` episodes of it — the node's
branches — and GRPO baselines them against each other.

``observe`` sees the finished branches of one node. Per root and depth it runs a
beam: once every node of a depth has reported, the ``keep`` best branches over the
whole depth continue, and the states the search can no longer reach are deleted.
A depth whose groups are still in flight is released greedily instead of idling
the dispatcher: when nothing is ready, the best branch seen so far at the
shallowest open depth is expanded, and the rest of the beam is picked from what
remains when the depth closes. With ``keep = 1`` the two rules coincide. A
stale-cancelled group never reaches ``observe``; its node is simply not expanded,
and a sweep deletes state files nothing references after ``sweep_after_s``.

A branch that reaches the final goal ends its lineage: it is recorded (depth, step
and the lineage of node ids) and the root is not expanded further. Nothing in
flight is ever waited for: a sampler must always return a task, so an empty frontier
starts the next root seed.
"""

from __future__ import annotations

import glob
import itertools
import os
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import TreeSamplerConfig

# The tech tree to the diamond, for the `deepest_tier` metric.
TIERS = (
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe",
    "place_furnace", "collect_coal", "collect_iron", "make_iron_pickaxe", "collect_diamond",
)


def _data_info(data: Any) -> dict[str, Any]:
    info = getattr(data, "info", None)
    return dict(info) if isinstance(info, dict) else {}


class TreeSampler(TaskSampler):
    def __init__(self, config: TreeSamplerConfig, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        self.config = config
        self.finite = tuple(tasks) if isinstance(tasks, Sequence) else None
        if self.finite is not None and not self.finite:
            raise ValueError("TreeSampler needs at least one root task")
        self._roots: Iterator[vf.Task] = itertools.cycle(self.finite) if self.finite is not None else tasks
        self.root_cursor = 0
        self.root_tasks: dict[str, vf.Task] = {}  # root id -> the taskset's task, the template of its lineage
        self.frontier: list[dict[str, Any]] = []  # nodes ready to expand, shallowest and best first
        # (root id, depth) -> the depth's beam bookkeeping while its groups are in flight
        self.levels: dict[tuple[str, int], dict[str, Any]] = {}
        self.finished_roots: set[str] = set()
        self.found: list[dict[str, Any]] = []  # lineages that reached the final goal
        self.roots_started = 0
        self.expansions = 0
        self.segments = 0
        self.segments_per_depth: Counter[int] = Counter()
        self.deepest = 0
        self.deepest_tier = -1
        self.states_deleted = 0
        self.swept = 0
        self._observes = 0
        self._last_sweep = time.time()

    # --- tasks out ---------------------------------------------------------------
    def __next__(self) -> vf.Task:
        node = self._pop_ready() or self._release_greedy() or self._new_root()
        level = self.levels.setdefault((node["root"], node["depth"] + 1), self._new_level())
        level["expected"] += 1
        self.expansions += 1
        return self._task_for(node)

    def _new_level(self) -> dict[str, Any]:
        return {"expected": 0, "observed": 0, "released": 0, "candidates": []}

    def _pop_ready(self) -> dict[str, Any] | None:
        if not self.frontier:
            return None
        self.frontier.sort(key=lambda n: (n["depth"], -float(n.get("score", 0.0))))
        return self.frontier.pop(0)

    def _release_greedy(self) -> dict[str, Any] | None:
        """The best candidate so far of the shallowest depth still in flight, if its beam has room."""
        for key in sorted(self.levels, key=lambda k: k[1]):
            level = self.levels[key]
            if level["released"] >= self.config.keep or not level["candidates"]:
                continue
            level["candidates"].sort(key=lambda n: -float(n.get("score", 0.0)))
            node = level["candidates"].pop(0)
            level["released"] += 1
            if node["depth"] >= self.config.max_depth:
                self._delete_state(node)
                continue
            return node
        return None

    def _new_root(self) -> dict[str, Any]:
        task = next(self._roots)
        index = self.root_cursor
        self.root_cursor += 1
        self.roots_started += 1
        info = _data_info(task.data)
        root_id = f"{info.get('seed', index)}-root{index}"
        self.root_tasks[root_id] = task
        return {"id": root_id, "parent": None, "root": root_id, "root_index": index, "depth": 0,
                "state_path": None, "state_sha256": None, "summary": None, "score": 0.0, "lineage": [root_id]}

    def _task_for(self, node: dict[str, Any]) -> vf.Task:
        root = self.root_tasks[node["root"]]
        info = _data_info(root.data)
        base = info.get("base_system_prompt") or root.data.system_prompt or ""
        update: dict[str, Any] = {"info": {**info, "base_system_prompt": base, "root_name": info.get("root_name") or root.data.name, "node": dict(node)}}
        if node["depth"] > 0:
            notes = (node.get("summary") or "").strip() or "(no notes were written)"
            update["system_prompt"] = f"{base}\n\n{self.config.summary_header}\n\n{notes}"
            update["prompt"] = None
            update["name"] = f"{update['info']['root_name']}/d{node['depth']}/{node['id']}"
        return root.with_data(**update)

    # --- results in --------------------------------------------------------------
    def observe(self, group: list[vf.Episode]) -> None:
        if not group:
            return
        node = _data_info(group[0].task.data).get("node")
        if not node or node.get("root") not in self.root_tasks:
            return  # not one of ours (a resumed run's stragglers, or another sampler's task)
        root, depth = node["root"], int(node["depth"]) + 1
        level = self.levels.setdefault((root, depth), self._new_level())
        level["observed"] += 1
        self._observes += 1
        key = self.config.info_key
        for episode in group:
            for trace in episode.traces:
                info = trace.info.get(key)
                if not isinstance(info, dict):
                    continue
                self.segments += 1
                self.segments_per_depth[depth] += 1
                self.deepest = max(self.deepest, depth)
                achievements = set(info.get("achievements") or [])
                for tier, name in enumerate(TIERS):
                    if name in achievements:
                        self.deepest_tier = max(self.deepest_tier, tier)
                child = {
                    "id": info.get("node_id") or trace.id, "parent": node["id"], "root": root, "root_index": node.get("root_index", 0),
                    "depth": depth, "state_path": info.get("state_path"), "state_sha256": info.get("state_sha256"),
                    "summary": info.get("summary"), "score": float(trace.reward), "achievements": sorted(achievements),
                    "lineage": [*node.get("lineage", [node["id"]]), info.get("node_id") or trace.id],
                }
                if info.get("final_goal_reached"):
                    self.found.append({"root": root, "depth": depth, "node": child["id"], "lineage": child["lineage"],
                                       "game_step": info.get("game_step"), "expansions": self.expansions, "segments": self.segments,
                                       "trace_id": trace.id, "at": time.time()})
                    get_logger().info(f"TreeSampler: final goal reached at depth {depth} by node {child['id']} (root {root})")
                    self.finished_roots.add(root)
                    self._delete_state(child)
                elif episode.ok and getattr(trace, "ok", True) and info.get("continues") and info.get("state_path") and depth < self.config.max_depth:
                    level["candidates"].append(child)
                else:
                    self._delete_state(child)
        self._delete_state(node)  # every branch restored it; the beam never comes back
        if root in self.finished_roots:
            self._drop_root(root)
        elif level["observed"] >= level["expected"]:
            self._close_level(root, depth)
        if self.config.state_dir and time.time() - self._last_sweep > self.config.sweep_after_s / 4:
            self._sweep()

    def _close_level(self, root: str, depth: int) -> None:
        level = self.levels.pop((root, depth), None)
        if level is None:
            return
        level["candidates"].sort(key=lambda n: -float(n.get("score", 0.0)))
        room = max(0, self.config.keep - level["released"])
        kept, dropped = level["candidates"][:room], level["candidates"][room:]
        self.frontier.extend(kept)
        for node in dropped:
            self._delete_state(node)

    def _drop_root(self, root: str) -> None:
        for key in [k for k in self.levels if k[0] == root]:
            for node in self.levels.pop(key)["candidates"]:
                self._delete_state(node)
        keep = []
        for node in self.frontier:
            (self._delete_state(node) if node["root"] == root else keep.append(node))
        self.frontier = keep

    # --- states ------------------------------------------------------------------
    def _delete_state(self, node: dict[str, Any]) -> None:
        path = node.get("state_path")
        if not path or not os.path.isabs(path):
            return  # memory-backed stores live in the env process
        try:
            os.remove(path)
            self.states_deleted += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            get_logger().warning(f"TreeSampler: could not delete {path}: {exc}")

    def _referenced(self) -> set[str]:
        nodes = list(self.frontier) + [n for level in self.levels.values() for n in level["candidates"]]
        return {n["state_path"] for n in nodes if n.get("state_path")}

    def _sweep(self) -> None:
        self._last_sweep = time.time()
        keep, cutoff = self._referenced(), time.time() - self.config.sweep_after_s
        for path in glob.glob(os.path.join(self.config.state_dir, "*.pkl")):
            if path in keep:
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    self.swept += 1
            except OSError:
                pass

    # --- checkpoint and metrics ----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        if self.finite is None:
            raise RuntimeError("TreeSampler can only checkpoint a finite taskset (its roots are re-derived by index)")
        return {
            "root_cursor": self.root_cursor,
            "root_index_by_id": {root_id: self._root_index(root_id) for root_id in self.root_tasks},
            "frontier": self.frontier,
            "levels": [{"root": k[0], "depth": k[1], **v} for k, v in self.levels.items()],
            "finished_roots": sorted(self.finished_roots),
            "found": self.found,
            "counters": {"roots_started": self.roots_started, "expansions": self.expansions, "segments": self.segments,
                         "deepest": self.deepest, "deepest_tier": self.deepest_tier, "states_deleted": self.states_deleted},
            "segments_per_depth": dict(self.segments_per_depth),
        }

    def _root_index(self, root_id: str) -> int:
        for node in self.frontier + [n for level in self.levels.values() for n in level["candidates"]]:
            if node["root"] == root_id:
                return int(node["root_index"])
        return int(root_id.rsplit("root", 1)[-1])

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.finite is None:
            raise RuntimeError("TreeSampler can only resume a finite taskset")
        self.root_cursor = int(state_dict["root_cursor"])
        self._roots = itertools.cycle(self.finite)
        for _ in range(self.root_cursor % len(self.finite)):
            next(self._roots)
        self.root_tasks = {root_id: self.finite[int(index) % len(self.finite)] for root_id, index in state_dict["root_index_by_id"].items()}
        self.frontier = list(state_dict["frontier"])
        self.levels = {(lvl["root"], int(lvl["depth"])): {k: v for k, v in lvl.items() if k not in {"root", "depth"}} for lvl in state_dict["levels"]}
        self.finished_roots = set(state_dict["finished_roots"])
        self.found = list(state_dict["found"])
        counters = state_dict["counters"]
        self.roots_started, self.expansions, self.segments = counters["roots_started"], counters["expansions"], counters["segments"]
        self.deepest, self.deepest_tier, self.states_deleted = counters["deepest"], counters["deepest_tier"], counters["states_deleted"]
        self.segments_per_depth = Counter({int(k): v for k, v in state_dict["segments_per_depth"].items()})
        # In-flight groups of the checkpointed run never report: let their depths close on what arrived.
        for level in self.levels.values():
            level["expected"] = level["observed"]
        for key in list(self.levels):
            self._close_level(*key)

    def metrics(self) -> dict[str, float]:
        return {
            "frontier": float(len(self.frontier)),
            "open_depths": float(len(self.levels)),
            "roots_started": float(self.roots_started),
            "expansions": float(self.expansions),
            "segments": float(self.segments),
            "deepest_depth": float(self.deepest),
            "deepest_tier": float(self.deepest_tier),
            "found": float(len(self.found)),
            "states_deleted": float(self.states_deleted + self.swept),
        }

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

``selection = "puct"`` replaces the beam with one archive over all worlds (TTT-Discover's reuse
rule, appendix A.2): the best ``puct_keep_children`` branches of every expansion enter it with the
value the env gives their state, a parent's saved game stays, and the next node is the archive's
best by ``Q + c * scale * P * sqrt(1 + T) / (1 + n)`` unless an ancestor or descendant of it is in
flight. ``fresh_root_share`` of the expansions start a new world instead.
"""

from __future__ import annotations

import glob
import itertools
import math
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


def _segments_info(info: dict[str, Any]) -> dict[str, Any]:
    """What the env's taskset says about its segments (``info["segments"]``: ``info_key``,
    ``continuation_header``); it overrides the sampler's config, so the child prompt is the
    one the env's own in-env beam builds."""
    seg = info.get("segments")
    return seg if isinstance(seg, dict) else {}


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
        self.deepest_floor = 0
        self.groups_without_info = 0
        self.states_deleted = 0
        self.swept = 0
        self._observes = 0
        self._last_sweep = time.time()
        self.step = 0  # the orchestrator step (TrainSource.next_task sets it): picks a new world's cap
        # selection = "puct": node id -> node (with "n" visits and "m", the best value among its branches)
        self.archive: dict[str, dict[str, Any]] = {}
        self.puct_inflight: dict[str, dict[str, Any]] = {}  # archive nodes being expanded: {"lineage", "sent_at"}
        self.puct_expansions = 0  # T: archive expansions that reported
        self.fresh_issued = 0
        self.reexpansions = 0  # picks of a node that had been expanded before
        self.archive_evicted = 0
        # puct_scope = "world": root id -> {"n" expansions, "best" value, "stale" expansions without a new best}
        self.worlds: dict[str, dict[str, Any]] = {}
        self.worlds_retired = 0
        # archive picks, and those that went back: the node is above the deepest node of its world that could be picked
        self.puct_picks, self.backtracks, self.backtrack_layers = 0, 0, 0

    # --- tasks out ---------------------------------------------------------------
    def __next__(self) -> vf.Task | None:
        if self.config.selection == "puct":
            return self._next_puct()
        node = self._pop_ready() or self._release_greedy()
        if node is None:
            if self.config.max_roots is not None and len(self._open_roots()) >= self.config.max_roots:
                return None  # every open world is in flight: wait for one (the dispatcher polls again)
            node = self._new_root()
        level = self.levels.setdefault((node["root"], node["depth"] + 1), self._new_level())
        level["expected"] += 1
        # Remember the node until its group reports: a checkpoint taken meanwhile must be able to play it
        # again (its saved game stays on disk until then), or a resume would end the world.
        level.setdefault("inflight", []).append(dict(node))
        self.expansions += 1
        return self._task_for(node)

    def _next_puct(self) -> vf.Task:
        now = time.time()
        for node_id in [k for k, v in self.puct_inflight.items() if now - v["sent_at"] > self.config.sweep_after_s]:
            del self.puct_inflight[node_id]  # a stale-cancelled group never reports: the node may be picked again
        node = None
        if self.fresh_issued >= self.config.fresh_root_share * (self.expansions + 1):
            node = self._select_puct()
        if node is None:
            node = self._new_root()
            self.fresh_issued += 1
        else:
            self.puct_inflight[node["id"]] = {"lineage": list(node.get("lineage", [node["id"]])), "sent_at": now, "root": node["root"]}
            self.reexpansions += node["n"] > 0
            deepest = max(n["depth"] for n in self.archive.values() if n["root"] == node["root"] and n["depth"] < self._cap(n))
            self.puct_picks += 1
            self.backtracks += node["depth"] < deepest
            self.backtrack_layers += deepest - node["depth"]
        self.expansions += 1
        return self._task_for(node)

    def _puct_scores(self, nodes: Any = None, expansions: int | None = None) -> list[tuple[float, dict[str, Any]]]:
        """(score, node) for every node of `nodes` (default: the whole archive), best value first."""
        ranked = sorted(self.archive.values() if nodes is None else nodes, key=lambda n: (-n["score"], n["n"], n["depth"]))
        if not ranked:
            return []
        size = len(ranked)
        total = size * (size + 1) / 2
        scale = (ranked[0]["score"] - ranked[-1]["score"]) or 1.0
        explore = self.config.puct_c * scale * math.sqrt(1 + (self.puct_expansions if expansions is None else expansions))
        out = []
        for rank, node in enumerate(ranked):
            q = node["m"] if node["n"] > 0 and node["m"] is not None else node["score"]
            out.append((q + explore * ((size - rank) / total) / (1 + node["n"]), node))
        return out

    def _select_puct(self) -> dict[str, Any] | None:
        """The archive's best node by the PUCT score, skipping the lineages in flight (the ancestors and
        descendants of a node being expanded) and nodes at their world's depth cap."""
        if self.config.puct_scope == "world":
            return self._select_world()
        return self._best_free(self._puct_scores())

    def _best_free(self, scored: list[tuple[float, dict[str, Any]]]) -> dict[str, Any] | None:
        busy_ids = set(self.puct_inflight)
        busy_ancestors = {a for sent in self.puct_inflight.values() for a in sent["lineage"]}
        best, best_score = None, -math.inf
        for score, node in scored:
            if score <= best_score or node["depth"] >= self._cap(node):
                continue
            if node["id"] in busy_ancestors or busy_ids.intersection(node.get("lineage", ())):
                continue
            best, best_score = node, score
        return best

    def _select_world(self) -> dict[str, Any] | None:
        """`puct_scope = "world"`: the world with the fewest expansions gets the next one (a world's
        value never decides how often it is picked: values only add up, so the deepest world would
        take every pick), and the PUCT score picks the node inside it."""
        flying: dict[str, int] = {}
        for sent in self.puct_inflight.values():
            flying[sent.get("root")] = flying.get(sent.get("root"), 0) + 1
        by_world: dict[str, list[dict[str, Any]]] = {}
        for node in self.archive.values():
            by_world.setdefault(node["root"], []).append(node)
        order = sorted(self.worlds, key=lambda r: (self.worlds[r]["n"] + flying.get(r, 0), -self.worlds[r]["best"]))
        for root in order:
            if flying.get(root, 0) >= self.config.world_inflight:
                continue
            node = self._best_free(self._puct_scores(by_world.get(root, []), self.worlds[root]["n"]))
            if node is not None:
                return node
        return None

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _cap_now(self) -> int:
        """The depth cap a world started now gets (max_depth_schedule, else max_depth)."""
        cap = self.config.max_depth
        for start, depth in self.config.max_depth_schedule or ():
            if self.step >= start:
                cap = depth
        return cap

    def _cap(self, node: dict[str, Any]) -> int:
        """A node's world's cap, fixed when the world started (nodes from older checkpoints: max_depth)."""
        return int(node.get("max_depth") or self.config.max_depth)

    def _open_roots(self) -> set[str]:
        """Worlds still being played: a depth in flight or a node waiting in the frontier."""
        return {root for root, _ in self.levels} | {node["root"] for node in self.frontier}

    def _new_level(self) -> dict[str, Any]:
        return {"expected": 0, "observed": 0, "released": 0, "candidates": [], "inflight": []}

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
            if node["depth"] >= self._cap(node):
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
                "state_path": None, "state_sha256": None, "summary": None, "score": 0.0, "lineage": [root_id],
                "max_depth": self._cap_now()}

    def _task_for(self, node: dict[str, Any]) -> vf.Task:
        root = self.root_tasks[node["root"]]
        info = _data_info(root.data)
        base = info.get("base_system_prompt") or root.data.system_prompt or ""
        update: dict[str, Any] = {"info": {**info, "base_system_prompt": base, "root_name": info.get("root_name") or root.data.name, "node": dict(node)}}
        if node["depth"] > 0:
            seg = _segments_info(info)
            notes = (node.get("summary") or "").strip() or seg.get("empty_summary") or "(no notes were written)"
            header = seg.get("continuation_header") or self.config.summary_header
            update["system_prompt"] = f"{base}\n\n{header}\n\n{notes}"
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
        puct = self.config.selection == "puct"
        level = self._new_level() if puct else self.levels.setdefault((root, depth), self._new_level())
        level["observed"] += 1
        inflight = level.setdefault("inflight", [])
        for i, sent in enumerate(inflight):
            if sent.get("id") == node.get("id"):
                inflight.pop(i)
                break
        self._observes += 1
        values: list[float] = []
        key = _segments_info(_data_info(group[0].task.data)).get("info_key") or self.config.info_key
        if not any(isinstance(trace.info.get(key), dict) for episode in group for trace in episode.traces):
            # The env wrote nothing under `key`: every branch would be a leaf and the tree never grows.
            self.groups_without_info += 1
            if self.groups_without_info in (1, 10, 100):
                get_logger().warning(f"TreeSampler: a group of node {node['id']} has no trace.info[{key!r}] "
                                     f"({self.groups_without_info} so far): check info_key / the env's segment settings")
        for episode in group:
            for trace in episode.traces:
                info = trace.info.get(key)
                if not isinstance(info, dict):
                    continue
                self.segments += 1
                self.segments_per_depth[depth] += 1
                self.deepest = max(self.deepest, depth)
                self.deepest_floor = max(self.deepest_floor, int(info.get("max_floor") or 0))
                achievements = set(info.get("achievements") or [])
                for tier, name in enumerate(TIERS):
                    if name in achievements:
                        self.deepest_tier = max(self.deepest_tier, tier)
                child = {
                    "id": info.get("node_id") or trace.id, "parent": node["id"], "root": root, "root_index": node.get("root_index", 0),
                    "depth": depth, "state_path": info.get("state_path"), "state_sha256": info.get("state_sha256"),
                    "summary": info.get("summary"), "score": self._value(info, trace), "achievements": sorted(achievements),
                    "lineage": [*node.get("lineage", [node["id"]]), info.get("node_id") or trace.id],
                    "max_depth": node.get("max_depth"),
                }
                values.append(child["score"])
                if info.get("final_goal_reached"):
                    self.found.append({"root": root, "depth": depth, "node": child["id"], "lineage": child["lineage"],
                                       "game_step": info.get("game_step"), "expansions": self.expansions, "segments": self.segments,
                                       "trace_id": trace.id, "at": time.time()})
                    get_logger().info(f"TreeSampler: final goal reached at depth {depth} by node {child['id']} (root {root})")
                    if not puct:
                        self.finished_roots.add(root)
                    self._delete_state(child)
                elif episode.ok and getattr(trace, "ok", True) and info.get("continues") and info.get("state_path") and depth < self._cap(node):
                    level["candidates"].append(child)
                else:
                    self._delete_state(child)
        if puct:
            self._archive_branches(node, level["candidates"], values)
        else:
            self._delete_state(node)  # every branch restored it; the beam never comes back
        if puct:
            pass
        elif root in self.finished_roots:
            self._drop_root(root)
        elif level["observed"] >= level["expected"]:
            self._close_level(root, depth)
        if self.config.state_dir and time.time() - self._last_sweep > self.config.sweep_after_s / 4:
            self._sweep()

    def _value(self, info: dict[str, Any], trace: Any) -> float:
        """A branch's value: the beam's score, and under `puct` the value of the state it reached."""
        key = self.config.value_key if self.config.selection == "puct" else None
        value = info.get(key) if key else None
        return float(value) if isinstance(value, (int, float)) else float(trace.reward)

    def _archive_branches(self, node: dict[str, Any], branches: list[dict[str, Any]], values: list[float]) -> None:
        """`puct`: one expansion of `node` reported. Its best branch value is the node's Q, the node and
        its ancestors count a visit, and its best branches enter the archive."""
        self.puct_inflight.pop(node["id"], None)
        if self.config.puct_scope == "world" and not self._world_reports(node, branches, values):
            return
        if node["depth"] > 0:  # an archive pick (a fresh root is not a state of the archive)
            self.puct_expansions += 1
            parent = self.archive.get(node["id"])
            if parent is not None and values:
                parent["m"] = max(values) if parent["m"] is None else max(parent["m"], max(values))
            for ancestor in node.get("lineage", [node["id"]]):
                if ancestor in self.archive:
                    self.archive[ancestor]["n"] += 1
        branches.sort(key=lambda n: -n["score"])
        for child in branches[: self.config.puct_keep_children]:
            self.archive[child["id"]] = {**child, "n": 0, "m": None}
        for child in branches[self.config.puct_keep_children :]:
            self._delete_state(child)
        extra = len(self.archive) - self.config.archive_size
        if extra > 0:
            busy = {a for sent in self.puct_inflight.values() for a in sent["lineage"][-1:]}
            for _ in range(extra):
                free = [n for n in self.archive.values() if n["id"] not in busy]
                if self.config.puct_scope == "world":  # the largest world gives up a node, not the youngest world
                    sizes: dict[str, int] = {}
                    for n in self.archive.values():
                        sizes[n["root"]] = sizes.get(n["root"], 0) + 1
                    free.sort(key=lambda n: (-sizes[n["root"]], n["score"], -n["n"]))
                else:
                    free.sort(key=lambda n: (n["score"], -n["n"]))
                if not free:
                    break
                old = free[0]
                del self.archive[old["id"]]
                self._delete_state(old)
                self.archive_evicted += 1

    def _world_reports(self, node: dict[str, Any], branches: list[dict[str, Any]], values: list[float]) -> bool:
        """`puct_scope = "world"`: admit, update or retire the world of the expansion that reported.
        False: its branches do not enter the archive (their saved games are deleted)."""
        root, world = node["root"], self.worlds.get(node["root"])
        if node["depth"] == 0 and world is None:
            if self.config.max_roots is None or len(self.worlds) < self.config.max_roots:
                # a new world starts level with the least expanded one, so it does not take every pick to catch up
                world = self.worlds[root] = {"n": min((w["n"] for w in self.worlds.values()), default=0), "best": max(values, default=0.0), "stale": 0}
                return True
        if world is None or node["depth"] == 0:  # no room for a new world, or the world was retired in flight
            for child in branches:
                self._delete_state(child)
            return False
        world["n"] += 1
        best = max(values, default=-math.inf)
        if best > world["best"]:
            world["best"], world["stale"] = best, 0
        else:
            world["stale"] += 1
        still_flying = any(sent.get("root") == root for sent in self.puct_inflight.values())
        if world["stale"] >= self.config.world_patience and not still_flying:
            for child in branches:
                self._delete_state(child)
            for old in [n for n in self.archive.values() if n["root"] == root]:
                del self.archive[old["id"]]
                self._delete_state(old)
            del self.worlds[root]
            self.worlds_retired += 1
            return False
        return True

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
        nodes = list(self.frontier) + [n for level in self.levels.values() for n in level["candidates"] + level.get("inflight", [])]
        nodes += list(self.archive.values())
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
                         "deepest": self.deepest, "deepest_tier": self.deepest_tier, "states_deleted": self.states_deleted,
                         "deepest_floor": self.deepest_floor},
            "segments_per_depth": dict(self.segments_per_depth),
            "puct": {"archive": list(self.archive.values()), "expansions": self.puct_expansions, "fresh_issued": self.fresh_issued,
                     "reexpansions": self.reexpansions, "evicted": self.archive_evicted,
                     "worlds": self.worlds, "worlds_retired": self.worlds_retired,
                     "picks": [self.puct_picks, self.backtracks, self.backtrack_layers]},
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
        self.deepest_floor = int(counters.get("deepest_floor", 0))
        self.segments_per_depth = Counter({int(k): v for k, v in state_dict["segments_per_depth"].items()})
        # In-flight groups of the checkpointed run never report: let their depths close on what arrived, and
        # play the nodes they were expanding again (from the same saved game) so their worlds go on. Checkpoints
        # written before the "inflight" field existed close those depths empty, which ends those worlds.
        replay = []
        for level in self.levels.values():
            replay.extend(level.get("inflight", []))
            level["inflight"] = []
            level["expected"] = level["observed"]
        for key in list(self.levels):
            self._close_level(*key)
        self.frontier.extend(replay)
        self.replayed = len(replay)
        # puct: the archive keeps every node an in-flight group was expanding, so nothing is replayed.
        puct = state_dict.get("puct") or {}
        self.archive = {n["id"]: n for n in puct.get("archive", [])}
        self.puct_inflight = {}
        self.puct_expansions, self.fresh_issued = int(puct.get("expansions", 0)), int(puct.get("fresh_issued", 0))
        self.reexpansions, self.archive_evicted = int(puct.get("reexpansions", 0)), int(puct.get("evicted", 0))
        self.worlds, self.worlds_retired = dict(puct.get("worlds", {})), int(puct.get("worlds_retired", 0))
        self.puct_picks, self.backtracks, self.backtrack_layers = (int(x) for x in puct.get("picks", (0, 0, 0)))

    def metrics(self) -> dict[str, float]:
        puct = {}
        if self.config.selection == "puct":
            nodes = list(self.archive.values())
            puct = {
                "archive": float(len(nodes)),
                "archive_best_value": max((n["score"] for n in nodes), default=0.0),
                "archive_mean_depth": sum(n["depth"] for n in nodes) / len(nodes) if nodes else 0.0,
                "archive_unvisited": float(sum(n["n"] == 0 for n in nodes)),
                "archive_worlds": float(len({n["root"] for n in nodes})),
                "archive_evicted": float(self.archive_evicted),
                "puct_picks": float(self.puct_picks),
                "backtracks": float(self.backtracks),  # picks above the deepest pickable node of their world
                "backtrack_share": self.backtracks / max(1, self.puct_picks),
                "backtrack_mean_layers": self.backtrack_layers / max(1, self.backtracks),
                "worlds_active": float(len(self.worlds)),
                "worlds_retired": float(self.worlds_retired),
                "world_expansions_max_share": max((w["n"] for w in self.worlds.values()), default=0) / max(1, sum(w["n"] for w in self.worlds.values())),
                "puct_expansions": float(self.puct_expansions),
                "puct_inflight": float(len(self.puct_inflight)),
                "reexpansions": float(self.reexpansions),
                "fresh_root_share": self.fresh_issued / self.expansions if self.expansions else 0.0,
            }
        return puct | {
            "frontier": float(len(self.frontier)),
            "replayed_on_resume": float(getattr(self, "replayed", 0)),
            "open_depths": float(len(self.levels)),
            "roots_started": float(self.roots_started),
            "open_roots": float(len(self._open_roots())),
            "expansions": float(self.expansions),
            "segments": float(self.segments),
            "deepest_depth": float(self.deepest),
            "max_depth_now": float(self._cap_now()),
            "deepest_tier": float(self.deepest_tier),
            "deepest_floor": float(self.deepest_floor),
            "groups_without_info": float(self.groups_without_info),
            "found": float(len(self.found)),
            "states_deleted": float(self.states_deleted + self.swept),
        }

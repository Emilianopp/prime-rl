"""A population of games played asynchronously, one segment per task.

The sampler keeps ``lineages`` games alive. Each game (a lineage) has one node at a time: a
fresh seed, or the saved world plus handoff that ended its last segment. A node goes out as
one task; the orchestrator plays it ``group_size`` times (its branches, each from the same
save, sampled independently) and the algorithm scores them against each other. When a
node's group comes back (``observe``):

- its best branch (highest reward, then most achievement points) becomes the lineage's next
  node; the other branches' saves and the node's own save are deleted;
- unless the lineage is recycled: it has played at least ``grace`` segments, fewer than
  ``young_frac`` of the lineages are young (under ``grace`` segments), at least
  ``min_mature`` lineages are mature, and its recent progress (achievement points over its
  last ``window`` segments) ranks in the bottom ``cull_frac`` of the mature lineages (ties:
  fewer total points first, then at random). A recycled lineage ends, and its slot goes
  to a fresh seed;
- a lineage also ends at ``max_depth`` segments or when its game ends (no branch can
  continue); a node whose branches all failed is replayed up to ``max_retries`` times.

Nothing waits for a generation: every decision happens when one node's group arrives, and
``__next__`` hands out a ready node (deepest first), else a fresh seed while fewer than
``lineages`` games are alive, else nothing (``None``: the dispatcher asks again shortly).
So at most ``lineages`` nodes are in flight. A node whose group never comes back (cancelled
as stale) is replayed after ``lost_after_s``. The population is written to
``{state_dir}/population.json`` after every change, and every decision is appended to
``{state_dir}/population.jsonl``: a run started on the same folder resumes every lineage
from its last kept node.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import PopulationSamplerConfig


def _branch_info(trace: vf.Trace) -> dict[str, Any] | None:
    info = trace.info.get("craftax") if isinstance(trace.info, dict) else None
    return info if isinstance(info, dict) else None


class PopulationSampler(TaskSampler):
    def __init__(self, config: PopulationSamplerConfig, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        if not isinstance(tasks, Sequence):
            tasks = list(tasks)
        if not tasks:
            raise ValueError("PopulationSampler needs root tasks (fresh seeds)")
        self.config = config
        self.tasks = tasks
        self.dir = os.path.expandvars(config.state_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.rng = random.Random(config.seed)
        self.lineages: dict[str, dict[str, Any]] = {}
        self.root_cursor = 0
        self.counts = {"fresh": 0, "continued": 0, "recycled": 0, "ended": 0, "retried": 0, "lost": 0}
        self._load()

    # --- persistence -------------------------------------------------------------
    @property
    def _state_path(self) -> str:
        return os.path.join(self.dir, "population.json")

    def _load(self) -> None:
        try:
            with open(self._state_path) as f:
                state = json.load(f)
        except FileNotFoundError:
            return
        self.root_cursor = state["root_cursor"]
        self.counts.update(state.get("counts", {}))
        for lineage in state["lineages"]:
            if lineage["status"] == "inflight":  # its group died with the last run: play it again
                lineage["status"] = "ready"
            self.lineages[lineage["id"]] = lineage
        get_logger().info(f"PopulationSampler: resumed {len(self.lineages)} lineages from {self._state_path}")

    def _save(self) -> None:
        state = {"root_cursor": self.root_cursor, "counts": self.counts, "lineages": list(self.lineages.values())}
        tmp = f"{self._state_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self._state_path)

    def _event(self, kind: str, lineage: dict[str, Any], **extra: Any) -> None:
        line = {"time": time.time(), "event": kind, "lineage": lineage["id"], "seed": lineage["seed"],
                "depth": lineage["depth"], "points": lineage["points"][-1], **extra}
        with open(os.path.join(self.dir, "population.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")

    def _delete(self, path: str | None) -> None:
        if path and os.path.dirname(os.path.abspath(path)) == os.path.abspath(self.dir):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    # --- tasks out ---------------------------------------------------------------
    def __next__(self) -> vf.Task | None:
        self._sweep_lost()
        ready = [lin for lin in self.lineages.values() if lin["status"] == "ready"]
        if ready:
            lineage = max(ready, key=lambda lin: (lin["depth"], lin["points"][-1]))  # deepest first
        elif len(self.lineages) < self.config.lineages:
            lineage = self._new_lineage()
        else:
            return None
        lineage["status"] = "inflight"
        lineage["attempt"] = uuid.uuid4().hex[:12]
        lineage["dispatched_at"] = time.time()
        self._save()
        return self._task_for(lineage)

    def _new_lineage(self) -> dict[str, Any]:
        index = self.root_cursor % len(self.tasks)
        self.root_cursor += 1
        root = self.tasks[index]
        seed = (root.data.info or {}).get("seed", index)
        lineage = {"id": f"L{self.root_cursor:05d}-s{seed}", "root": index, "seed": seed, "depth": 0,
                   "points": [0.0], "load": None, "start_score": None, "status": "ready", "retries": 0,
                   "born": time.time()}
        self.lineages[lineage["id"]] = lineage
        self.counts["fresh"] += 1
        self._event("fresh", lineage)
        return lineage

    def _task_for(self, lineage: dict[str, Any]) -> vf.Task:
        root = self.tasks[lineage["root"]]
        info = dict(root.data.info or {})
        info["tree"] = {"lineage": lineage["id"], "attempt": lineage["attempt"], "depth": lineage["depth"],
                        "seed": lineage["seed"], "id": f"{lineage['id']}/d{lineage['depth']}",
                        "load": lineage["load"], "start_score": lineage["start_score"], "dir": self.dir}
        return root.with_data(info=info, name=f"{root.data.name}/{lineage['id']}/d{lineage['depth']}")

    def _sweep_lost(self) -> None:
        now = time.time()
        for lineage in list(self.lineages.values()):
            if lineage["status"] == "inflight" and now - lineage.get("dispatched_at", now) > self.config.lost_after_s:
                self.counts["lost"] += 1
                self._retry_or_end(lineage, "lost")

    def _retry_or_end(self, lineage: dict[str, Any], why: str) -> None:
        if lineage["retries"] < self.config.max_retries:
            lineage["retries"] += 1
            lineage["status"] = "ready"
            self.counts["retried"] += 1
            self._event("retry", lineage, why=why)
        else:
            self._end(lineage, f"{why}, retries spent")

    def _end(self, lineage: dict[str, Any], why: str, kind: str = "ended") -> None:
        self._delete(lineage["load"])
        self.lineages.pop(lineage["id"], None)
        self.counts[kind] += 1
        self._event(kind, lineage, why=why)

    # --- results in --------------------------------------------------------------
    def observe(self, group: list[vf.Episode]) -> None:
        if not group:
            return
        node = (group[0].task.data.info or {}).get("tree") or {}
        lineage = self.lineages.get(node.get("lineage"))
        if lineage is None or lineage.get("attempt") != node.get("attempt") or lineage["status"] != "inflight":
            for episode in group:  # a stale or replayed attempt's result: only clean up its saves
                for trace in episode.traces:
                    self._delete((_branch_info(trace) or {}).get("state_path"))
            return
        branches = []
        for episode in group:
            for trace in episode.traces:
                info = _branch_info(trace)
                if info is None:
                    continue
                score = trace.info.get("score") if isinstance(trace.info.get("score"), dict) else {}
                branches.append({"ok": episode.ok and not trace.has_error, "info": info, "reward": score.get("reward"),
                                 "end": score.get("end"), "path": info.get("state_path")})
        candidates = [b for b in branches if b["ok"] and b["info"].get("continues") and b["path"] and os.path.exists(b["path"])]
        rewards = [b["reward"] for b in branches if b["reward"] is not None]
        if not candidates:
            for b in branches:
                self._delete(b["path"])
            if any(b["ok"] and b["info"].get("ended_by") in ("won", "died", "max_episode_steps") for b in branches):
                self._end(lineage, "game over")
            else:
                self._retry_or_end(lineage, "no branch could continue")
            self._save()
            return
        best = max(candidates, key=lambda b: (b["reward"] if b["reward"] is not None else float("-inf"),
                                              float(b["info"].get("points") or 0.0)))
        for b in branches:
            if b is not best:
                self._delete(b["path"])
        self._delete(lineage["load"])
        lineage.update(depth=lineage["depth"] + 1, load=best["path"], start_score=best["end"], retries=0, status="ready")
        lineage["points"].append(float(best["info"].get("points") or 0.0))
        stats = {"reward": best["reward"], "rewards": rewards, "branches": len(branches),
                 "progress": self._progress(lineage), "turn": best["info"].get("game_steps")}
        verdict = self._verdict(lineage)
        if verdict is None:
            self.counts["continued"] += 1
            self._event("continue", lineage, **stats)
        else:
            self._end(lineage, verdict, kind="recycled" if verdict.startswith("recycled") else "ended")
        self._save()

    def _progress(self, lineage: dict[str, Any]) -> float:
        points = lineage["points"]
        return points[-1] - points[max(0, len(points) - 1 - self.config.window)]

    def _verdict(self, lineage: dict[str, Any]) -> str | None:
        """None to continue the lineage, else why it ends."""
        cfg = self.config
        if lineage["depth"] >= cfg.max_depth:
            return f"max_depth {cfg.max_depth}"
        if lineage["depth"] < cfg.grace:
            return None
        live = list(self.lineages.values())
        young = sum(1 for lin in live if lin["depth"] < cfg.grace)
        if young >= cfg.young_frac * cfg.lineages:
            return None
        mature = [lin for lin in live if lin["depth"] >= cfg.grace]
        if len(mature) < cfg.min_mature:
            return None
        ranked = sorted(mature, key=lambda lin: (self._progress(lin), lin["points"][-1], self.rng.random()))
        rank = next(i for i, lin in enumerate(ranked) if lin is lineage)
        if rank < cfg.cull_frac * len(ranked):
            return (f"recycled: progress {self._progress(lineage):g} over {cfg.window} segments ranks {rank + 1} of "
                    f"{len(ranked)} mature, {young} of {len(live)} lineages young")
        return None

    # --- reporting ---------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {}  # the population lives in {state_dir}/population.json

    def metrics(self) -> dict[str, float]:
        live = list(self.lineages.values())
        depths = [lin["depth"] for lin in live] or [0]
        return {
            "lineages": float(len(live)),
            "inflight": float(sum(lin["status"] == "inflight" for lin in live)),
            "young": float(sum(lin["depth"] < self.config.grace for lin in live)),
            "depth/mean": sum(depths) / len(depths),
            "depth/max": float(max(depths)),
            "points/max": max((lin["points"][-1] for lin in live), default=0.0),
            **{f"count/{k}": float(v) for k, v in self.counts.items()},
        }

from __future__ import annotations

import asyncio
import importlib
import json
import time
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TraceCreditAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.turn_credit import load_scorer
from prime_rl.orchestrator.utils import train_work

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class TraceCreditAlgorithm(Algorithm):
    """Per-trace credit from an env's own async scorer. With ``baseline = "running"`` every
    sampled token of a trace carries the trace's reward minus a running baseline (an
    exponential average of the rewards scored before it, so a trace never sets its own
    baseline): for segments shipped one at a time (``stream_segments``), where no group
    shares a start. With ``baseline = "group"`` each episode is one segment played from a
    saved world its group shares (the population sampler's branches): each branch's
    segment trace carries its reward minus the group's mean, and the branches share one
    start score, so the judge's noise on the start cancels out of the comparison. A trace
    the scorer cannot score (it returns None) carries 0. With ``log_path`` set, each scored
    trace appends a JSON line: its reward, advantage, baseline and the scorer's info."""

    def __init__(self, config: TraceCreditAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.scorer = load_scorer(config.scorer)
        self.scorer_args = config.scorer_args
        self.decay = config.baseline_decay
        self.group_baseline = config.baseline == "group"
        self.log_path = config.log_path
        self.baseline: float | None = None

    async def score_episode(self, episode: vf.Episode) -> None:
        if self.group_baseline:
            return  # scored with its group in score_group
        for _, trace in iter_trainable_traces([episode]):
            reward, info = await self.scorer(trace, **self.scorer_args)
            baseline = self.baseline
            if reward is None:
                advantage = 0.0
            else:
                advantage = reward - (baseline if baseline is not None else reward)
                self.baseline = reward if baseline is None else self.decay * baseline + (1 - self.decay) * reward
            assign_advantages(trace, advantage)
            self._log(episode, trace, reward, advantage, baseline, info)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        if not self.group_baseline:
            return
        main: list[tuple[vf.Episode, vf.Trace]] = []
        for episode, trace in iter_trainable_traces(episodes):
            if trace.info.get("segment", 0) == 0 and not any(e is episode for e, _ in main):
                main.append((episode, trace))
            else:
                assign_advantages(trace, 0.0)  # not a node's segment (none expected): no credit
        if not main:
            return
        node = ((main[0][0].task.data.info or {}).get("tree") or {})
        start = node.get("start_score")
        results: dict[str, tuple[float | None, dict]] = {}
        pending = list(main)
        if start is None:
            # A fresh node: judge the first branch's start (and end), then share that start.
            while pending and start is None:
                episode, trace = pending.pop(0)
                results[trace.id] = await self.scorer(trace, **self.scorer_args)
                start = (results[trace.id][1] or {}).get("start")
        if start is not None:
            self._share_start([trace for _, trace in pending], start)
        scored = await asyncio.gather(*(self.scorer(trace, **self.scorer_args) for _, trace in pending))
        results.update({trace.id: r for (_, trace), r in zip(pending, scored)})
        rewards = [r for r, _ in results.values() if r is not None]
        baseline = sum(rewards) / len(rewards) if rewards else None
        for episode, trace in main:
            reward, info = results[trace.id]
            advantage = 0.0 if reward is None or baseline is None else reward - baseline
            assign_advantages(trace, advantage)
            trace.info["score"] = {"reward": reward, "advantage": advantage, "baseline": baseline,
                                   "end": (info or {}).get("end"), "start": (info or {}).get("start")}
            self._log(episode, trace, reward, advantage, baseline, info)

    def _share_start(self, traces: list[vf.Trace], start: dict) -> None:
        """Hand ``start`` to the scorer as these traces' start score (the Q/V scorer takes a
        segment's start from ``_RUN.handoff[trace id]`` when one is there)."""
        module = importlib.import_module(self.scorer.__module__)
        run = getattr(module, "_RUN", None)
        if run is None or not hasattr(run, "handoff"):
            return
        loop = asyncio.get_running_loop()
        for trace in traces:
            future = loop.create_future()
            future.set_result(dict(start))
            run.handoff[trace.id] = future

    def _log(self, episode: vf.Episode, trace: vf.Trace, reward, advantage, baseline, info) -> None:
        if self.log_path is None:
            return
        try:
            policy = train_work(episode).policy
            version = policy.start if policy is not None else None
        except ValueError:
            version = None
        node = ((episode.task.data.info or {}).get("tree") or {}) if self.group_baseline else {}
        record = {"time": time.time(), "episode": episode.id, "trace": trace.id, "policy": version,
                  "reward": reward, "advantage": advantage, "baseline": baseline,
                  "tokens": sum(sum(n.mask) for n in trace.nodes), "info": info,
                  **({"node": {k: node.get(k) for k in ("id", "lineage", "depth", "seed")}} if node else {})}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

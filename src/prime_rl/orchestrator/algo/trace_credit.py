from __future__ import annotations

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
    """Per-trace credit from an env's own async scorer: every sampled token of a trace
    carries the trace's reward minus a running baseline (an exponential average of the
    rewards scored before it, so a trace never sets its own baseline). For segments
    shipped one at a time (``stream_segments``), where no group shares a start. A trace
    the scorer cannot score (it returns None) carries 0. With ``log_path`` set, each
    scored trace appends a JSON line: its reward, advantage, baseline and the scorer's info."""

    def __init__(self, config: TraceCreditAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.scorer = load_scorer(config.scorer)
        self.scorer_args = config.scorer_args
        self.decay = config.baseline_decay
        self.log_path = config.log_path
        self.baseline: float | None = None

    async def score_episode(self, episode: vf.Episode) -> None:
        for _, trace in iter_trainable_traces([episode]):
            reward, info = await self.scorer(trace, **self.scorer_args)
            baseline = self.baseline
            if reward is None:
                advantage = 0.0
            else:
                advantage = reward - (baseline if baseline is not None else reward)
                self.baseline = reward if baseline is None else self.decay * baseline + (1 - self.decay) * reward
            assign_advantages(trace, advantage)
            if self.log_path is None:
                continue
            try:
                policy = train_work(episode).policy
                version = policy.start if policy is not None else None
            except ValueError:
                version = None
            record = {"time": time.time(), "episode": episode.id, "trace": trace.id, "policy": version,
                      "reward": reward, "advantage": advantage, "baseline": baseline,
                      "tokens": sum(sum(node.mask) for node in trace.nodes), "info": info}
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

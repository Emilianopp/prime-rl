from __future__ import annotations

import importlib
import json
import time
from collections import Counter
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TurnCreditAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.orchestrator.utils import train_work

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def load_scorer(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


class TurnCreditAlgorithm(Algorithm):
    """Per-reply credit: every sampled token of a reply carries the reward the env's
    scorer gives that reply (0 when it scores none). Rollout-local, so a group of one
    is enough. With ``log_path`` set, each scored episode appends a JSON line: the
    numbers a run monitor needs (forfeits per reply, sampled tokens, game reward,
    policy version) without reading the traces, and each scored reply's reasons in
    play order (``turns``), so a monitor can look at one stretch of the game only."""

    def __init__(self, config: TurnCreditAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.scorer = load_scorer(config.scorer)
        self.scorer_args = config.scorer_args
        self.log_path = config.log_path

    async def score_episode(self, episode: vf.Episode) -> None:
        replies = scored = tokens = 0
        reasons: Counter[str] = Counter()
        turns: list[list[str]] = []
        credit = 0.0
        for _, trace in iter_trainable_traces([episode]):
            scores = {score.node: score for score in self.scorer(trace, **self.scorer_args)}
            index = {id(node): i for i, node in enumerate(trace.nodes)}
            values: list[float] = []
            for node in trainable_nodes(trace):
                score = scores.get(index[id(node)])
                values += [score.reward if score else 0.0] * sum(node.mask)
            assign_advantages(trace, values)
            for i, score in scores.items():
                replies += 1
                tokens += sum(trace.nodes[i].mask)
                scored += score.reward != 0.0
                credit += score.reward
                reasons.update(score.reasons)
                turns.append(list(score.reasons))
        if self.log_path is None:
            return
        try:
            policy = train_work(episode).policy
            version = policy.start if policy is not None else None
        except ValueError:
            version = None
        record = {
            "time": time.time(),
            "episode": episode.id,
            "policy": version,
            "replies": replies,
            "scored": scored,
            "tokens": tokens,
            "credit": credit,
            "reward": sum(trace.reward or 0.0 for trace in episode.traces),
            "reasons": dict(reasons),
            "turns": turns,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

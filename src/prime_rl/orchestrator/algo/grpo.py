from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import apply_node_penalties, assign_advantages, trainable_nodes

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation
        self.advantage_mode = config.advantage_mode
        self.term_std_floor = dict(config.term_std_floor)
        self.default_term_std_floor = config.default_term_std_floor

    def per_term_contributions(self, traces: list[vf.Trace]) -> list[dict[str, float]]:
        """``advantage_mode = "per_term"``: per trace, each reward term's w_t (r_t - mean) / max(std, floor_t)."""
        contributions: list[dict[str, float]] = [{} for _ in traces]
        names = sorted({name for trace in traces for name, r in trace.rewards.items() if r is not None})
        for name in names:
            present = [(i, trace.rewards[name]) for i, trace in enumerate(traces) if trace.rewards.get(name) is not None]
            if len(present) < 2:
                continue
            scores = [float(r.score) for _, r in present]
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            denom = max(std, self.term_std_floor.get(name, self.default_term_std_floor))
            for (i, r), s in zip(present, scores):
                contributions[i][name] = float(r.weight) * (s - mean) / denom
        return contributions

    def per_term_advantages(self, traces: list[vf.Trace]) -> list[float]:
        """The per-term advantages summed per trace (the scalar form, no routing)."""
        return [sum(c.values()) for c in self.per_term_contributions(traces)]

    @staticmethod
    def route_terms(trace: vf.Trace, contribution: dict[str, float]) -> float | list[float]:
        """An env's ``trace.info["term_routing"] = {term: [node index, ...]}`` sends that term's
        advantage to those sampled nodes only and every unrouted term's to the other sampled nodes
        (crafter_env.taskset.request_term_routing: the notes reward trains the notes turn, progress
        trains the action turns). Returns one value per sampled token in trainable-node order, or the
        plain sum when the trace routes nothing that applies."""
        routes = (trace.info or {}).get("term_routing") if isinstance(trace.info, dict) else None
        total = sum(contribution.values())
        if not isinstance(routes, dict) or not any(name in contribution for name in routes):
            return total
        node_index = {id(node): i for i, node in enumerate(trace.nodes)}
        routed = {name: set(idx) for name, idx in routes.items() if name in contribution}
        unrouted = sum(v for name, v in contribution.items() if name not in routed)
        values: list[float] = []
        for node in trainable_nodes(trace):
            i = node_index[id(node)]
            here = [contribution[name] for name, idx in routed.items() if i in idx]
            advantage = sum(here) if here else unrouted
            values.extend([advantage] * sum(node.mask))
        return values

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if self.advantage_mode == "per_term":
            for trace, contribution in zip(traces, self.per_term_contributions(traces), strict=True):
                assign_advantages(trace, self.route_terms(trace, contribution))
                apply_node_penalties(trace)
            self._apply_loss_aggregation(traces)
            return
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            shaped_rewards = rewards
        else:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor(
                [sum(sum(node.mask) for node in trainable_nodes(trace)) for trace in traces], dtype=rewards.dtype
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
            apply_node_penalties(trace)
        self._apply_loss_aggregation(traces)

    def _apply_loss_aggregation(self, traces: list[vf.Trace]) -> None:
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}

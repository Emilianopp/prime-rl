import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    MaxRLAlgoConfig,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples


def _build_episode(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback).
    """
    obs_lengths = obs_lengths or []
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0

    def _take(n: int) -> list[int]:
        nonlocal next_token
        ids = list(range(next_token, next_token + n))
        next_token += n
        return ids

    # Leading user prompt (never trainable).
    prompt_ids = _take(1)
    nodes.append(
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
            sampled=False,
            parent=parent,
        )
    )
    parent = len(nodes) - 1

    # Trace token counts are usage-based, so carry provider usage on the final turn's call:
    # every model-generated token as completion, the leading prompt + tool observations as the
    # fed-in context (num_input_tokens = num_total_tokens - num_output_tokens).
    output_tokens = sum(sampled_lengths)
    input_tokens = 1 + sum(obs_lengths)
    calls: list[vf.ModelCall] = []

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        is_last = i == len(sampled_lengths) - 1
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="a"),
                token_ids=ids,
                mask=[True] * n_sampled,
                logprobs=[-0.1] * n_sampled,
                sampled=True,
                parent=parent,
            )
        )
        parent = len(nodes) - 1
        if is_last:
            calls.append(
                vf.ModelCall(
                    node=parent,
                    usage=vf.Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens),
                )
            )
        if i < len(obs_lengths):
            obs_ids = _take(obs_lengths[i])
            nodes.append(
                vf.MessageNode(
                    message=vf.ToolMessage(content="t", tool_call_id="x"),
                    token_ids=obs_ids,
                    mask=[False] * obs_lengths[i],
                    logprobs=[0.0] * obs_lengths[i],
                    sampled=False,
                    parent=parent,
                )
            )
            parent = len(nodes) - 1

    trace = vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=calls,
        rewards={"reward": vf.Reward(score=reward)},
        metrics=metrics or {},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id="group"),
        traces=[trace],
        ok=True,
    )
    return episode


def _make_episode(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_episode(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[vf.Episode]:
    """Build one group of training traces from 1D arrays of rewards/lengths/turns —
    exactly what ``score_group`` sees."""
    episodes = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        episodes.append(_make_episode(float(reward), cl, nt))
    return episodes


def _scalar(episode: vf.Episode) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    for node in episode.traces[0].nodes:
        if node.advantages:
            return node.advantages[0]
    raise AssertionError("episode has no trainable token")


def _grpo(group: list[vf.Episode], length_penalty=None, length_weighted_baseline=False) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(length_penalty=length_penalty, length_weighted_baseline=length_weighted_baseline), clients=None
    )
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


def _max_rl(group: list[vf.Episode]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), clients=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


# --------------------------------------------------------------------------
# GRPO / MaxRL: group-relative credit, assigned in score_group.
# --------------------------------------------------------------------------


def test_grpo_plain_mean():
    advs = _grpo(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))
    assert len(advs) == 3
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_singleton_group_is_zero():
    # A group of size 1 has reward == mean, so its advantage is 0.
    assert _grpo([_build_episode(0.7, sampled_lengths=[2])]) == pytest.approx([0.0], abs=1e-6)


def test_grpo_length_weighted_baseline():
    # L = [10 + 20 (two turns, observation excluded), 10]: b = (30 * 1 + 10 * 0) / 40 = 0.75
    group = [
        _build_episode(1.0, sampled_lengths=[10, 20], obs_lengths=[5]),
        _build_episode(0.0, sampled_lengths=[10]),
    ]
    advs = _grpo(group, length_weighted_baseline=True)
    assert advs == pytest.approx([0.25, -0.75])
    # per-token advantage is zero-mean across the group's trainable tokens
    assert 30 * advs[0] + 10 * advs[1] == pytest.approx(0.0, abs=1e-6)
    # with a length penalty, the weighted baseline applies to the shaped rewards:
    # penalty = mean(r) * 0.5 * L / max(L) = [0.25, 1/12], shaped = [0.75, -1/12], b = (30 * 0.75 - 10 / 12) / 40
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.5, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    group = [
        _build_episode(1.0, sampled_lengths=[10, 20], obs_lengths=[5]),
        _build_episode(0.0, sampled_lengths=[10]),
    ]
    advs = _grpo(group, length_penalty=cfg, length_weighted_baseline=True)
    assert advs == pytest.approx([0.75 - 65 / 120, -1 / 12 - 65 / 120])


def test_max_rl_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    assert _max_rl(_make_group(rewards=[1.0, 0.0, 0.0, 0.0])) == pytest.approx([3.0, -1.0, -1.0, -1.0])
    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert _max_rl(_make_group(rewards=[0.0, 0.0])) == pytest.approx([0.0, 0.0])
    # ... and all-success groups center to zero like GRPO
    assert _max_rl(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


def test_grpo_prompt_loss_aggregation_weights_sum_to_one_per_group():
    """Each group's rl weights total 1, spread as 1/T_q over its trainable tokens, whatever its length."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(loss_aggregation="prompt"), clients=None)
    for lengths in ([10, 30], [100, 300]):
        group = _make_group(rewards=[1.0, 0.0], completion_lengths=lengths)
        asyncio.run(algo.score_group(group))
        weights = [w for episode in group for sample in trace_to_samples(episode.traces[0]) for w in sample.rl_weights]
        assert set(weights) == {0.0, 1.0 / sum(lengths)}
        assert sum(weights) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# GRPO linear length penalty: pass_rate-scaled penalty before the baseline.
# --------------------------------------------------------------------------


def test_linear_equal_lengths_reduce_to_plain_grpo():
    """Equal completion length and turns → every rollout takes the same penalty
    fraction, so subtracting it leaves the centered advantages unchanged."""
    penalized = _grpo(
        _make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]),
        length_penalty=LinearLengthPenaltyConfig(),
    )
    plain = _grpo(_make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]))
    assert penalized == pytest.approx(plain, abs=1e-6)


def test_linear_completion_term_penalizes_longer():
    """With only the completion term, longer completions get a larger penalty and a
    lower advantage; advantages stay zero-mean."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    advs = _grpo(_make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30]), length_penalty=cfg)
    assert advs[0] > advs[1] > advs[2]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_context_term_penalizes_more_context():
    """The context term penalizes non-completion (prompt / tool-response) tokens: at
    equal completion length, more context tokens yields a lower advantage."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.25, num_turns_weight=0.0)
    group = [
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[]),
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[100]),
    ]
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(length_penalty=cfg), clients=None).score_group(group))
    advs = [_scalar(episode) for episode in group]
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_turns_term_penalizes_more_turns():
    """The turns term penalizes higher turn counts at equal token lengths."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.0, num_turns_weight=0.25)
    advs = _grpo(
        _make_group(rewards=[1.0, 1.0], completion_lengths=[100, 100], num_turns=[1, 4]),
        length_penalty=cfg,
    )
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    episode = _build_episode(0.0, sampled_lengths=[2])
    trace = episode.traces[0]
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.7]


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    episode = _build_episode(0.0, sampled_lengths=[1], obs_lengths=[1])
    trace = episode.traces[0]
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.0]


def test_assign_advantages_rejects_misaligned():
    episode = _build_episode(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        assign_advantages(episode.traces[0], [0.5])


# --------------------------------------------------------------------------
# GRPO advantage_mode = "per_term": each reward term standardized in the group.
# --------------------------------------------------------------------------


def _termed_group(terms: list[dict[str, tuple[float, float]]]) -> list[vf.Episode]:
    group = []
    for t in terms:
        episode = _build_episode(0.0, sampled_lengths=[2])
        episode.traces[0].rewards = {name: vf.Reward(score=s, weight=w) for name, (s, w) in t.items()}
        group.append(episode)
    return group


def _per_term(group, **kwargs) -> list[float]:
    algo = GRPOAlgorithm(GRPOAlgoConfig(advantage_mode="per_term", **kwargs), clients=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


def test_per_term_standardizes_each_term_then_weights():
    # progress 0/2 (std 1), notes 0.4/0.6 (std 0.1): both terms count as one std each.
    group = _termed_group([{"progress": (0.0, 1.0), "notes": (0.6, 0.5)}, {"progress": (2.0, 1.0), "notes": (0.4, 0.5)}])
    advs = _per_term(group, term_std_floor={"progress": 0.5}, default_term_std_floor=0.05)
    assert advs == pytest.approx([-1.0 + 0.5, 1.0 - 0.5])


def test_per_term_floor_caps_tiny_spreads_and_flat_terms_add_nothing():
    group = _termed_group([{"progress": (1.0, 1.0), "notes": (0.50, 0.5)}, {"progress": (1.0, 1.0), "notes": (0.52, 0.5)}])
    advs = _per_term(group)  # notes std 0.01 < floor 0.1
    assert advs == pytest.approx([0.5 * -0.01 / 0.1, 0.5 * 0.01 / 0.1])


def test_per_term_missing_term_gets_zero_and_stats_use_present_only():
    group = _termed_group([{"progress": (0.0, 1.0), "adherence": (0.2, 0.25)}, {"progress": (0.0, 1.0), "adherence": (0.8, 0.25)},
                           {"progress": (0.0, 1.0)}])
    advs = _per_term(group)
    assert advs[2] == 0.0 and advs[0] == pytest.approx(-0.25) and advs[1] == pytest.approx(0.25)


def test_per_term_rejects_length_penalty():
    with pytest.raises(ValueError):
        GRPOAlgoConfig(advantage_mode="per_term", length_weighted_baseline=True)


def test_per_term_routing_sends_each_term_to_its_nodes():
    """``trace.info["term_routing"] = {"notes": [node index]}``: the notes term's advantage lands on
    the notes turn's tokens only, the other terms' on the other sampled tokens (crafter_env's
    request_term_routing)."""
    group = _termed_group([{"progress": (0.0, 1.0), "notes": (0.6, 0.5)}, {"progress": (2.0, 1.0), "notes": (0.4, 0.5)}])
    for episode in group:  # a second sampled turn (the notes turn) after an observation
        trace = episode.traces[0]
        last = trace.nodes[-1]
        obs_ids = [900]
        trace.nodes.append(vf.MessageNode(message=vf.UserMessage(content="END OF SEGMENT"), token_ids=obs_ids,
                                          mask=[False], logprobs=[0.0], sampled=False, parent=len(trace.nodes) - 1))
        notes_ids = [901, 902, 903]
        trace.nodes.append(vf.MessageNode(message=vf.AssistantMessage(content="<notes>x</notes>"), token_ids=notes_ids,
                                          mask=[True] * 3, logprobs=[0.0] * 3, sampled=True, parent=len(trace.nodes) - 1))
        trace.info = {"term_routing": {"notes": [len(trace.nodes) - 1]}}
        assert last is not trace.nodes[-1]
    algo = GRPOAlgorithm(GRPOAlgoConfig(advantage_mode="per_term", term_std_floor={"progress": 0.5}, default_term_std_floor=0.05), clients=None)
    asyncio.run(algo.score_group(group))
    first = [node for node in group[0].traces[0].nodes if any(node.mask)]
    assert first[0].advantages == pytest.approx([-1.0, -1.0])  # the action turn: progress only
    assert first[-1].advantages == pytest.approx([0.5, 0.5, 0.5])  # the notes turn: notes only
    second = [node for node in group[1].traces[0].nodes if any(node.mask)]
    assert second[0].advantages == pytest.approx([1.0, 1.0]) and second[-1].advantages == pytest.approx([-0.5] * 3)
    plain = _termed_group([{"progress": (0.0, 1.0), "notes": (0.6, 0.5)}, {"progress": (2.0, 1.0), "notes": (0.4, 0.5)}])
    assert _per_term(plain, term_std_floor={"progress": 0.5}, default_term_std_floor=0.05) == pytest.approx([-0.5, 0.5])  # unrouted: the sum


def test_route_terms_to_no_node_drops_that_term():
    from types import SimpleNamespace

    from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm

    nodes = [SimpleNamespace(sampled=True, mask=[1, 1]), SimpleNamespace(sampled=True, mask=[1])]
    trace = SimpleNamespace(info={"term_routing": {"notes": []}}, nodes=nodes)
    import prime_rl.orchestrator.algo.grpo as grpo
    original = grpo.trainable_nodes
    grpo.trainable_nodes = lambda t: [n for n in t.nodes if n.sampled]
    try:
        values = GRPOAlgorithm.route_terms(trace, {"progress": 0.5, "state": -0.2, "notes": 3.0})
    finally:
        grpo.trainable_nodes = original
    assert values == [0.3, 0.3, 0.3]  # notes (3.0) goes to no token; every token gets progress + state

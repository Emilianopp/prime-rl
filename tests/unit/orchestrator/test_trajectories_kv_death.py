"""A KV-Streams trace's per-token evictions reach its training sample as kv_death."""

import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.types import TurnTokens

from prime_rl.orchestrator.batch import prepare_sample
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.trainer.rl.data import DataLoader


def _turn(trace, messages, prompt_ids, completion_ids, spans, eviction=None):
    graph.prepare_turn(trace, messages).commit(
        vf.Response(
            id="r",
            created=0,
            model="t",
            message=vf.AssistantMessage(content=f"a{len(messages)}"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                completion_logprobs=[-0.5] * len(completion_ids),
                message_spans=spans,
                eviction=eviction,
            ),
        )
    )


def _trace(evict: bool) -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    u1, u2 = vf.UserMessage(content="u1"), vf.UserMessage(content="u2")
    _turn(trace, [u1], [10, 11, 12], [20, 21], [(0, 2)])
    _turn(
        trace,
        [u1, vf.AssistantMessage(content="a1"), u2],
        [10, 11, 12, 20, 21, 30, 31],
        [40],
        [(0, 2), None, (5, 7)],
        eviction={"spans": [[2, 5]], "at": 5} if evict else None,
    )
    return trace


def test_trace_to_samples_carries_kv_death():
    (sample,) = trace_to_samples(_trace(evict=True), env_name="kv")
    assert sample.token_ids == [10, 11, 12, 20, 21, 30, 31, 40]
    assert sample.kv_death == [-1, -1, 5, 5, 5, -1, -1, -1]


def test_full_context_trace_has_no_kv_death():
    (sample,) = trace_to_samples(_trace(evict=False), env_name="kv")
    assert sample.kv_death is None


def test_trainer_refuses_kv_death_until_flex_attention():
    (sample,) = trace_to_samples(_trace(evict=True), env_name="kv")
    sample.temperatures = [1.0] * len(sample.token_ids)
    sample.advantages = [0.0] * len(sample.token_ids)
    micro_batch = prepare_sample(sample, seq_len=64)
    with pytest.raises(NotImplementedError, match="flex attention"):
        DataLoader._micro_batch_to_tensor(None, micro_batch)

import asyncio
import re

import pytest

from prime_rl.inference.sglang.server import (
    GenerateRequestError,
    SessionTable,
    TraceAffinity,
    TurnWindow,
    generate_sglang,
)


class FakeEngine:
    """Records SGLang calls; each generate returns two fixed completion tokens."""

    def __init__(self, completion=(90, 91), abort=False):
        self.completion = list(completion)
        self.abort = abort
        self.calls: list[tuple] = []
        self.open: set[str] = set()
        self.gate: asyncio.Event | None = None
        self.ranks: list[int | None] = []
        self.evictions: list[dict | None] = []

    async def generate(
        self,
        input_ids,
        sampling_params,
        *,
        request=None,
        session_id=None,
        cache_salt=None,
        priority=None,
        routed_dp_rank=None,
        kv_eviction=None,
    ):
        self.calls.append(("generate", session_id, list(input_ids)))
        self.ranks.append(routed_dp_rank)
        self.evictions.append(kv_eviction)
        if self.gate is not None:
            await self.gate.wait()
        finish = {"type": "abort", "message": "boom"} if self.abort else {"type": "stop"}
        state = {"state_id": f"s{len(self.evictions)}"} if kv_eviction is not None else None
        return {
            "output_ids": list(self.completion),
            "meta_info": {
                "id": "r",
                "kv_eviction": {"cache_state": state} if state else None,
                "finish_reason": finish,
                "output_token_logprobs": [(-0.1, t, None) for t in self.completion],
            },
        }

    async def open_session(self, session_id, timeout):
        self.calls.append(("open", session_id))
        self.open.add(session_id)
        return True

    async def close_session(self, session_id):
        self.calls.append(("close", session_id))
        self.open.discard(session_id)


def run(coro):
    return asyncio.run(coro)


def test_append_only_turns_send_only_the_new_tail():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        r1 = await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")
        # Next prompt = previous prompt + completion + new user tokens.
        r2 = await generate_sglang(engine, [1, 2, 3, 90, 91, 7, 8], {}, sessions=table, trace_id="t")
        return engine, r1, r2

    engine, r1, r2 = run(scenario())
    assert engine.calls == [
        ("open", "t.0"),
        ("generate", "t.0", [1, 2, 3]),
        ("generate", "t.0", [7, 8]),
    ]
    # The response always carries the full prompt, as without a session.
    assert r2["prompt_token_ids"] == [1, 2, 3, 90, 91, 7, 8]
    assert r1["choices"][0]["token_ids"] == [90, 91]


def test_rewritten_history_reopens_the_session_on_the_full_prompt():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")
        await generate_sglang(engine, [1, 5, 6], {}, sessions=table, trace_id="t")
        return engine

    engine = run(scenario())
    assert engine.calls[-3:] == [("close", "t.0"), ("open", "t.1"), ("generate", "t.1", [1, 5, 6])]
    assert engine.open == {"t.1"}


def test_concurrent_branch_runs_stateless():
    async def scenario():
        engine = FakeEngine()
        engine.gate = asyncio.Event()
        table = SessionTable(engine, timeout=60)
        first = asyncio.ensure_future(generate_sglang(engine, [1, 2], {}, sessions=table, trace_id="t"))
        await asyncio.sleep(0)
        second = asyncio.ensure_future(generate_sglang(engine, [9, 9], {}, sessions=table, trace_id="t"))
        await asyncio.sleep(0)
        engine.gate.set()
        await asyncio.gather(first, second)
        return engine

    engine = run(scenario())
    assert ("generate", None, [9, 9]) in engine.calls
    assert ("generate", "t.0", [1, 2]) in engine.calls


def test_abort_closes_the_session_and_raises():
    async def scenario():
        engine = FakeEngine(abort=True)
        table = SessionTable(engine, timeout=60)
        with pytest.raises(GenerateRequestError):
            await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")
        engine.abort = False
        await generate_sglang(engine, [1, 2, 3, 90, 91, 4], {}, sessions=table, trace_id="t")
        return engine

    engine = run(scenario())
    # After the abort the next call starts a fresh session on the full prompt.
    assert engine.calls[-2:] == [("open", "t.1"), ("generate", "t.1", [1, 2, 3, 90, 91, 4])]


def test_finish_closes_the_trace_session():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")
        await table.finish("t")
        await table.finish("never-opened")
        return engine

    engine = run(scenario())
    assert engine.open == set()
    assert engine.calls[-1] == ("close", "t.0")


def test_without_a_trace_id_generation_is_stateless():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id=None)
        return engine

    assert run(scenario()).calls == [("generate", None, [1, 2, 3])]


def test_traces_stick_to_the_least_loaded_dp_rank():
    affinity = TraceAffinity(dp_size=2)
    assert [affinity.rank(t) for t in ("a", "b", "a", "c")] == [0, 1, 0, 0]
    affinity.release("a")
    affinity.release("c")
    assert affinity.rank("d") == 0  # rank 0 is empty again, rank 1 still holds "b"
    assert affinity.rank(None) is None  # no trace id: SGLang balances


def test_session_turns_carry_the_trace_rank():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        affinity = TraceAffinity(dp_size=2)
        await generate_sglang(engine, [1], {}, sessions=table, affinity=affinity, trace_id="a")
        await generate_sglang(engine, [7], {}, sessions=table, affinity=affinity, trace_id="b")
        await generate_sglang(engine, [1, 90, 91, 2], {}, sessions=table, affinity=affinity, trace_id="a")
        return engine

    assert run(scenario()).ranks == [0, 1, 0]


@pytest.mark.parametrize(
    "max_turns,stride,expected",
    [
        # completed turns 0..10 -> live turns, as kv_eviction.truncation.kv_eviction_live_turns
        (8, 1, [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7]),
        (8, 2, [0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 6]),
        (4, 4, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]),
    ],
)
def test_turn_window_live_turns(max_turns, stride, expected):
    window = TurnWindow(max_turns=max_turns, stride=stride, message_end_id=7)
    assert [window.live_turns(n) for n in range(11)] == expected


E = 7  # message end
SYSTEM, USER, GEN = [1, 2, E], [3, E], [4]  # system message, a user message, the generation prompt
BRIDGE = [5]  # the newline between a reply's end and the next message


def test_turn_window_layout_protects_the_first_message():
    window = TurnWindow(max_turns=2, stride=1, message_end_id=E)
    prompt = SYSTEM + USER + GEN
    assert window.layout(prompt) == (3, 0)
    for _ in range(2):  # two completed turns: [user, reply] pairs
        prompt += [90, E] + BRIDGE + USER + GEN
    # The oldest turn (user [3, E] + generation prompt + reply [90, E]) is dead.
    assert window.layout(prompt) == (3, 5)


def test_eviction_retires_the_oldest_turn_each_call():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        window = TurnWindow(max_turns=2, stride=1, message_end_id=E)
        table = SessionTable(engine, timeout=60, eviction=window)
        prompt = SYSTEM + USER + GEN
        for _ in range(4):
            await table.generate("t", prompt, {})
            prompt = prompt + [90, E] + BRIDGE + USER + GEN
        return engine, table

    engine, table = run(scenario())
    calls = engine.evictions
    assert [c["call_index"] for c in calls] == [0, 1, 2, 3]
    assert [c["expected_state_id"] for c in calls] == [None, "s1", "s2", "s3"]
    assert all(c["protected_prefix_len"] == 3 and c["cache_id"] == "t.0" for c in calls)
    # Turns complete at calls 1, 2, 3; from the second one on, the oldest live
    # turn dies each call. Survivor coordinates: evicted turns are gone, so the
    # next retired turn always starts right after the protected prefix.
    assert [c["evict_spans"] for c in calls] == [[], [], [[3, 8]], [[3, 9]]]
    assert table.stats["evicting_calls"] == 2 and table.stats["evicted_tokens"] == 11


def test_aborted_eviction_call_is_retried_on_a_fresh_session():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt = SYSTEM + USER + GEN
        await table.generate("t", prompt, {})
        engine.abort = True
        prompt = prompt + [90, E] + BRIDGE + USER + GEN
        await table.generate("t", prompt, {})
        return engine, table

    engine, table = run(scenario())
    # The aborted call on t#0 is retried once, as call 0 of a fresh session with
    # the full prompt; that abort is returned.
    assert [(c["cache_id"], c["call_index"]) for c in engine.evictions] == [("t.0", 0), ("t.0", 1), ("t.1", 0)]
    assert engine.calls[-2:] == [
        ("generate", "t.1", SYSTEM + USER + GEN + [90, E] + BRIDGE + USER + GEN),
        ("close", "t.1"),
    ]
    assert table.stats["failed_calls"] == 2 and table.stats["reopened"] == 1


def test_session_and_call_ids_match_sglang_id_pattern():
    sglang_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")

    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt = SYSTEM + USER + GEN
        await table.generate("trace/with#odd chars", prompt, {})
        await table.generate("trace/with#odd chars", prompt + [90, E] + BRIDGE + USER + GEN, {})
        return engine

    for call in run(scenario()).evictions:
        assert sglang_id.match(call["cache_id"]) and sglang_id.match(call["call_id"]), call

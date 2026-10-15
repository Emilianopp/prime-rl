import asyncio

import pytest

from prime_rl.inference.sglang.server import GenerateRequestError, SessionTable, TraceAffinity, generate_sglang


class FakeEngine:
    """Records SGLang calls; each generate returns two fixed completion tokens."""

    def __init__(self, completion=(90, 91), abort=False):
        self.completion = list(completion)
        self.abort = abort
        self.calls: list[tuple] = []
        self.open: set[str] = set()
        self.gate: asyncio.Event | None = None
        self.ranks: list[int | None] = []

    async def generate(
        self, input_ids, sampling_params, *, session_id=None, cache_salt=None, priority=None, routed_dp_rank=None
    ):
        self.calls.append(("generate", session_id, list(input_ids)))
        self.ranks.append(routed_dp_rank)
        if self.gate is not None:
            await self.gate.wait()
        finish = {"type": "abort", "message": "boom"} if self.abort else {"type": "stop"}
        return {
            "output_ids": list(self.completion),
            "meta_info": {
                "id": "r",
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
        ("open", "t#0"),
        ("generate", "t#0", [1, 2, 3]),
        ("generate", "t#0", [7, 8]),
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
    assert engine.calls[-3:] == [("close", "t#0"), ("open", "t#1"), ("generate", "t#1", [1, 5, 6])]
    assert engine.open == {"t#1"}


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
    assert ("generate", "t#0", [1, 2]) in engine.calls


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
    assert engine.calls[-2:] == [("open", "t#1"), ("generate", "t#1", [1, 2, 3, 90, 91, 4])]


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
    assert engine.calls[-1] == ("close", "t#0")


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

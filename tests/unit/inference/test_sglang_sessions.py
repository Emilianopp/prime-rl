import asyncio
import re

import pytest

from prime_rl.inference.sglang.eviction import occurrences
from prime_rl.inference.sglang.model_evictions import qwen35
from prime_rl.inference.sglang.server import (
    GenerateRequestError,
    SessionDesyncError,
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
        self.rids: list[str | None] = []
        self.aborted: set[str] = set()

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
        rid=None,
    ):
        self.calls.append(("generate", session_id, list(input_ids)))
        self.ranks.append(routed_dp_rank)
        self.evictions.append(kv_eviction)
        self.rids.append(rid)
        if self.gate is not None:
            await self.gate.wait()
        aborted = self.abort or rid in self.aborted
        finish = {"type": "abort", "message": "boom"} if aborted else {"type": "stop"}
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

    def abort_request(self, rid):
        self.calls.append(("abort", rid))
        self.aborted.add(rid)
        if self.gate is not None:
            self.gate.set()


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


def test_finish_aborts_the_inflight_request_and_never_reopens():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt = SYSTEM + USER + GEN
        await table.generate("t", prompt, {})
        engine.gate = asyncio.Event()  # the next turn sits in the engine (e.g. queued on a full KV pool)
        turn = asyncio.create_task(table.generate("t", prompt + [90, E] + BRIDGE + USER + GEN, {}))
        await asyncio.sleep(0)
        await table.finish("t")  # the orchestrator cancelled the episode
        ret = await turn
        return engine, table, ret

    engine, table, ret = run(scenario())
    assert ("abort", "t.0.c1") in engine.calls  # a kv_eviction request's rid is its call id
    assert ret["meta_info"]["finish_reason"]["type"] == "abort"
    assert [c for c in engine.calls if c[0] == "open"] == [("open", "t.0")]  # no retry on a fresh session
    assert engine.open == set() and table.stats["failed_calls"] == 0


def test_finish_while_the_session_opens_closes_it():
    async def scenario():
        engine = FakeEngine()
        gate = asyncio.Event()
        open_session = engine.open_session

        async def slow_open(session_id, timeout):  # e.g. the scheduler is overloaded
            await gate.wait()
            return await open_session(session_id, timeout)

        engine.open_session = slow_open
        table = SessionTable(engine, timeout=60)
        turn = asyncio.create_task(table.generate("t", [1, 2, 3], {}))
        await asyncio.sleep(0)
        await table.finish("t")  # no session or request yet: nothing to abort or close
        gate.set()
        with pytest.raises(GenerateRequestError):
            await turn
        return engine

    engine = run(scenario())
    assert engine.open == set()  # not left open until the session timeout
    assert not [c for c in engine.calls if c[0] == "generate"]  # and nothing generated on it


def test_finish_without_a_request_in_flight_only_closes():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        await table.generate("t", [1, 2, 3], {})
        await table.finish("t")
        return engine

    engine = run(scenario())
    assert not [c for c in engine.calls if c[0] == "abort"]
    assert engine.calls[-1] == ("close", "t.0") and engine.rids == ["t.0.r0"]


def test_evicting_calls_report_chronological_spans_and_the_first_unseeing_position():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt, choices = SYSTEM + USER + GEN, []
        for _ in range(4):
            response = await generate_sglang(engine, prompt, {}, sessions=table, trace_id="t")
            assert response["prompt_token_ids"] == prompt  # always the full chronological prompt
            choices.append(response["choices"][0])
            prompt = prompt + [90, E] + BRIDGE + USER + GEN
        return choices

    choices = run(scenario())
    # Turn 1 is chronological [3, 8) (user, generation prompt, reply), turn 2 [8, 14).
    # Each dies from the previous call's last sampled token on: resident 14 -> 13, 20 -> 19.
    assert [c.get("eviction") for c in choices] == [
        None,
        None,
        {"spans": [[3, 8]], "at": 13},
        {"spans": [[8, 14]], "at": 19},
    ]


OPEN, CLOSE, NL, TOOL = 60, 61, 62, 66  # <think>, </think>, its blank line, <tool_response>
THINK = [OPEN, NL, CLOSE, NL]  # the empty think block the generation prompt prefills
THINK_GEN = [4, *THINK]
TOOL_RESULT = [3, TOOL, E]  # rendered as a user message that is not a query


def _qwen():
    return qwen35.Plan(empty_think_ids=THINK, user_ids=USER[:1], tool_response_ids=[TOOL])


def _think_run(max_turns: int, calls: int, user=USER, think=THINK):
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        window = TurnWindow(max_turns=max_turns, stride=1, message_end_id=E)
        table = SessionTable(engine, timeout=60, eviction=window, plans=[_qwen()])
        prompt, choices = SYSTEM + USER + [4, *think], []
        for _ in range(calls):
            response = await generate_sglang(engine, prompt, {}, sessions=table, trace_id="t")
            choices.append(response["choices"][0])
            prompt = prompt + [90, E] + BRIDGE + user + [4, *think]
        return engine, choices

    return run(scenario())


def test_an_earlier_turns_think_block_dies_when_the_next_turn_arrives():
    engine, choices = _think_run(max_turns=100, calls=3)
    # Turn 1's block is chronological [6, 10), turn 2's [16, 20); each dies at the next
    # call, while the call's own block (in its new tail) stays visible.
    assert [c.get("eviction") for c in choices] == [
        None,
        {"spans": [[6, 10]], "at": 11},
        {"spans": [[16, 20]], "at": 21},
    ]
    # In survivor coordinates turn 2's block sits 4 tokens lower: turn 1's block is gone.
    assert [c["evict_spans"] for c in engine.evictions] == [[], [[6, 10]], [[12, 16]]]


def test_model_eviction_and_retired_turns_evict_together():
    engine, choices = _think_run(max_turns=2, calls=4)
    # Call 2 retires turn 1 ([3, 12), around its already-dead block) and turn 2's block;
    # call 3 retires turn 2 ([12, 22), around its block) and turn 3's block [26, 30).
    assert [c.get("eviction") for c in choices][2:] == [
        {"spans": [[3, 6], [10, 12], [16, 20]], "at": 21},
        {"spans": [[12, 16], [20, 22], [26, 30]], "at": 31},
    ]
    # The parts of a retired turn are one survivor span; the span beyond it shifts down.
    assert [c["evict_spans"] for c in engine.evictions][2:] == [[[3, 8], [12, 16]], [[3, 9], [13, 17]]]


def test_qwen_keeps_reasoning_and_evicts_only_empty_think_blocks():
    # Thinking on: each turn's block holds reasoning, which stays attended.
    _, choices = _think_run(max_turns=100, calls=3, think=[OPEN, 63, 64, CLOSE, NL])
    assert [c.get("eviction") for c in choices] == [None, None, None]


def test_occurrences_find_whole_markers_and_rescan_one_cut_off():
    tokens = SYSTEM + THINK + [90, E, OPEN, 63, CLOSE, NL] + THINK
    assert occurrences(tokens, 3, len(tokens), tuple(THINK)) == ([[3, 7], [13, 17]], len(tokens))
    # A marker straddling the end of what the engine holds is looked for again next call.
    assert occurrences(tokens, 3, 15, tuple(THINK)) == ([[3, 7]], 13)
    assert occurrences(tokens, 7, 9, tuple(THINK)) == ([], 9)


def test_qwen_reasoning_survives_tool_results_until_the_next_user_query():
    # Every turn answers a tool result: no user query after turn 1, so nothing dies.
    _, choices = _think_run(max_turns=100, calls=3, user=TOOL_RESULT)
    assert [c.get("eviction") for c in choices] == [None, None, None]


def test_reopening_an_evicting_trace_is_counted():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt = SYSTEM + USER + GEN
        for _ in range(3):
            await table.generate("t", prompt, {})
            prompt = prompt + [90, E] + BRIDGE + USER + GEN
        await table.generate("t", SYSTEM + [3, 3, E] + GEN, {})  # rewritten history
        return table

    assert run(scenario()).stats["resurrected"] == 1


def test_with_overlap_scheduling_evictions_start_after_the_last_sampled_token():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        window = TurnWindow(max_turns=2, stride=1, message_end_id=E)
        table = SessionTable(engine, timeout=60, eviction=window, overlap_schedule=True)
        prompt, choices = SYSTEM + USER + GEN, []
        for _ in range(3):
            response = await generate_sglang(engine, prompt, {}, sessions=table, trace_id="t")
            choices.append(response["choices"][0])
            prompt = prompt + [90, E] + BRIDGE + USER + GEN
        return choices

    # Resident 14 tokens: overlap scheduling already computed the last sampled token's
    # KV (position 13) with turn 1 visible, so turn 1 is unseen from position 14 on.
    assert run(scenario())[-1].get("eviction") == {"spans": [[3, 8]], "at": 14}


def _evicting_run(delta: bool):
    """Four r2-1 turns, each as a full prompt or as a session delta (prefix_len)."""

    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt, held, responses = SYSTEM + USER + GEN, 0, []
        for _ in range(4):
            send, prefix = (prompt[held:], held) if delta and held else (prompt, 0)
            responses.append(
                await generate_sglang(engine, send, {}, sessions=table, trace_id="t", prefix_len=prefix)
            )
            held = len(prompt) + 2
            prompt = prompt + [90, E] + BRIDGE + USER + GEN
        return engine, table, responses

    return run(scenario())


def test_delta_requests_match_full_prompts_and_skip_the_echo():
    full_engine, full_table, full = _evicting_run(delta=False)
    delta_engine, delta_table, delta = _evicting_run(delta=True)
    # The engine sees the same calls, evictions and all, and the trainer the same records.
    assert delta_engine.calls == full_engine.calls
    assert delta_engine.evictions == full_engine.evictions
    assert [r["choices"][0].get("eviction") for r in delta] == [r["choices"][0].get("eviction") for r in full]
    assert delta_table.stats == full_table.stats and delta_table.stats["evicting_calls"] == 2
    # Only the first (full) request gets its prompt back; usage still counts the whole prompt.
    assert "prompt_token_ids" in delta[0] and all("prompt_token_ids" not in r for r in delta[1:])
    assert [r["usage"]["prompt_tokens"] for r in delta] == [len(r["prompt_token_ids"]) for r in full]


@pytest.mark.parametrize("prefix", [0, 4, 6])
def test_delta_on_a_prefix_the_session_does_not_hold_is_rejected(prefix):
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        if prefix:
            await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")  # holds 5 tokens
        with pytest.raises(SessionDesyncError):
            await generate_sglang(engine, [7, 8], {}, sessions=table, trace_id="t", prefix_len=prefix or 5)
        return engine

    calls = run(scenario()).calls
    assert sum(c[0] == "generate" for c in calls) == (1 if prefix else 0)


def test_delta_without_a_session_table_is_rejected():
    with pytest.raises(SessionDesyncError):
        run(generate_sglang(FakeEngine(), [7, 8], {}, trace_id="t", prefix_len=5))


def test_only_a_response_that_ran_on_the_session_is_marked():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        on_session = await generate_sglang(engine, [1, 2, 3], {}, sessions=table, trace_id="t")
        no_table = await generate_sglang(engine, [1, 2, 3], {}, trace_id="t")
        engine.gate = asyncio.Event()
        first = asyncio.create_task(generate_sglang(engine, [1, 2, 3, 90, 91, 4], {}, sessions=table, trace_id="t"))
        await asyncio.sleep(0)
        busy = asyncio.create_task(generate_sglang(engine, [1, 2, 3, 90, 91, 5], {}, sessions=table, trace_id="t"))
        await asyncio.sleep(0)
        engine.gate.set()
        return on_session, no_table, await first, await busy

    on_session, no_table, first, busy = run(scenario())
    # The client sends deltas only after a marked response: never to a server without
    # sessions, and not after a busy trace's stateless call.
    assert on_session.get("session") and first.get("session")
    assert "session" not in no_table and "session" not in busy


def test_delta_while_the_trace_is_busy_is_rejected():
    async def scenario():
        engine = FakeEngine()
        table = SessionTable(engine, timeout=60)
        await table.generate("t", [1, 2, 3], {})
        engine.gate = asyncio.Event()
        first = asyncio.create_task(table.generate("t", [7], {}, prefix_len=5))
        await asyncio.sleep(0)
        with pytest.raises(SessionDesyncError):
            await table.generate("t", [8], {}, prefix_len=5)
        engine.gate.set()
        await first

    run(scenario())


def test_aborted_delta_call_is_retried_on_the_full_prompt():
    async def scenario():
        engine = FakeEngine(completion=(90, E))
        table = SessionTable(engine, timeout=60, eviction=TurnWindow(max_turns=2, stride=1, message_end_id=E))
        prompt = SYSTEM + USER + GEN
        await table.generate("t", prompt, {})
        engine.abort = True
        tail = BRIDGE + USER + GEN
        await table.generate("t", tail, {}, prefix_len=len(prompt) + 2)
        return engine

    engine = run(scenario())
    # The retry reopens on the full chronological prompt, rebuilt from the session.
    assert engine.calls[-2] == ("generate", "t.1", SYSTEM + USER + GEN + [90, E] + BRIDGE + USER + GEN)

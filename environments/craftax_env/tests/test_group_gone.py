"""A branch that fails or is cancelled marks itself gone in the group rendezvous, so its siblings are
judged without it instead of waiting out the timeout."""

import asyncio
import json
import os
from types import SimpleNamespace

from crafter_env.capi import CrafterJudgeConfig
from crafter_env.judge import GroupRendezvous

from test_segments import FakeTrace, make
from craftax_env.taskset import child_task


def test_gone_members_complete_the_set(tmp_path):
    rv = GroupRendezvous(str(tmp_path / "groups"), size=3, timeout_s=30, poll_s=0.05)
    calls = []

    async def call(infos):
        calls.append([i["who"] for i in infos])
        return [{"score": i["who"]} for i in infos], 0.1, None, None

    async def scenario():
        a = asyncio.create_task(rv.meet("node", "a", {"who": "a"}, call))
        await asyncio.sleep(0.2)
        rv.mark_gone("node", "b")  # b failed: a and c are judged together as soon as c arrives
        c = asyncio.create_task(rv.meet("node", "c", {"who": "c"}, call))
        return await asyncio.gather(a, c)

    (va, ta), (vc, tc) = asyncio.run(scenario())
    assert calls == [["a", "c"]] and va == {"score": "a"} and vc == {"score": "c"} and ta["group_size"] == 2


def test_alone_after_siblings_gone_is_judged_alone_at_once(tmp_path):
    rv = GroupRendezvous(str(tmp_path / "groups"), size=3, timeout_s=30, poll_s=0.05)

    async def call(infos):
        raise AssertionError("no comparative call for one member")

    async def scenario():
        rv.mark_gone("node", "b")
        rv.mark_gone("node", "c")
        return await asyncio.wait_for(rv.meet("node", "a", {"who": "a"}, call), timeout=5)

    verdict, timing = asyncio.run(scenario())
    assert verdict is None and timing["group_incomplete"] == 0.0  # None: judged alone (code only under rank mode)


def test_finalize_marks_a_failed_branch_gone(tmp_path):
    task, env = make(tmp_path, segment_turns=3, code_facts=True, judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    task = child_task(task, {"id": "parent-node", "parent": None, "depth": 1, "state_path": None, "summary": None})
    failed = FakeTrace("dead"); failed.ok = False  # the segment raised: no info["craftax"] was written
    asyncio.run(env.finalize(task, SimpleNamespace(traces=[failed])))
    d = tmp_path / "state" / "judge_groups" / "parent-node"
    assert [f for f in os.listdir(d) if f.startswith("gone_")] == ["gone_dead.json"]
    assert json.load(open(d / "gone_dead.json"))["member"] == "dead"


def test_cancelled_run_marks_gone(tmp_path):
    task, env = make(tmp_path, segment_turns=3, code_facts=True, judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    task = child_task(task, {"id": "parent-node", "parent": None, "depth": 1, "state_path": None, "summary": None})

    async def cancelled(*a, **k):
        raise asyncio.CancelledError

    env.play_segment = cancelled
    try:
        asyncio.run(env.run(task, SimpleNamespace(player=None)))
    except asyncio.CancelledError:
        pass
    d = tmp_path / "state" / "judge_groups" / "parent-node"
    assert len([f for f in os.listdir(d) if f.startswith("gone_cancelled-")]) == 1

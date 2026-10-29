import asyncio
import uuid
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.types import InflightEpisode


def _live_trace_dict() -> dict:
    """A segment's trace as the env client assembles it: a dumped Trace plus its
    `pending` preview, at its completion delta (its final `ok` not landed yet)."""
    nodes = [
        vf.MessageNode(message=vf.UserMessage(content="obs"), token_ids=[1, 2], mask=[False, False], logprobs=[0.0, 0.0]),
        vf.MessageNode(
            message=vf.AssistantMessage(content="<action>noop</action>"),
            token_ids=[3, 4],
            mask=[True, True],
            logprobs=[-0.1, -0.1],
            sampled=True,
            parent=0,
        ),
    ]
    trace = vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=[vf.ModelCall(node=1, usage=vf.Usage(prompt_tokens=2, completion_tokens=2))],
        ok=False,
    )
    return {**trace.model_dump(mode="json"), "pending": []}


def test_emit_segment_ships_the_settled_trace():
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.out_q = asyncio.Queue()
    dispatcher.progress = SimpleNamespace(step=3)
    dispatcher.policy = SimpleNamespace(version=5)
    dispatcher.run_id, dispatcher.run_name = "run", "run"
    meta = InflightEpisode(
        kind="train", env_name="crafter", group_id=uuid.uuid4(), task=vf.Task(vf.TaskData(idx=0)), policy_version=0, step=0
    )
    live = _live_trace_dict()
    # The dispatcher ships a segment once the next one opens: by then the trace's last
    # delta (its final ok, after scoring) has landed on the client's live dict.
    live["ok"] = True

    async def scenario():
        await dispatcher.emit_segment(meta, live, policy_start=4)
        return await dispatcher.out_q.get()

    episode = asyncio.run(scenario())
    assert episode.ok and episode.traces[0].ok
    assert [trace for _, trace in iter_trainable_traces([episode])]
    assert episode.group is not None and episode.group.id != str(meta.group_id)
    assert episode.run.work.policy.start == 4 and episode.run.work.policy.end == 5
    assert episode.run.work.step == 3


def test_emit_segment_keeps_a_failed_trace_failed():
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.out_q = asyncio.Queue()
    dispatcher.progress = SimpleNamespace(step=0)
    dispatcher.policy = SimpleNamespace(version=0)
    dispatcher.run_id, dispatcher.run_name = "run", "run"
    meta = InflightEpisode(
        kind="train", env_name="crafter", group_id=uuid.uuid4(), task=vf.Task(vf.TaskData(idx=0)), policy_version=0, step=0
    )

    async def scenario():
        await dispatcher.emit_segment(meta, _live_trace_dict(), policy_start=0)
        return await dispatcher.out_q.get()

    episode = asyncio.run(scenario())
    assert not episode.ok and not list(iter_trainable_traces([episode]))


def test_a_failed_side_task_is_logged(monkeypatch):
    """A segment shipped in the background that fails is logged, not lost silently."""
    import prime_rl.orchestrator.dispatcher as dispatcher_module

    warnings: list[str] = []
    monkeypatch.setattr(dispatcher_module, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.side_tasks = set()

    async def fails():
        raise ValueError("bad segment")

    async def scenario():
        dispatcher._spawn(fails(), "Shipping segment s1")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert warnings == ["Shipping segment s1 failed: ValueError('bad segment')"]
    assert not dispatcher.side_tasks

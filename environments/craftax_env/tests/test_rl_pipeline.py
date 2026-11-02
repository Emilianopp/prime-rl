"""Craftax on the RL pipeline: the tree sampler builds the in-env beam's child prompt, and
train_on_notes = False masks the notes turn out of the loss."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import verifiers.v1 as vf

from craftax_env.segments import CONTINUATION_HEADER, NOTES_PROTOCOL, with_notes_protocol
from craftax_env.taskset import child_task
from prime_rl.configs.orchestrator import TreeSamplerConfig
from prime_rl.orchestrator.curriculum.samplers import TreeSampler

from test_segments import ACTIONS, NOTES, FakeTrace, make


def test_sampler_child_prompt_matches_child_task(tmp_path):
    task, _ = make(tmp_path, segment_turns=4)
    assert task.data.info["segments"]["info_key"] == "craftax"
    sampler = TreeSampler(TreeSamplerConfig(keep=1, max_depth=3), [task])  # config defaults are Crafter's
    root = next(sampler)
    summary = "GAME RECORD\nturn 4\n\nYOUR NOTES FROM THE PREVIOUS SEGMENT:\nMAP: tree F0 (1,2)"
    trace = vf.Trace(task=vf.TraceTask(type=type(root).__name__, data=root.data, key=root.key, hash=root.hash),
                     agent=vf.AgentInfo(config=vf.AgentConfig()),
                     nodes=[vf.MessageNode(message=vf.AssistantMessage(content="x"), sampled=True)],
                     rewards={"progress": vf.Reward(score=1.0)}, ok=True)
    path = tmp_path / "n1.pkl"
    path.write_bytes(b"w")
    trace.info["craftax"] = {"node_id": "n1", "continues": True, "state_path": str(path), "state_sha256": "x",
                             "summary": summary, "achievements": ["collect_wood"], "max_floor": 1}
    sampler.observe([vf.Episode(env=vf.EnvInfo(id="craftax", name="craftax"), task=trace.task, traces=[trace], ok=True)])
    assert sampler.metrics()["groups_without_info"] == 0 and sampler.metrics()["deepest_floor"] == 1
    child = next(sampler)
    node = child.data.info["node"]
    assert node["depth"] == 1 and node["summary"] == summary
    beam = child_task(task, {**node, "branch": 0})
    assert child.data.system_prompt == beam.data.system_prompt
    assert CONTINUATION_HEADER in child.data.system_prompt
    # play_segment adds the protocol before the record and notes, as on the in-env beam
    sp = with_notes_protocol(child.data.system_prompt)
    assert sp.index(NOTES_PROTOCOL) < sp.index(CONTINUATION_HEADER) < sp.index("MAP: tree F0 (1,2)")


class NodeTrace(FakeTrace):
    def __init__(self, trace_id):
        super().__init__(trace_id)
        self.nodes = []


class NodePlayer:
    """Scripted actions, then notes; every reply is a sampled node of 3 tokens."""

    @asynccontextmanager
    async def interaction(self, task):
        trace = NodeTrace("t")

        class Interaction:
            def __init__(self):
                self.trace, self.n = trace, 0

            async def turn(self, message=None):
                trace.nodes.append(SimpleNamespace(sampled=False, mask=[False], message={"content": message}))
                reply = NOTES if message.startswith("END OF SEGMENT") else ACTIONS[self.n % len(ACTIONS)]
                self.n += 1
                trace.nodes.append(SimpleNamespace(sampled=True, mask=[True] * 3, message={"content": reply}))
                return SimpleNamespace(last_reply=reply, terminated=False)

        yield Interaction()


def test_train_on_notes_false_masks_the_notes_turn(tmp_path):
    task, env = make(tmp_path, segment_turns=4)
    assert env.config.train_on_notes is False
    out = asyncio.run(env.play_segment(task, NodePlayer()))
    trace = out["trace"]
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    assert trace.info["loss_mask"]["drop_nodes"] == [sampled[-1]]  # the notes reply only
    assert not any(trace.nodes[sampled[-1]].mask) and all(trace.nodes[sampled[0]].mask)
    assert trace.metrics["notes_masked_tokens"] == 3.0 and out["notes_masked"]["turn"] == 1


def test_train_on_notes_true_keeps_it(tmp_path):
    task, env = make(tmp_path, segment_turns=4, train_on_notes=True)
    out = asyncio.run(env.play_segment(task, NodePlayer()))
    assert "loss_mask" not in out["trace"].info and out["trace"].metrics["notes_masked_tokens"] == 0.0

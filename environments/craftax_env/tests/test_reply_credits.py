"""Per-reply credits and the deaths term (2026-11-11): valid_action_bonus -> trace.info["node_bonuses"],
missing_notes_penalty -> node_penalties on refused notes replies, death_weight -> the `deaths` reward
term, the think_tag_in_reply forfeit kind, and the group judge's retry."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from craftax_env import judge
from craftax_env.forfeits import classify
from craftax_env.taskset import request_node_bonuses, request_node_penalties

from test_forfeit_penalty import run
from test_segments import ACTIONS, PLAN_NOTES, FakeTrace, make


def test_valid_action_bonus_names_the_replies_that_played(tmp_path):
    interaction, line = run(tmp_path, valid_action_bonus=0.25, forfeit_penalty=1.0)
    trace = interaction.trace
    bonuses, penalties = trace.info["node_bonuses"], trace.info["node_penalties"]
    assert len(bonuses) == line["turns"] - sum(line["forfeits"].values()) == trace.metrics["valid_action_rewarded"]
    assert not {i for i, _ in bonuses} & {i for i, _ in penalties}
    for index, bonus in bonuses:
        assert bonus == 0.25 and trace.nodes[index].text in ("<action>DO</action>", "<action>MOVE_LEFT</action>")


def test_bonus_off_by_default(tmp_path):
    interaction, _ = run(tmp_path)
    assert "node_bonuses" not in interaction.trace.info and interaction.trace.metrics["valid_action_rewarded"] == 0.0


def test_requests_add_to_what_is_listed():
    nodes = [SimpleNamespace(sampled=s) for s in (False, True, False, True, False, True)]
    trace = SimpleNamespace(info={}, nodes=nodes)
    assert request_node_penalties(trace, [1], 1.0) == 1 and request_node_penalties(trace, [3], 0.5) == 1
    assert trace.info["node_penalties"] == [[1, 1.0], [5, 0.5]]
    assert request_node_bonuses(trace, [2], 0.25) == 1 and trace.info["node_bonuses"] == [[3, 0.25]]
    assert request_node_bonuses(trace, [2], 0.0) == 0


def test_think_tag_in_reply_kind():
    assert classify("<action>DO</action>\n</think>", "", "stop", "no_action_tag") == "think_tag_in_reply"
    assert classify("<action>DO</action></think>\n<action>", "", "stop", "no_action_tag") == "malformed_tag"
    assert classify("</think>\nno tag", "", "stop", "no_action_tag") == "content_no_tag"


class NodePlayer:
    """Scripted segment player whose trace grows an observation node and a sampled reply node per turn;
    the first notes reply is `first`, the retry's is `second`."""

    def __init__(self, first, second):
        self.first, self.second, self.notes_turns, self.trace = first, second, 0, None

    @asynccontextmanager
    async def interaction(self, task):
        player = self
        trace = self.trace = FakeTrace("seg")
        trace.nodes = []

        class Interaction:
            def __init__(self):
                self.trace, self.n = trace, 0

            async def turn(self, message=None):
                if message.startswith("END OF SEGMENT") or player.notes_turns:
                    player.notes_turns += 1
                    reply = player.first if player.notes_turns == 1 else player.second
                else:
                    reply = ACTIONS[self.n % len(ACTIONS)]
                    self.n += 1
                trace.nodes.append(SimpleNamespace(sampled=False, text=message))
                trace.nodes.append(SimpleNamespace(sampled=True, text=reply))
                return SimpleNamespace(last_reply=reply, terminated=False)

        yield Interaction()


def test_missing_notes_penalty_names_the_refused_notes_replies(tmp_path):
    task, env = make(tmp_path, segment_turns=3, code_facts=True, missing_notes_penalty=1.0, valid_action_bonus=0.25)
    player = NodePlayer("I will keep going.", PLAN_NOTES)  # the first notes reply has no <notes>, the retry does
    out = asyncio.run(env.play_segment(task, player))
    trace = out["trace"]
    penalized = [trace.nodes[i].text for i, _ in trace.info["node_penalties"]]
    assert penalized == ["I will keep going."] and trace.metrics["notes_penalized"] == 1.0
    assert len(trace.info["node_bonuses"]) == 3  # the three game turns, not the notes replies

    task, env = make(tmp_path / "b", segment_turns=3, code_facts=True, missing_notes_penalty=1.0)
    player = NodePlayer(PLAN_NOTES, PLAN_NOTES)
    out = asyncio.run(env.play_segment(task, player))
    assert "node_penalties" not in out["trace"].info and out["trace"].metrics["notes_penalized"] == 0.0


def test_deaths_term(tmp_path):
    task, env = make(tmp_path, segment_turns=3, code_facts=True, death_weight=0.25)
    out = asyncio.run(env.play_segment(task, NodePlayer(PLAN_NOTES, PLAN_NOTES)))
    trace = out["trace"]
    assert trace.rewards["deaths"] == -trace.metrics["deaths"] == 0.0 and trace.weights["deaths"] == 0.25
    task, env = make(tmp_path / "b", segment_turns=3, code_facts=True)
    out = asyncio.run(env.play_segment(task, NodePlayer(PLAN_NOTES, PLAN_NOTES)))
    assert "deaths" not in out["trace"].rewards


def test_group_judge_retries_a_bad_verdict(monkeypatch):
    calls = []

    async def once(judge_, infos, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("CAPI judge returned an invalid JSON verdict.")
        return ["v"] * len(infos), 0.0, None

    async def no_sleep(_):
        return None

    monkeypatch.setattr(judge, "_group_verdicts_once", once)
    monkeypatch.setattr(judge.asyncio, "sleep", no_sleep)
    verdicts, _, error = asyncio.run(judge.group_verdicts(None, [{}, {}]))
    assert verdicts == ["v", "v"] and error is None and len(calls) == 3
    calls.clear()

    async def never(judge_, infos, **kw):
        calls.append(1)
        raise RuntimeError("bad")

    monkeypatch.setattr(judge, "_group_verdicts_once", never)
    verdicts, _, error = asyncio.run(judge.group_verdicts(None, [{}, {}]))
    assert verdicts is None and error == "bad" and len(calls) == judge.GROUP_JUDGE_RETRIES + 1

"""Adherence (Milestone 3): the PLAN parser, the per-turn tracker and digest, the score from
statuses, and the judge in the beam (single, group) and the RL pipeline (finalize + the
plan-score file the children read), with stub clients (no network)."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from craftax_env import glyph
from craftax_env.adherence import AdherenceTracker, match_achievement, parse_plan
from craftax_env.judge import (
    AdherenceVerdict,
    CraftaxAdherenceJudge,
    CraftaxGroupAdherenceJudge,
    CraftaxNotesJudge,
    GroupAdherenceVerdict,
    NotesVerdict,
    adherence_score,
)
from craftax_env.taskset import child_task
from crafter_env.capi import CrafterJudgeConfig

from test_segments import SegmentPlayer, make

NAMES = None
PLACE = re.compile(r"\bF\d+ \(\s*-?\d+\s*,\s*-?\d+\s*\)")
CANONICAL = re.compile(r"F\d+ \(\d+,\d+\)")


@pytest.fixture(scope="module")
def game():
    return glyph.CraftaxGame(1000042, None)


@pytest.fixture(scope="module")
def names(game):
    return [a.name.lower() for a in game._rt["achievements"]]


def test_parse_plan(names):
    lines = ["1. goal: craft wood pickaxe | where: table F0 (4,2) | needs: 1 wood | done when: Wood Pickaxe in Equipment",
             "2. goal: Mine coal with wood pickaxe | where: coal F0 (6,1) | done when: Coal: 1",
             "3. gather 8+ more wood near F0 (22,25)"]
    sg = parse_plan(lines, names)
    assert [s.achievement for s in sg] == ["make_wood_pickaxe", "collect_coal", "collect_wood"]
    assert sg[0].place == (0, 4, 2) and sg[0].thing == "table" and sg[1].items == [("coal", 1, False)]
    assert sg[2].items == [("wood", 8, True)] and sg[2].place == (0, 22, 25)
    inline = parse_plan(["1: Mine coal F0 (6,1) | done when: Coal:1 2: Place furnace F0 (3,4) | done when: furnace"], names)
    assert [s.achievement for s in inline] == ["collect_coal", "place_furnace"]
    bullets = parse_plan(["- place a crafting table", "- descend the ladder"], names)
    assert [s.achievement for s in bullets] == ["place_table", "enter_dungeon"]
    assert match_achievement("sleep when energy is low", names) == "wake_up"
    assert parse_plan([], names) == []


def test_adherence_score():
    s = adherence_score({1: "done", 2: "attempted", 3: "not_reached"}, 3)
    assert s["score"] == pytest.approx((0.5 + 0.3 * 0.6) / 0.8)
    assert adherence_score({1: "not_reached"}, 2)["statuses"] == {1: "ignored", 2: "ignored"}
    assert adherence_score({1: "ignored", 2: "not_reached"}, 2)["score"] == 0.0  # not_reached after ignored counts as ignored
    assert adherence_score({1: "deviated_justified", 2: "done"}, 2)["score"] == pytest.approx(1.0)
    assert adherence_score({2: "done"}, 2)["score"] == pytest.approx(0.3 / 0.8)  # a missing status is "ignored"


def test_tracker_digest(names):
    game = glyph.CraftaxGame(1000042, None)
    r, c = game.position
    plan = [f"1. goal: collect wood | where: tree F0 ({r},{c}) | done when: Wood: 1",
            f"2. goal: place a table | where: F0 ({r},{c})"]
    tracker = AdherenceTracker(parse_plan(plan, names), game, sorted(game.achievements))
    right = glyph.ACTION_NAMES.index("MOVE_RIGHT")
    for k in range(16):  # walk away from subgoal 1's place
        tracker.reply(None, f"I will go right, turn {k}")
        game.step(right)
        tracker.observe(game, {})
    d = tracker.digest()
    assert "closest 0 tiles" in d["subgoal_status"] and "not done by code" in d["subgoal_status"]
    assert "moved away from subgoal 1" in d["decision_excerpts"] and "I will go right" in d["decision_excerpts"]
    assert len(d["position_track"].splitlines()) == 1 + 16 // 5
    text = "\n".join(d[k] for k in ("subgoal_status", "position_track", "decision_excerpts"))
    assert PLACE.findall(text) and all(CANONICAL.fullmatch(p) for p in PLACE.findall(text))
    tracker.observe(game, {"collect_wood": game.turn})
    assert tracker.digest()["subgoals"][0]["code_done"] is True and tracker.active().n == 2


def _stubs(monkeypatch, plan=0.6, statuses=("done",), fail=False):
    calls = {"notes": [], "adherence": [], "group": []}

    async def notes(self, *, trace=None, **fields):
        calls["notes"].append(fields)
        return SimpleNamespace(parsed=NotesVerdict(findings=0.5, plan=plan, rationale="stub"), usage=None)

    async def adherence(self, *, trace=None, **fields):
        calls["adherence"].append(self.build_messages(**fields))
        if fail:
            raise RuntimeError("judge down")
        return SimpleNamespace(parsed=AdherenceVerdict(subgoals=[{"n": i + 1, "status": s} for i, s in enumerate(statuses)],
                                                       rationale="stub"), usage=None)

    async def group(self, messages, **_):
        n = len(re.findall(r"== Branch \d+ ==", messages))
        calls["group"].append(n)
        return SimpleNamespace(parsed=GroupAdherenceVerdict(verdicts=[
            {"branch": i, "subgoals": [{"n": 1, "status": "done" if i == 0 else "ignored"}], "rationale": f"b{i}"}
            for i in range(n)]), usage=None)

    monkeypatch.setattr(CraftaxNotesJudge, "evaluate", notes)
    monkeypatch.setattr(CraftaxAdherenceJudge, "evaluate", adherence)
    monkeypatch.setattr(CraftaxGroupAdherenceJudge, "complete", group)
    return calls


def _lines(tmp_path, task):
    return [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if f'"{task}"' in l]


def test_beam_single_adherence(tmp_path, monkeypatch):
    calls = _stubs(monkeypatch, plan=0.6, statuses=("attempted",))
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=1, keep=1, root_branches=1,
                     judge=CrafterJudgeConfig(model="stub"), adherence_weight=0.25)
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer())))
    assert len(calls["notes"]) == 2 and len(calls["adherence"]) == 1  # depth 1 has no parent plan
    prompt = calls["adherence"][0]
    for part in ("== The plan (", "1. goal: place_table", "Code status of each subgoal", "place_table:", "Position every 5 turns"):
        assert part in prompt
    env_text = prompt.split("== The plan (", 1)[1].split("== What happened", 1)[0]
    assert all(CANONICAL.fullmatch(p) for p in PLACE.findall(env_text))
    (line,) = _lines(tmp_path, "craftax-adherence")
    assert line["gate"] == pytest.approx(0.6) and line["adherence_score"] == pytest.approx(0.6)
    assert line["reward"] == pytest.approx(0.36) and line["depth"] == 2 and line["statuses"] == {"1": "attempted"}


def test_beam_group_adherence(tmp_path, monkeypatch):
    calls = _stubs(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=3, keep=1, root_branches=1,
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group", adherence_weight=0.25)

    async def notes_group(self, messages, **_):
        raise RuntimeError("notes group down")  # the notes fall back to single; adherence stays grouped

    from craftax_env.judge import CraftaxGroupNotesJudge
    monkeypatch.setattr(CraftaxGroupNotesJudge, "complete", notes_group)
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer())))
    assert calls["group"] == [3] and not calls["adherence"]
    lines = _lines(tmp_path, "craftax-adherence")
    assert sorted(l["adherence_score"] for l in lines) == [0.0, 0.0, 1.0] and all(l["mode"] == "group" for l in lines)


def test_rl_finalize_and_plan_score_file(tmp_path, monkeypatch):
    _stubs(monkeypatch, plan=0.8, statuses=("done",))
    task, env = make(tmp_path, segment_turns=3, judge=CrafterJudgeConfig(model="stub"), adherence_weight=0.25)
    parent = asyncio.run(env.play_segment(task, SegmentPlayer()))
    asyncio.run(env.finalize(task, SimpleNamespace(traces=[parent["trace"]])))
    assert "adherence" not in parent["trace"].rewards
    assert json.load(open(tmp_path / "state" / "plan_scores" / f"{parent['node_id']}.json"))["plan"] == 0.8
    node = {"id": parent["node_id"], "parent": parent["parent"], "depth": 1, "state_path": parent["state_path"],
            "state_sha256": parent["state_sha256"], "summary": parent["summary"]}  # what the tree sampler hands over
    child = asyncio.run(env.play_segment(child_task(task, node), SegmentPlayer()))
    assert child["parent_plan_score"] == pytest.approx(0.8)
    asyncio.run(env.finalize(task, SimpleNamespace(traces=[child["trace"]])))
    tr = child["trace"]
    assert tr.rewards["adherence"] == pytest.approx(0.8) and tr.weights["adherence"] == 0.25
    assert tr.metrics["adherence_gate_missing"] == 0.0


def test_no_parent_plan_is_skipped(tmp_path, monkeypatch):
    calls = _stubs(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, judge=CrafterJudgeConfig(model="stub"), adherence_weight=0.25)
    parent = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes="<notes>MAP:\ntree F0 (24,20)\n"
                                                                          "FINDINGS:\n- nothing notable happened in this segment, no crafts, no deaths, nothing to report\n</notes>")))
    node = {"id": parent["node_id"], "depth": 1, "state_path": parent["state_path"], "summary": parent["summary"]}
    child = asyncio.run(env.play_segment(child_task(task, node), SegmentPlayer()))
    asyncio.run(env._judge_branch(child))
    assert not calls["adherence"] and child["trace"].rewards["adherence"] == 0.0
    assert _lines(tmp_path, "craftax-adherence")[0]["skipped"] == "no parent plan"


def test_adherence_failure_is_neutral(tmp_path, monkeypatch):
    _stubs(monkeypatch, plan=1.0, fail=True)
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=1, keep=1, root_branches=1,
                     judge=CrafterJudgeConfig(model="stub"), adherence_weight=0.25)
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer())))
    (line,) = _lines(tmp_path, "craftax-adherence")
    assert line["judge_failed"] and line["adherence_score"] == 0.5


def test_smoke_digest_fixes(names):
    """From smoke 11489206: an achievement unlocked in an earlier segment does not make a
    count goal done; a subgoal true at the start is not scored; a place the agent never
    saw (a grid-view coordinate) is not tracked."""
    import numpy as np
    game = glyph.CraftaxGame(1000042, None)
    r, c = game.position
    explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
    explored[0, r - 4 : r + 5, c - 5 : c + 6] = True
    plan = ["1. mine wood | where: trees F0 (3,5) | done when: Wood ≥ 3",
            "2. place a crafting table | done when: table placed",
            f"3. drink water | where: water F0 ({r},{c + 1})"]
    tracker = AdherenceTracker(parse_plan(plan, names), game, ["collect_wood", "place_table"], explored)
    d = tracker.digest()
    status = d["subgoal_status"].split("\n")
    assert "not done by code" in status[1] and "never saw" in status[1]  # wood 0 -> target 3, place (3,5) unseen
    assert "already true at the start" in status[3]  # place_table was unlocked before
    assert [g["done_at_start"] for g in d["subgoals"]] == [False, True, False]
    assert d["subgoals"][0]["place_seen"] is False and d["subgoals"][2]["place_seen"] is True
    assert "active subgoal 1 (its place is a cell the agent never saw)" in d["position_track"]
    s = adherence_score({1: "attempted", 2: "done", 3: "not_reached"}, 3, {2})
    assert s["statuses"][2] == "already_done" and s["score"] == pytest.approx(0.6)


def test_not_reached_after_done_is_ignored():
    """From smoke 11489207: a branch that finished subgoal 1 and stopped must not score like
    one that finished all three, unless subgoal 1 finished in the last turns."""
    stopped = adherence_score({1: "done", 2: "not_reached", 3: "not_reached"}, 3)
    assert stopped["statuses"] == {1: "done", 2: "ignored", 3: "ignored"} and stopped["score"] == pytest.approx(0.5)
    finished_late = adherence_score({1: "done", 2: "not_reached", 3: "not_reached"}, 3, late={1})
    assert finished_late["score"] == pytest.approx(1.0)
    assert adherence_score({1: "done", 2: "done", 3: "done"}, 3)["score"] == pytest.approx(1.0)
    assert adherence_score({1: "done", 2: "attempted", 3: "not_reached"}, 3)["score"] == pytest.approx((0.5 + 0.18) / 0.8)

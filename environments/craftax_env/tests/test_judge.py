"""The notes judge: the world view (map, nearest, next step, claim check) and the judge in
single and group mode, with a stub client (no network)."""

import asyncio
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from craftax_env import glyph, world
from craftax_env.judge import (
    CraftaxGroupNotesJudge,
    CraftaxNotesJudge,
    GroupNotesVerdict,
    NotesVerdict,
    judge_fields,
    notes_score,
)
from crafter_env.capi import CrafterJudgeConfig

from test_segments import SegmentPlayer, make

PLACE = re.compile(r"\bF\d+ \(\s*-?\d+\s*,\s*-?\d+\s*\)")
CANONICAL = re.compile(r"F\d+ \(\d+,\d+\)")


@pytest.fixture(scope="module")
def game():
    return glyph.CraftaxGame(1000042, None)


def _cell(game, feature):
    return tuple(int(v) for v in np.argwhere(world._feature_mask(game, 0, feature))[0])


def test_ascii_map(game):
    explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
    r, c = game.position
    explored[0, r - 2 : r + 3, c - 2 : c + 3] = True
    lines = world.ascii_map(game, explored).splitlines()
    assert len(lines) == 2 + 48 and lines[0].startswith("    0000000000111") and lines[1].startswith("    0123456789")
    row = lines[2 + r]
    assert row.startswith(f"{r:>3} ") and row[4 + c] == "@"
    assert row[4 + c + 1].isupper() or not row[4 + c + 1].isalpha()  # seen: upper case
    assert any(ch.islower() for ch in lines[2 + (r + 10) % 48][4:])  # far away: never seen, lower case


def test_claim_check(game):
    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    tr, tc = _cell(game, "tree")
    blind = np.zeros_like(seen)
    lines = [f"tree F0 ({tr},{tc})", f"water floor 0 (0, 0) is a guess", "cow F0 (1,1)", "trees somewhere east"]
    check = world.check_map_claims(game, lines[:1], seen)
    assert (check.correct, check.wrong, check.invented) == (1, 0, 0) and check.text().splitlines()[1] == f"ok: tree F0 ({tr},{tc})"
    assert world.check_map_claims(game, lines[:1], blind).invented == 1  # a real tree the agent never saw
    assert "never saw" in world.check_map_claims(game, lines[:1], blind).text()
    rest = world.check_map_claims(game, lines[2:], seen)
    assert rest.unchecked == 2 and rest.claims == 0 and rest.score == 0.0
    assert world.parse_places("iron (12, 30) on F2", 0) == [(2, 12, 30)]
    assert world.parse_places("coal floor 1 (3,4) and F0 (5, 6)", 0) == [(1, 3, 4), (0, 5, 6)]
    assert world.parse_places("stone at (24,24-25)", 0) == [(0, 24, 24)]
    many = world.ClaimCheck(correct=3, invented=1)
    assert many.score == pytest.approx((3 - 2) / 4)


def test_next_step_and_nearest(game):
    assert world.next_step(game).startswith("collect_wood")
    seen = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
    text = world.nearest(game, seen)
    assert "tree F0 (" in text and "unseen" in text
    assert all(CANONICAL.fullmatch(p) for p in PLACE.findall(text))


def _stub_single(monkeypatch, findings=0.8, plan=0.6, fail=False, cf=None):
    """`cf`: the verdict a code_facts judge returns (default: one true, applied lesson; deviation = plan)."""
    from craftax_env.judge import CFNotesVerdict, FindingVerdict

    calls = []

    async def evaluate(self, *, trace=None, **fields):
        calls.append(self.build_messages(**fields))
        if fail:
            raise RuntimeError("judge down")
        if self.schema is CFNotesVerdict:
            return SimpleNamespace(parsed=cf or CFNotesVerdict(
                findings=[FindingVerdict(n=1, truth="true", lesson=True, applied=True)], deviation=plan, rationale="stub"), usage=None)
        return SimpleNamespace(parsed=NotesVerdict(findings=findings, plan=plan, rationale="stub"), usage=None)

    monkeypatch.setattr(CraftaxNotesJudge, "evaluate", evaluate)
    return calls


def test_single_judge_reward_and_inputs(tmp_path, monkeypatch):
    calls = _stub_single(monkeypatch)
    task, env = make(tmp_path, segment_turns=4, judge=CrafterJudgeConfig(model="stub"), notes_weight=0.5)
    out = asyncio.run(env.play_segment(task, SegmentPlayer()))
    asyncio.run(env._judge_branch(out))
    trace, info = out["trace"], out["trace"].info["craftax"]
    expect = notes_score(info["map_check"], 0.8, 0.6)["score"]
    assert trace.rewards["notes"] == pytest.approx(expect) and trace.weights["notes"] == 0.5
    assert out["reward"] == pytest.approx(trace.rewards["progress"] + 0.5 * expect)
    assert info["judge"]["mode"] == "single" and not info["judge"]["failed"]
    prompt = calls[0]
    for part in ("GAME RECORD", "== The current floor", "nearest features:", "Code check of the MAP lines", "MAP:"):
        assert part in prompt
    # one coordinate convention in everything the judge gets that the env wrote
    fields = judge_fields(info)
    env_text = "\n".join(str(fields[k]) for k in ("record_before", "record_after", "events", "nearest", "next_step", "map_claims"))
    assert PLACE.findall(env_text) and all(CANONICAL.fullmatch(p) for p in PLACE.findall(env_text))
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert judged and judged[0]["notes_score"] == pytest.approx(expect)


def test_single_judge_failure_is_neutral(tmp_path, monkeypatch):
    _stub_single(monkeypatch, fail=True)
    task, env = make(tmp_path, segment_turns=3, judge=CrafterJudgeConfig(model="stub"))
    out = asyncio.run(env.play_segment(task, SegmentPlayer()))
    asyncio.run(env._judge_branch(out))
    trace = out["trace"]
    assert trace.metrics["judge_failed"] == 1.0
    assert trace.rewards["notes"] == pytest.approx(notes_score(trace.info["craftax"]["map_check"], 0.5, 0.5)["score"])


def test_group_judge_in_the_beam(tmp_path, monkeypatch):
    seen_n = []

    async def complete(self, messages, *, trace=None, schema=None, parse=None, **_):
        n = len(re.findall(r"== Branch \d+ ==", messages))
        seen_n.append(n)
        return SimpleNamespace(parsed=GroupNotesVerdict(verdicts=[
            {"branch": i, "findings": 0.2 * i, "plan": 1.0 - 0.2 * i, "rationale": f"b{i}"} for i in range(n)]), usage=None)

    monkeypatch.setattr(CraftaxGroupNotesJudge, "complete", complete)
    _stub_single(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, tree_depth=1, branches=3, keep=1, root_branches=3,
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer())))
    assert seen_n == [3]
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert sorted(j["plan"] for j in judged) == pytest.approx([0.6, 0.8, 1.0]) and all(j["mode"] == "group" for j in judged)


def test_group_judge_under_code_facts(tmp_path, monkeypatch):
    """code_facts in group mode: the CF prompt and schema, per-finding verdicts scored by code."""
    from test_segments import PLAN_NOTES

    from craftax_env.judge import CFGroupNotesVerdict

    seen = []

    async def complete(self, messages, *, trace=None, schema=None, parse=None, **_):
        n = len(re.findall(r"== Branch \d+ ==", messages))
        seen.append((schema, "TURNS LOST" in messages, "truth" in messages))
        truth = ["true", "unsupported", "false"]
        return SimpleNamespace(parsed=CFGroupNotesVerdict(verdicts=[
            {"branch": i, "findings": [{"n": 1, "truth": truth[i], "lesson": True, "applied": True}], "deviation": 1.0,
             "rationale": f"b{i}"} for i in range(n)]), usage=None)

    monkeypatch.setattr(CraftaxGroupNotesJudge, "complete", complete)
    _stub_single(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, tree_depth=1, branches=3, keep=1, root_branches=3, code_facts=True, notes_judge="verdict",
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer(first_notes=PLAN_NOTES))))
    assert seen == [(CFGroupNotesVerdict, True, True)]
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert sorted(j["findings"] for j in judged) == [0.0, 0.0, 1.0]  # false and unsupported both 0 (clipped)
    assert all(j["mode"] == "group" and j["claims"] for j in judged)


def test_group_failure_falls_back_to_single(tmp_path, monkeypatch):
    async def complete(self, messages, **_):
        raise RuntimeError("group call failed")

    monkeypatch.setattr(CraftaxGroupNotesJudge, "complete", complete)
    _stub_single(monkeypatch, findings=0.4, plan=0.4)
    task, env = make(tmp_path, segment_turns=3, tree_depth=1, branches=2, keep=1, root_branches=2,
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer())))
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert len(judged) == 2 and all(j["mode"] == "single" and j["plan"] == 0.4 for j in judged)


def test_finalize_grades_rl_segments(tmp_path, monkeypatch):
    _stub_single(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, judge=CrafterJudgeConfig(model="stub"))
    out = asyncio.run(env.play_segment(task, SegmentPlayer()))
    asyncio.run(env.finalize(task, SimpleNamespace(traces=[out["trace"]])))
    assert "notes" in out["trace"].rewards and out["trace"].info["craftax"]["judge"]["mode"] == "single"
    asyncio.run(env.finalize(task, SimpleNamespace(traces=[out["trace"]])))  # already graded: no second call


def test_empty_notes_are_logged_as_skipped(tmp_path, monkeypatch):
    calls = _stub_single(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, judge=CrafterJudgeConfig(model="stub"), summary_retry=False)
    out = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes="")))
    asyncio.run(env._judge_branch(out))
    assert not calls and out["trace"].rewards["notes"] == 0.0
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert judged and judged[0]["mode"] == "skipped" and judged[0]["skipped"] == "no notes"


def test_notes_score_without_map_under_code_facts():
    assert notes_score(0.0, 0.8, 0.6, map_weight=0.0)["score"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)
    assert notes_score(1.0, 0.0, 0.0)["grounding"] == pytest.approx(0.6)


def test_plan_check_feasible_and_path(game):
    from craftax_env.plan_check import check_plan

    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    good = ["1. collect_wood at the tree F0 (23,24) - done when Wood:1", "2. place_table - done when a table is placed",
            "3. make_iron_pickaxe - done when the HUD shows an iron pickaxe"]
    check = check_plan(game, seen, good)
    assert check.on_path == 1.0 and check.feasible == pytest.approx(2 / 3)  # the iron pickaxe lacks a furnace and pick2
    assert check.critical == ["collect_wood"] and "lacks" in check.text()
    side = ["1. make_wood_sword - done when the HUD shows a sword", "2. collect_wood at a tree - done when Wood:2"]
    check = check_plan(game, seen, side)
    assert check.on_path == 0.5 and check.feasible == pytest.approx(0.5)  # the sword lacks a table
    assert check_plan(game, seen, []).on_path == 0.0 and check_plan(game, seen, []).feasible is None


def test_code_facts_verdict_mix():
    from craftax_env.judge import code_facts_verdict

    off = {"code_facts": True, "plan_feasible": 1.0, "plan_on_path": 0.0, "findings_weight": 0.3}
    v = code_facts_verdict(off, 0.6, 1.0, "", False)  # off the path, the judge says the departure is right
    assert v["path"] == 1.0 and v["plan"] == 1.0 and v["score"] == pytest.approx(0.7 + 0.3 * 0.6)
    v = code_facts_verdict(off, 0.6, 0.0, "", False)
    assert v["path"] == 0.0 and v["plan"] == 0.5 and v["score"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.6)
    on = {**off, "plan_on_path": 1.0}
    assert code_facts_verdict(on, 0.6, 0.0, "", False)["path"] == 1.0  # on the path: the verdict is ignored
    failed = code_facts_verdict(off, 0.5, 0.5, "judge failed", True)  # no neutral 0.5: code only
    assert failed["score"] == pytest.approx(0.5) and "deviation" not in failed


def test_code_facts_judge_reward(tmp_path, monkeypatch):
    from test_segments import PLAN_NOTES

    from craftax_env.judge import CFNotesVerdict, FindingVerdict

    unapplied = CFNotesVerdict(findings=[FindingVerdict(n=1, truth="true", lesson=True, applied=False)], deviation=0.0)
    calls = _stub_single(monkeypatch, cf=unapplied)
    task, env = make(tmp_path, segment_turns=4, code_facts=True, notes_judge="verdict", judge=CrafterJudgeConfig(model="stub"), notes_weight=0.5)
    out = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes=PLAN_NOTES)))
    asyncio.run(env._judge_branch(out))
    info = out["trace"].info["craftax"]
    assert info["plan_on_path"] == 1.0 and info["plan_feasible"] == 1.0  # wood then a table: on the path, feasible
    assert info["findings_count"] == 1 and info["turns_lost_total"] < 10  # 4 turns: no waste score owed
    assert out["trace"].rewards["notes"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.6)  # a true lesson, fix not in the plan
    assert info["judge"]["claims"] == [{"n": 1, "truth": "true", "lesson": True, "applied": False}]
    prompt = calls[0]
    assert "KNOWN PLACES" in prompt and "TURNS LOST" in prompt and "Code check of the plan" in prompt and "critical path" in prompt
    assert "Code check of the MAP lines" not in prompt
    assert "from KNOWN PLACES, in the same F<floor> (row,col) form" in prompt and "itself at (4,5)" not in prompt


def test_findings_score_claims():
    from craftax_env.judge import FindingVerdict, findings_score

    def c(truth, lesson=True, applied=True):
        return FindingVerdict(n=1, truth=truth, lesson=lesson, applied=applied)

    assert findings_score([c("true")], 1, 0, False) == 1.0
    assert findings_score([c("true"), c("false")], 2, 0, False) == 0.0  # a false finding cancels a good one
    assert findings_score([c("true"), c("true"), c("false")], 3, 0, False) == pytest.approx(1 / 3)
    assert findings_score([c("true", lesson=False)], 1, 0, False) == pytest.approx(0.2)  # a rule restated
    assert findings_score([c("true", applied=False)], 1, 0, False) == pytest.approx(0.6)
    assert findings_score([c("true")], 2, 0, False) == pytest.approx(0.5)  # one finding left unjudged counts 0
    assert findings_score([], 0, 3, False) == 0.5 and findings_score([], 0, 30, False) == 0.2  # "none"
    assert findings_score([], 0, 0, True) == 0.2


def test_cf_verdict_waste():
    from craftax_env.judge import CFNotesVerdict, FindingVerdict, cf_verdict

    info = {"code_facts": True, "plan_feasible": 1.0, "plan_on_path": 1.0, "findings_weight": 0.3,
            "findings_count": 1, "turns_lost_total": 40}
    good = [FindingVerdict(n=1, truth="true", lesson=True, applied=True)]
    v = cf_verdict(info, CFNotesVerdict(findings=good, waste=0.0, deviation=1.0))
    assert v["waste"] == 0.0 and v["score"] == pytest.approx(0.7 + 0.3 * 0.5)  # findings 1, waste 0
    v = cf_verdict(info, CFNotesVerdict(findings=good, waste=None, deviation=1.0))  # owed but not given
    assert v["waste"] == 0.0
    v = cf_verdict({**info, "turns_lost_total": 5}, CFNotesVerdict(findings=good, waste=0.0, deviation=1.0))
    assert "waste" not in v and v["score"] == pytest.approx(1.0)  # few turns lost: no waste score


def test_plan_check_material_runs_are_not_wasted(game):
    from craftax_env.plan_check import check_plan

    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    game2 = glyph.CraftaxGame.restore(glyph.CraftaxGame.save(game), None)
    game2.achievements = set(game2.achievements) | {"collect_wood"}  # unlocked once, wood spent since
    check = check_plan(game2, seen, ["1. 2 wood at a tree - done when Wood:2"])
    assert check.feasible == 1.0 and "already unlocked" not in check.text()


def test_on_path_counts_the_next_two_steps_materials(game):
    """2 wood, no table: the table (2 wood) is next, then the wood pickaxe (1 wood): a wood run is on
    the path (3 wood summed), a pickaxe craft as the first subgoal is not (the table comes first)."""
    import numpy as np

    from craftax_env import world
    from craftax_env.plan_check import check_plan

    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    g = glyph.CraftaxGame.restore(glyph.CraftaxGame.save(game), None)
    g.state = g.state.replace(inventory=g.state.inventory.replace(wood=2))
    g.achievements = set(g.achievements) | {"collect_wood"}
    achs, mats, _ = world.critical_path(g, seen, steps=2)
    assert achs[:2] == ["place_table", "make_wood_pickaxe"] and "wood" in mats
    assert world.critical_path(g, seen, steps=1)[1] == set()  # the table alone: 2 wood held, nothing missing
    assert check_plan(g, seen, ["1. 1 wood at a tree - done when Wood:3"]).on_path == 1.0
    assert check_plan(g, seen, ["1. make_wood_pickaxe at a table - done when a pickaxe"]).on_path == 0.0

"""The 2026-11-04 reward: on_track (the plan chains to a new achievement), the state term (held
tools), the cumulative beam score, the rank judge (one comparative call per parent, false findings
marked) with its code-only fallback, code adherence, and the per-term routing request."""

import asyncio
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from craftax_env import glyph, world
from craftax_env.judge import (
    CFRankVerdict,
    CraftaxGroupNotesJudge,
    CraftaxNotesJudge,
    code_adherence_verdict,
    code_only_verdict,
    rank_verdict,
    RankBranchVerdict,
    RankFinding,
)
from craftax_env.plan_check import check_plan, plan_track_score
from crafter_env.capi import CrafterJudgeConfig
from crafter_env.taskset import notes_node_indexes, request_term_routing, select_children

from test_segments import PLAN_NOTES, SegmentPlayer, make


@pytest.fixture(scope="module")
def game():
    return glyph.CraftaxGame(1000042, None)


def _with(game, achievements=(), **inventory):
    g = glyph.CraftaxGame.restore(glyph.CraftaxGame.save(game), None)
    if inventory:
        g.state = g.state.replace(inventory=g.state.inventory.replace(**inventory))
    g.achievements = set(g.achievements) | set(achievements)
    return g


def test_on_track_targets_a_new_achievement(game):
    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    done = ("collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "collect_sapling")
    g = _with(game, done)  # died: tools gone, the critical path says "rebuild the wood pickaxe"
    assert world.critical_path(g, seen)[0] == ["place_table", "make_wood_pickaxe"]
    restart = ["1. 2 wood at the tree F0 (23,24) - done when Wood:2", "2. place_table at F0 (24,24) - done when a table",
               "3. make_wood_pickaxe at the table - done when the HUD shows a wood pickaxe"]
    c = check_plan(g, seen, restart)
    assert c.on_path == 1.0 and c.on_track == 0.0 and c.target is None  # on the path, but it reaches nothing new
    assert "reaches no achievement this lineage lacks" in c.text()
    forward = restart + ["4. make_stone_pickaxe at the table - done when the HUD shows a stone pickaxe"]
    c = check_plan(g, seen, forward)
    assert c.on_track == 1.0 and c.target == "make_stone_pickaxe" and c.target_n == 4  # every earlier step serves it
    detour = ["1. collect_sapling at F0 (24,20) - done when Sapling:1", "2. make_wood_sword at a table - done when a sword"]
    c = check_plan(g, seen, detour)
    assert c.on_track == 0.5 and c.target == "make_wood_sword"  # a new target, but the sapling run serves it not
    assert check_plan(g, seen, ["1. make_wood_sword at a table - done when a sword"]).on_track == 1.0
    assert plan_track_score(None, 0.5) == 0.5 and plan_track_score(1.0, 0.0) == 0.5


def test_state_value_counts_held_tools_and_stations(game):
    seen = np.ones(np.asarray(game.state.map).shape, dtype=bool)
    assert world.state_value(game, seen) == (0.0, {"pickaxe": 0.0, "sword": 0.0})
    g = _with(game, pickaxe=2, sword=1)
    points, parts = world.state_value(g, seen)
    assert points == 3.0 and parts["pickaxe"] == 2.0 and parts["sword"] == 1.0
    assert world.state_value(_with(game, pickaxe=4), seen)[0] == 6.0


def test_rank_verdict_scores():
    info = {"code_facts": True, "notes_judge": "rank", "plan_feasible": 1.0, "plan_on_track": 1.0, "findings_count": 2,
            "judge_weight": 0.5, "false_finding_penalty": 0.3}
    best = rank_verdict(info, RankBranchVerdict(branch=0, rank=1, findings=[RankFinding(n=1, truth="true")]), 4)
    assert best["score"] == pytest.approx(1.0) and best["rank_score"] == 1.0 and best["plan"] == 1.0
    worst = rank_verdict(info, RankBranchVerdict(branch=1, rank=4, findings=[RankFinding(n=1, truth="false")]), 4)
    assert worst["rank_score"] == 0.0 and worst["false_findings"] == 1.0
    assert worst["score"] == pytest.approx(0.5 * 1.0 + 0.0 - 0.3 * 0.5)  # one of two findings false
    mid = rank_verdict({**info, "plan_on_track": 0.0, "plan_feasible": None}, RankBranchVerdict(branch=2, rank=2), 4)
    assert mid["plan"] == 0.0 and mid["score"] == pytest.approx(0.5 * (2 / 3))
    alone = code_only_verdict(info, "alone")
    assert alone["score"] == 1.0 and alone["findings"] == 0.0 and not alone["failed"]


def _rank_stub(monkeypatch, ranks=None, fail=False):
    seen = []

    async def complete(self, messages, *, trace=None, schema=None, parse=None, **_):
        n = len(re.findall(r"== Branch \d+ ==", messages))
        seen.append((schema, messages))
        if fail:
            raise RuntimeError("judge down")
        order = ranks or list(range(1, n + 1))
        return SimpleNamespace(parsed=CFRankVerdict(rationale="stub", verdicts=[
            {"branch": i, "rank": order[i], "findings": [{"n": 1, "truth": "false" if i == n - 1 else "true"}]}
            for i in range(n)]), usage=None)

    async def evaluate(self, *, trace=None, **fields):  # the single judge must never be called in rank mode
        raise AssertionError("single judge called")

    monkeypatch.setattr(CraftaxGroupNotesJudge, "complete", complete)
    monkeypatch.setattr(CraftaxNotesJudge, "evaluate", evaluate)
    return seen


def test_rank_judge_in_the_beam(tmp_path, monkeypatch):
    seen = _rank_stub(monkeypatch)
    task, env = make(tmp_path, segment_turns=3, tree_depth=1, branches=3, keep=1, root_branches=3, code_facts=True,
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group", adherence_weight=0.25)
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer(first_notes=PLAN_NOTES))))
    assert len(seen) == 1 and seen[0][0] is CFRankVerdict
    prompt = seen[0][1]
    assert "Rank the branches" in prompt and "== Branch 2 ==" in prompt and "KNOWN PLACES" in prompt
    assert "Code check of the plan" not in prompt and "legend:" not in prompt and "deviation" not in prompt
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert sorted(j["rank"] for j in judged) == [1.0, 2.0, 3.0] and all(j["mode"] == "group" for j in judged)
    by_rank = {j["rank"]: j for j in judged}
    assert by_rank[1.0]["rank_score"] == 1.0 and by_rank[3.0]["rank_score"] == 0.0 and by_rank[3.0]["false_findings"] == 1.0
    assert by_rank[1.0]["notes_score"] > by_rank[3.0]["notes_score"]
    segs = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"task": "craftax"' in l]
    assert all("plan_on_track" in s and "state_value" in s and "achievement_score_before" in s for s in segs)


def test_rank_judge_failure_is_code_only(tmp_path, monkeypatch):
    _rank_stub(monkeypatch, fail=True)
    task, env = make(tmp_path, segment_turns=3, tree_depth=1, branches=2, keep=1, root_branches=2, code_facts=True,
                     judge=CrafterJudgeConfig(model="stub"), judge_mode="group")
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer(first_notes=PLAN_NOTES))))
    judged = [json.loads(l) for l in open(tmp_path / "episodes.jsonl") if '"craftax-judge"' in l]
    assert len(judged) == 2 and all(j["mode"] == "code" and not j["judge_failed"] for j in judged)
    assert all(j["notes_score"] == pytest.approx(0.5 * (j["plan_feasible"] + j["plan_on_track"])) for j in judged)


def test_code_adherence_from_the_digest():
    def g(n, **kw):
        return {"n": n, "place": [0, 1, 1], "place_seen": True, "code_done": False, "closest": None,
                "done_at_start": False, "done_late": False, **kw}

    info = {"adherence_digest": {"subgoals": [g(1, code_done=True), g(2, closest=1), g(3, closest=9),
                                              g(4, code_done=None, place=None), g(5, done_at_start=True, code_done=True)]}}
    v = code_adherence_verdict(info)
    assert v["statuses"] == {1: "done", 2: "attempted", 3: "ignored", 4: "unchecked", 5: "already_done"}
    assert v["score"] == pytest.approx((0.5 * 1.0 + 0.3 * 0.6 + 0.2 * 0.0) / 1.0) and not v["failed"]
    assert code_adherence_verdict({"adherence_digest": {"subgoals": [g(1, code_done=None, place=None)]}})["score"] == 0.0


def test_code_adherence_in_the_beam_needs_no_judge(tmp_path):
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=1, keep=1, root_branches=1, code_facts=True,
                     adherence_weight=0.25)
    assert env.adhering() and env.config.judge is None
    asyncio.run(env.run(task, SimpleNamespace(player=SegmentPlayer(first_notes=PLAN_NOTES))))
    lines = [json.loads(l) for l in open(tmp_path / "episodes.jsonl")]
    adh = [l for l in lines if l["task"] == "craftax-adherence"]
    assert len(adh) == 1 and adh[0]["mode"] == "code" and adh[0]["gate"] == 1.0 and adh[0]["rationale"].startswith("code:")
    assert not [l for l in lines if l["task"] == "craftax-judge"]  # no notes judge configured


def test_beam_score_is_cumulative_and_select_children_reads_it():
    ahead = {"continues": True, "reward": 0.3, "beam_score": 10.3, "deaths": [], "parent": "a"}
    behind = {"continues": True, "reward": 0.9, "beam_score": 8.9, "deaths": [], "parent": "b"}
    assert select_children([behind, ahead], 1) == [ahead]  # the lineage ahead on achievements stays
    plain = [{"continues": True, "reward": 0.9, "deaths": []}, {"continues": True, "reward": 0.3, "deaths": []}]
    assert select_children(plain, 1) == [plain[0]]  # no beam score: the reward, as before


def test_beam_score_in_play_segment(tmp_path):
    task, env = make(tmp_path, segment_turns=4, code_facts=True, beam_cumulative=True)
    out = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes=PLAN_NOTES)))
    assert out["beam_score"] == pytest.approx(out["reward"] + out["achievement_score_before"])
    assert "state" in out["trace"].rewards  # the held-tools term, weight state_weight (1.0)
    env.config.beam_cumulative = False
    assert env._beam_score(out) == pytest.approx(out["reward"])


def test_term_routing_request(tmp_path, monkeypatch):
    import craftax_env.taskset as ts

    monkeypatch.setattr(ts, "notes_node_indexes", lambda trace, summary_from: [7])  # the fake trace has no nodes
    task, env = make(tmp_path, segment_turns=3, code_facts=True, train_on_notes=True)
    out = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes=PLAN_NOTES)))
    assert out["trace"].info["term_routing"] == {"notes": [7]}  # the notes term trains the notes turn only
    task, env = make(tmp_path, segment_turns=3, code_facts=True, train_on_notes=False)
    out = asyncio.run(env.play_segment(task, SegmentPlayer(first_notes=PLAN_NOTES)))
    assert "term_routing" not in out["trace"].info  # masked notes: nothing to route
    fake = SimpleNamespace(info={}, nodes=[SimpleNamespace(sampled=True, message=SimpleNamespace(content="x")) for _ in range(3)])
    assert notes_node_indexes(fake, 2) == [2]
    request_term_routing(fake, {"notes": [2], "empty": []})
    assert fake.info["term_routing"] == {"notes": [2]}

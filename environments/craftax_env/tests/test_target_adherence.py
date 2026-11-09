"""adherence_mode = "target" and notes_findings = False (2026-11-09)."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from craftax_env import glyph, world
from craftax_env.judge import target_adherence_verdict
from craftax_env.segments import PLAN_PROMPT, PLAN_PROMPT_NO_FINDINGS, PLAN_RETRY_NO_FINDINGS, parse_notes
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


def test_requirement_units_fall_as_materials_come_in(game):
    seen = np.zeros(np.asarray(game.state.map).shape, dtype=bool)  # no table seen: it has to be placed
    bare = world.requirement_units(_with(game), seen, "make_wood_pickaxe")
    achs, _, _ = world.requirements(_with(game), seen, ["make_wood_pickaxe"])
    assert achs == ["place_table", "make_wood_pickaxe"] and bare > len(achs)  # two steps plus the wood they cost
    inv = _with(game).state.inventory
    some = world.requirement_units(_with(game, wood=inv.wood * 0 + 1), seen, "make_wood_pickaxe")
    plenty = world.requirement_units(_with(game, wood=inv.wood * 0 + 20), seen, "make_wood_pickaxe")
    assert bare - 1 == some and plenty == 2  # each wood counts; with all the wood only the two steps remain


def test_target_verdict():
    tp = {"target": "make_stone_pickaxe", "on_track": 1.0, "units": 4, "units_end": 1, "unlocked": False, "score": 0.75}
    v = target_adherence_verdict({"target_progress": tp})
    assert v["score"] == 0.75 and v["target"] == "make_stone_pickaxe" and "4 pieces missing at the start, 1" in v["rationale"]
    assert target_adherence_verdict({"target_progress": None})["score"] == 0.0  # the plan names nothing new
    assert target_adherence_verdict({})["score"] == 0.0


def test_prompt_without_findings():
    assert "FINDINGS" in PLAN_PROMPT and "FINDINGS" not in PLAN_PROMPT_NO_FINDINGS
    assert "PLAN:" in PLAN_PROMPT_NO_FINDINGS and "SURVIVAL:" in PLAN_PROMPT_NO_FINDINGS
    assert "FINDINGS" not in PLAN_RETRY_NO_FINDINGS and "PLAN and SURVIVAL" in PLAN_RETRY_NO_FINDINGS
    example = PLAN_PROMPT_NO_FINDINGS[PLAN_PROMPT_NO_FINDINGS.index("<notes>\nPLAN") + 7:PLAN_PROMPT_NO_FINDINGS.rindex("</notes>")]
    assert len(parse_notes(example)["PLAN"]) == 3 and parse_notes(example)["FINDINGS"] == []


def test_target_adherence_in_the_beam(tmp_path):
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=1, keep=1, root_branches=1, code_facts=True,
                     adherence_weight=0.25, adherence_mode="target", notes_findings=False)
    player = SegmentPlayer(first_notes=PLAN_NOTES)
    asyncio.run(env.run(task, SimpleNamespace(player=player)))
    lines = [json.loads(l) for l in open(tmp_path / "episodes.jsonl")]
    adh = [l for l in lines if l["task"] == "craftax-adherence"]
    assert len(adh) == 1 and adh[0]["mode"] == "target" and adh[0]["gate"] == 1.0
    child = [l for l in lines if l["task"] == "craftax" and l["depth"] == 2][0]
    tp = child["target_progress"]
    assert tp["target"] == "collect_wood" and tp["units"] >= 1 and 0.0 <= tp["score"] <= 1.0  # PLAN_NOTES: 2 wood first, a new achievement in a fresh world
    assert adh[0]["adherence_score"] == pytest.approx(tp["score"])
    asked = [m for m in player.received if isinstance(m, str) and "END OF SEGMENT" in m]
    assert asked and all("FINDINGS:" not in m and "under these two headers" in m for m in asked)

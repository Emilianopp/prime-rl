"""Craftax segments: save/restore, the cut, the notes turn, the game record, the beam."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from craftax_env import glyph
from craftax_env.segments import (
    SUMMARY_PROMPT,
    CONTINUATION_HEADER,
    NOTES_PROTOCOL,
    extract_notes,
    notes_problem,
    parse_notes,
)
from craftax_env.taskset import CraftaxConfig, CraftaxEnvConfig, CraftaxTaskset, child_task

from test_run import make_env

NOTES = """<notes>
MAP:
tree F0 (24,20)
water F0 (30,18)
FINDINGS:
- PLACE_TABLE needs 2 wood
PLAN:
1. goal: place_table | where: tree F0 (24,20) | needs: 2 wood | done when: Achievements shows place_table
SURVIVAL:
if Drink <= 3 then go to water F0 (30,18)
</notes>"""
ACTIONS = ["<action>MOVE_LEFT</action>", "<action>DO</action>", "<action>MOVE_UP</action>", "<action>DO</action>"]


class FakeTrace:
    """The parts of vf.Trace the env and the judge touch: rewards (score and weight), metrics, info."""

    def __init__(self, trace_id):
        self.id, self.rewards, self.weights, self.metrics, self.info = trace_id, {}, {}, {}, {}
        self.num_total_tokens, self.ok = 0, True

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name], self.weights[name] = value, weight

    def record_metrics(self, metrics):
        self.metrics.update(metrics)

    @property
    def reward(self):
        return sum(v * self.weights[k] for k, v in self.rewards.items())


class SegmentPlayer:
    """Scripted: actions in a loop; notes when the env says END OF SEGMENT."""

    def __init__(self, first_notes=NOTES):
        self.received, self.systems, self.first_notes = [], [], first_notes
        self.notes_turns = 0

    @asynccontextmanager
    async def interaction(self, task):
        self.systems.append(task.data.system_prompt)
        player = self
        trace = FakeTrace(f"seg{len(self.systems)}")

        class Interaction:
            def __init__(self):
                self.trace = trace
                self.n = 0

            async def turn(self, message=None):
                player.received.append(message)
                if message.startswith("END OF SEGMENT"):
                    player.notes_turns += 1
                    reply = player.first_notes if player.notes_turns == 1 else NOTES
                else:
                    reply = ACTIONS[self.n % len(ACTIONS)]
                    self.n += 1
                return SimpleNamespace(last_reply=reply, terminated=False)

        yield Interaction()


def make(tmp_path, **cfg):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048, respawn=True)).load()))
    config = CraftaxEnvConfig(episode_log=str(tmp_path / "episodes.jsonl"), state_dir=str(tmp_path / "state"),
                              max_episode_steps=0, on_death="respawn", reward="achievements", **cfg)
    return task, make_env(config)


def test_save_restore_is_bit_identical():
    game = glyph.CraftaxGame(1000042, None)
    acts = [glyph.ACTION_NAMES.index(a) for a in ("DO", "MOVE_LEFT", "DO", "MOVE_UP")] * 6
    for a in acts:
        game.step(a)
    copy = glyph.CraftaxGame.restore(glyph.CraftaxGame.save(game), None)
    for a in acts * 2:
        assert game.step(a) == copy.step(a)
    same = jax.tree.map(lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), game.state, copy.state)
    assert all(jax.tree.leaves(same)) and game.render() == copy.render() and game.turn == copy.turn


def test_two_segments_chain(tmp_path):
    task, env = make(tmp_path, segment_turns=6)
    player = SegmentPlayer()
    first = asyncio.run(env.play_segment(task, player))
    assert first["continues"] and first["ended_by"] == "segment_turns" and first["depth"] == 1
    assert os.path.exists(first["state_path"])
    assert NOTES_PROTOCOL in player.systems[0]
    request = player.received[-1]
    assert request.startswith("END OF SEGMENT") and request.endswith(SUMMARY_PROMPT)
    assert request.index("[Grid]") < request.index(SUMMARY_PROMPT)  # the request comes last
    assert "[Achievements already unlocked]" in request
    assert "Now reason as much as useful" not in request  # the action request is dropped
    assert first["summary"].startswith("GAME RECORD") and "YOUR NOTES FROM THE PREVIOUS SEGMENT:" in first["summary"]
    assert "tree F0 (24,20)" in first["summary"] and "Previous segment: segment 1, turns 1-6" in first["summary"]
    assert first["notes_sections"]["PLAN"][0].startswith("1. goal: place_table")
    trace = first["trace"]
    assert trace.rewards["progress"] == pytest.approx(first["achievement_reward_delta"])
    assert trace.metrics["notes_format"] == 1.0 and trace.metrics["continues"] == 1.0

    node = {"id": first["node_id"], "parent": first["parent"], "depth": 1, "state_path": first["state_path"],
            "state_sha256": first["state_sha256"], "summary": first["summary"], "branch": 0}
    second_player = SegmentPlayer()
    second = asyncio.run(env.play_segment(child_task(task, node), second_player))
    system = second_player.systems[0]
    assert CONTINUATION_HEADER in system and "GAME RECORD" in system and NOTES_PROTOCOL in system
    assert system.index(NOTES_PROTOCOL) < system.index(CONTINUATION_HEADER)  # the protocol stays with the rules
    assert "[Current Observation — turn 6]" in second_player.received[0]  # the game's clock continued
    assert second["first_turn"] == 6 and second["depth"] == 2 and second["parent"] == first["node_id"]
    assert set(first["achievements"]) <= set(second["achievements"])
    lines = [json.loads(l) for l in open(tmp_path / "episodes.jsonl")]
    assert [l["depth"] for l in lines] == [1, 2] and lines[0]["notes"].startswith("MAP:")


def test_notes_retry(tmp_path):
    task, env = make(tmp_path, segment_turns=3)
    player = SegmentPlayer(first_notes="<action>DO</action>")
    out = asyncio.run(env.play_segment(task, player))
    assert player.notes_turns == 2 and player.received[-1].startswith("END OF SEGMENT, still")
    assert out["notes_retry_reason"] == "action" and "MAP:" in out["notes"]
    assert out["trace"].metrics["notes_retry_action"] == 1.0


def test_beam_runs_and_checkpoints(tmp_path):
    task, env = make(tmp_path, segment_turns=3, tree_depth=2, branches=2, keep=1, root_branches=2)
    player = SegmentPlayer()
    asyncio.run(env.run(task, SimpleNamespace(player=player)))
    lines = [json.loads(l) for l in open(tmp_path / "episodes.jsonl")]
    assert [l["depth"] for l in lines] == [1, 1, 2, 2]
    beam = json.load(open(tmp_path / "state" / "beam" / f"{task.data.info['seed']}.json"))
    assert beam["done"] == "tree_depth" and beam["depth"] == 2 and len(beam["beam"]) == 1
    left = [f for f in os.listdir(tmp_path / "state") if f.endswith(".pkl")]
    assert len(left) == 1  # only the kept node of the last level stays (for a deeper run)


def test_notes_helpers():
    assert extract_notes("", NOTES).startswith("MAP:")  # notes written inside the thinking count
    assert extract_notes(NOTES + "\n<action>DO</action>").startswith("MAP:")
    assert notes_problem(NOTES, None, extract_notes(NOTES)) is None
    assert notes_problem("<action>DO</action>", None, "") == "action"
    assert notes_problem("just text " * 10, None, "just text " * 10) == "no_notes_tag"
    assert notes_problem("<notes>short</notes>", None, "short") == "short"
    sections = parse_notes("**MAP:** tree F0 (1,2)\n## PLAN\n1. goal: collect_wood\nSURVIVAL: if food<3 then eat cow")
    assert sections["MAP"] == ["tree F0 (1,2)"] and sections["PLAN"] == ["1. goal: collect_wood"]
    assert sections["SURVIVAL"] == ["if food<3 then eat cow"] and sections["FINDINGS"] == []


def test_one_coordinate_convention(tmp_path, monkeypatch):
    """Every env-written place reads F<floor> (row,col), as glyph.place writes it."""
    import re

    assert glyph.place(0, (19, 29)) == "F0 (19,29)" and glyph.place(2, np.array([3, 4])) == "F2 (3,4)"
    task, env = make(tmp_path, segment_turns=4)
    out = asyncio.run(env.play_segment(task, SegmentPlayer()))
    record = out["record"]
    r, c = out["position"]
    assert f"You are at F0 ({r},{c})" in record and f"Pos: ({r},{c})" in record  # the record and the HUD agree
    places = re.findall(r"\bF\d+ \(\s*-?\d+\s*,\s*-?\d+\s*\)", record + "\n".join(out["events"]))
    assert places and all(re.fullmatch(r"F\d+ \(\d+,\d+\)", p) for p in places)
    assert "(row,col) in that grid" in SUMMARY_PROMPT and "the arrow at (4,5)" in SUMMARY_PROMPT
    game = glyph.CraftaxGame(1000042, None)
    before = game.snapshot()
    move = glyph.ACTION_NAMES.index("MOVE_LEFT")
    game.step(move)
    text = game.feedback(move, before, game.snapshot())
    assert re.search(r"moved to F0 \(\d+,\d+\)|blocked by", text), text


def test_summary_prompt_grid_rule():
    """The notes' view frame: the player is the arrow at grid (4,5), and world.view_to_map
    turns a grid cell into the map cell that holds what the grid shows."""
    from craftax_env import world
    assert "you are the arrow at (4,5)" in SUMMARY_PROMPT
    game = glyph.CraftaxGame(1000043, None)
    trees, checked = world._feature_mask(game, 0, "tree"), 0
    for a in ["MOVE_RIGHT"] * 4 + ["MOVE_DOWN"] * 3:
        game.step(glyph.ACTION_NAMES.index(a))
        rows = game.render()[1].splitlines()
        assert rows[4][5] in "→←↑↓"
        r, c = game.position
        for i, row in enumerate(rows):
            for j, ch in enumerate(row):
                if ch == "♣":
                    f, R, C = world.view_to_map((0, r, c), i, j)
                    assert (f, R, C) == (0, r + i - 4, c + j - 5) and trees[R, C]
                    checked += 1
    assert checked > 0


def test_view_frame_places():
    """Notes places without a floor tag and inside the 9x11 view are view cells; a floor tag
    or a cell outside the view keeps the map place. The judges and adherence see map places."""
    from craftax_env import adherence, world
    view = (0, 24, 24)
    assert world.parse_places("coal (6,7)", 0, view) == [(0, 26, 26)]
    assert world.parse_places("coal (6,7)", 0) == [(0, 6, 7)]  # no view: as written
    assert world.parse_places("table F0 (25,25)", 0, view) == [(0, 25, 25)]
    assert world.parse_places("water (30,18)", 0, view) == [(0, 30, 18)]  # outside the view
    assert "F0 (24,24)" in world.view_note(view)
    assert world.parse_places("cow row 1 col 8", 0, view) == [(0, 21, 27)]
    assert world.parse_places("coal (row 6, col 7) and tree (4,6)", 0, view) == [(0, 26, 26), (0, 24, 25)]
    assert world.parse_places("stone rows 3-8, row 2 col 3-5", 0, view) == []  # ranges are not cells
    names = ["collect_coal", "make_stone_pickaxe"]
    sg = adherence.parse_plan(["1. collect_coal at (0,9) - done when Coal:1"], names, 0, view=view)
    assert sg[0].achievement == "collect_coal" and sg[0].place == (0, 20, 28) and sg[0].done_when == "Coal:1"
    assert "[map place: F0 (20,28)]" in adherence.plan_text(sg)


def test_view_frame_claim_check():
    from craftax_env import world
    game = glyph.CraftaxGame(1000043, None)
    floor, mask = game.visible()
    explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
    explored[floor] |= mask
    rows = game.render()[1].splitlines()
    i, j = next((i, j) for i, row in enumerate(rows) for j, ch in enumerate(row) if ch == "♣")
    view = (floor, *map(int, game.position))
    check = world.check_map_claims(game, [f"tree ({i},{j})"], explored, view=view)
    assert check.correct == 1, check.text()
    assert f"view ({i},{j}) = " in check.text()


def test_summary_frame_roundtrip():
    from craftax_env.segments import NOTES_FRAME, notes_of_summary, summary_text
    text = summary_text("GAME RECORD", "MAP:\n- coal (6,7)")
    assert NOTES_FRAME in text and notes_of_summary(text) == "MAP:\n- coal (6,7)"
    assert notes_of_summary(summary_text("GAME RECORD", "")) == ""

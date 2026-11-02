"""Craftax rules the env and the judges state, checked against the pinned Craftax source:
stations get destroyed (DO facing one, an arrow), a furnace needs no table, and floor-entry
achievements and ladder kills depend on the floor."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import craftax.craftax.constants as C
from craftax_env import glyph, taskset, world
from craftax_env.adherence import FLOOR_ENTRY, LADDER_KILLS, AdherenceTracker, parse_plan
from craftax_env.judge import CRAFTAX_GUIDE
from craftax_env.segments import game_record, standing_stations

from test_segments import NOTES, FakeTrace, make

PLACE_TABLE, DO, NOOP = (glyph.ACTION_NAMES.index(a) for a in ("PLACE_TABLE", "DO", "NOOP"))


@pytest.fixture(scope="module")
def names():
    return [a.name.lower() for a in glyph.CraftaxGame(1000042, None)._rt["achievements"]]


def _ready_game(seed: int = 1000042, max_turns=None) -> glyph.CraftaxGame:
    """A fresh game with 5 wood, facing a grass cell (a table can go there)."""
    game = glyph.CraftaxGame(seed, max_turns)
    (r, c), _ = game.block_ahead()
    st = game.state
    game.state = st.replace(map=st.map.at[0, r, c].set(C.BlockType.GRASS.value),
                            inventory=st.inventory.replace(wood=5))
    return game


def _record(game, stations):
    return game_record(game, stations=stations, record_stations=True, depth=1, first_turn=0, deaths=[],
                       death_inventory="drop_all", previous=False)


def test_do_destroys_a_table_and_the_record_drops_it():
    game = _ready_game()
    cell, _ = game.block_ahead()
    game.step(PLACE_TABLE)
    assert int(np.asarray(game.state.map[0])[cell]) == C.BlockType.CRAFTING_TABLE.value
    stations = [{"kind": "table", "floor": 0, "pos": list(cell), "turn": 1}]
    assert standing_stations(game, stations) == stations and "Stations you placed: table F0" in _record(game, stations)
    game.step(DO)  # Craftax: DO facing a table turns it into path
    assert int(np.asarray(game.state.map[0])[cell]) == C.BlockType.PATH.value
    assert standing_stations(game, stations) == [] and standing_stations(game, stations, floor=0) == []
    assert "Stations you placed: none" in _record(game, stations)


class ScriptPlayer:
    """Plays a fixed action list, then NOOPs; notes at END OF SEGMENT."""

    def __init__(self, actions):
        self.actions = actions

    @asynccontextmanager
    async def interaction(self, task):
        script = self.actions

        class Interaction:
            trace = FakeTrace("script")
            n = 0

            async def turn(self, message=None):
                if message.startswith("END OF SEGMENT"):
                    return SimpleNamespace(last_reply=NOTES, terminated=False)
                act = script[self.n] if self.n < len(script) else "NOOP"
                self.n += 1
                return SimpleNamespace(last_reply=f"<action>{act}</action>", terminated=False)

        yield Interaction()


def test_segment_logs_the_destroyed_table(tmp_path, monkeypatch):
    monkeypatch.setattr(taskset, "_new_game", _ready_game)
    task, env = make(tmp_path, segment_turns=4)
    out = asyncio.run(env.play_segment(task, ScriptPlayer(["PLACE_TABLE", "DO"])))
    line = next(json.loads(x) for x in open(tmp_path / "episodes.jsonl") if "events" in json.loads(x))
    events = "\n".join(line["events"])
    assert "placed a table at F0 (23,24)" in events
    assert "the table at F0 (23,24) was destroyed by your DO facing it" in events
    assert line["stations"] == [] and "Stations you placed: none" in out["record"]


def test_furnace_needs_no_table():
    game = glyph.CraftaxGame(1000042, None)
    st = game.state
    have = {"collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe"}
    game.state = st.replace(achievements=st.achievements.at[np.array([C.Achievement[a.upper()].value for a in have])].set(True),
                            inventory=st.inventory.replace(stone=1, pickaxe=2))
    game.achievements = game._achievement_names()
    step = world.next_step(game)
    assert step.startswith("place_furnace") and "table" not in step and "requirements met" in step
    assert "no table needed" in CRAFTAX_GUIDE and "near a table" not in CRAFTAX_GUIDE


def test_floor_constants_match_craftax():
    level = [C.Achievement(int(i)).name.lower() for i in C.LEVEL_ACHIEVEMENT_MAP]
    assert list(FLOOR_ENTRY[1:]) == level[1:] and LADDER_KILLS == C.MONSTERS_KILLED_TO_CLEAR_LEVEL


def test_descend_is_floor_aware(names):
    plan = ["1. descend the ladder | done when: Floor: 2"]
    assert parse_plan(plan, names, default_floor=0)[0].achievement == "enter_dungeon"
    assert parse_plan(plan, names, default_floor=1)[0].achievement == "enter_gnomish_mines"
    assert parse_plan(plan, names, default_floor=4)[0].achievement == "enter_troll_mines"
    assert parse_plan(plan, names, default_floor=8)[0].achievement is None  # nothing below the Graveyard
    assert parse_plan(["1. go down to the gnomish mines"], names, default_floor=0)[0].achievement == "enter_gnomish_mines"


def test_ladder_kills_target(names):
    plan = ["1. kill 8 monsters to open the ladder | done when: Ladder: OPEN", "2. descend the ladder"]
    assert parse_plan(plan, names, default_floor=0)[0].ladder_floor is None  # the Overworld's starts open
    sg = parse_plan(plan, names, default_floor=1)
    assert sg[0].ladder_floor == 1 and sg[1].ladder_floor is None and sg[1].achievement == "enter_gnomish_mines"
    for text in ("clear the floor", "until the ladder opens", "8 kills on this floor", "unlock the down-ladder"):
        assert parse_plan([f"1. {text}"], names, default_floor=2)[0].ladder_floor == 2, text
    game = glyph.CraftaxGame(1000042, None)
    game.state = game.state.replace(monsters_killed=game.state.monsters_killed.at[1].set(3))
    tracker = AdherenceTracker(sg, game, sorted(game.achievements))
    assert tracker.digest()["subgoals"][0]["code_done"] is False
    game.state = game.state.replace(monsters_killed=game.state.monsters_killed.at[1].set(8))
    tracker.observe(game, {})
    d = tracker.digest()
    assert d["subgoals"][0]["code_done"] is True
    assert "F1's down-ladder: 3 kills at start, 8 at end (8 open it): opened at turn" in d["subgoal_status"]

"""CraftaxEnv.run against a scripted player: real Craftax worlds (JAX on CPU), no model."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from craftax_env import glyph
from craftax_env.taskset import CraftaxConfig, CraftaxEnv, CraftaxEnvConfig, CraftaxTaskset

REPLIES = [
    "<think>maybe <action>DO</action></think>\n<action>MOVE_LEFT</action>",
    "<action>do</action>",
    "<action>JUMP</action>",
    "no tags",
    "<action>wait</action>",
]


class ScriptedInteraction:
    def __init__(self, turns=None):
        self.turns = turns  # stop answering after this many replies (the player's turn limit)
        self.received = []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={}, num_total_tokens=0,
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        reply = REPLIES[(len(self.received) - 1) % len(REPLIES)]
        return SimpleNamespace(last_reply=reply, terminated=self.turns is not None and len(self.received) > self.turns)


def play(tmp_path, turns=None, respawn=False, **env_config):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048, respawn=respawn)).load()))
    assert task.data.prompt is None  # the env opens the episode
    interaction = ScriptedInteraction(turns)

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CraftaxEnvConfig(episode_log=str(log), **env_config))
    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(CraftaxEnv.run(env, task, agents))
    return task, interaction, json.loads(log.read_text())


def test_parse_action():
    assert glyph.parse_action(REPLIES[0]) == (glyph.ACTION_NAMES.index("MOVE_LEFT"), None)  # after </think>
    assert glyph.parse_action("<action>do</action>") == (glyph.ACTION_NAMES.index("DO"), None)
    assert glyph.parse_action("<action>MAKE_IRON_ARMOUR</action>")[0] == glyph.ACTION_NAMES.index("MAKE_IRON_ARMOR")
    assert glyph.parse_action("<action>wait</action>") == (glyph.NOOP, None)
    assert glyph.parse_action("<action>JUMP</action>") == (None, "unknown_name")
    assert glyph.parse_action("no tags") == (None, "no_action_tag")


def test_system_prompt():
    system = glyph.system_prompt(True, 2048, "freeze")
    assert system.startswith("You are playing glyphbench/craftaxfull-v0")
    assert "You may use up to 2048 output tokens" in system
    assert "  MAKE_DIAMOND_SWORD — " in system
    assert "MOVE_FORWARD" in system  # GlyphBench's own format example, kept verbatim


@pytest.mark.parametrize("forfeit_mode", ["freeze", "noop"])
def test_run_plays_an_episode(tmp_path, forfeit_mode):
    task, interaction, line = play(tmp_path, max_episode_steps=12, forfeit_mode=forfeit_mode)
    opening, *rest = interaction.received
    assert opening.startswith("[Legend]\n")
    assert "[Current Observation — turn 0]\n[Grid]\n" in opening
    assert "Step: 0 / 12" in opening and "Floor: 0 (Overworld)" in opening
    assert opening.endswith("action name from the system prompt's action list.")
    assert rest and all("[Current Observation — turn " in m for m in rest)
    assert "Last action MOVE_LEFT (LEFT) yielded reward" in rest[0]
    trace = interaction.trace
    assert trace.rewards["craftax_return"] == pytest.approx(line["return"] / 226)
    # "JUMP" and "no tags" forfeit; turns still count, so the episode ends at 12 turns.
    assert line["forfeits"] == {"unknown_name": 2, "no_action_tag": 2} or line["ended_by"] == "died"
    assert line["turns"] == trace.info["craftax"]["turns"]
    assert line["ended_by"] in {"max_episode_steps", "died"}
    played = sum(line["actions"].values())
    assert played == (line["turns"] if forfeit_mode == "noop" else line["turns"] - sum(line["forfeits"].values()))
    assert line["end"] >= line["start"]


def test_respawn_prompt():
    system = glyph.system_prompt(True, 2048, "freeze", respawn=True)
    assert "Death does not end the episode" in system and "The episode terminates on death" not in system


@pytest.mark.parametrize("death_inventory", ["drop_all", "keep"])
def test_respawn_continues_the_game(tmp_path, monkeypatch, death_inventory):
    step, calls = glyph.CraftaxGame.step, []

    def lethal_step(game, action):
        result = step(game, action)
        calls.append(action)
        if len(calls) in (3, 6):  # die twice
            game.state = game.state.replace(
                player_health=game.state.player_health * 0,
                player_level=game.state.player_level * 0 + 1,  # killed underground
                inventory=game.state.inventory.replace(wood=game.state.inventory.wood * 0 + 5),
            )
            return glyph.Step(result.reward, True, result.unlocked, 0)
        return result

    monkeypatch.setattr(glyph.CraftaxGame, "step", lethal_step)
    task, interaction, line = play(
        tmp_path, turns=10, respawn=True, max_episode_steps=0, on_death="respawn",
        death_inventory=death_inventory, death_penalty=0.01, reward="achievements",
    )
    assert line["ended_by"] == "terminated"  # the player stopped answering, not a death
    assert [d["turn"] for d in line["deaths"]] == [5, 10]  # the 3rd and 6th played actions; forfeits count as turns
    assert all(d["floor"] == 1 for d in line["deaths"])
    after = interaction.received[5]  # the observation after the first death
    assert "You died (death 1, on floor 1" in after and "You respawned on the Overworld" in after
    assert "Floor: 0 (Overworld)" in after and "HP: 9/9" in after
    assert ("Wood:0" in after) == (death_inventory == "drop_all")
    assert "Step: 5    Upstream timestep" in after  # no turn limit in the HUD
    assert interaction.trace.rewards["craftax_achievements"] == pytest.approx(line["achievement_score"] / 226 - 0.02)


def test_parse_actions():
    L, D = glyph.ACTION_NAMES.index("MOVE_LEFT"), glyph.ACTION_NAMES.index("DO")
    reply = "<think>maybe <action>NOOP</action></think>plan: <action>DO</action> later.\n<action>MOVE_LEFT</action>\n<action>left</action>\n<action>DO</action>"
    assert glyph.parse_actions(reply, 1) == ([D], None, 0)  # one allowed: the last tag, as GlyphBench
    assert glyph.parse_actions(reply, 5) == ([L, L, D], None, 0)  # the final run only, not the tag in the prose
    assert glyph.parse_actions(reply, 2) == ([L, L], None, 1)
    assert glyph.parse_actions("<action>DO</action><action>JUMP</action><action>DO</action>", 5) == ([D], None, 2)
    assert glyph.parse_actions("<action>JUMP</action>", 5) == ([], "unknown_name", 0)
    assert glyph.parse_actions("no tags", 5) == ([], "no_action_tag", 0)


def test_multi_action_prompt():
    system = glyph.system_prompt(True, 2048, "freeze", max_actions=4)
    assert "MULTIPLE ACTIONS (active for this run)" in system and "up to 4 actions" in system
    assert "MULTIPLE ACTIONS" not in glyph.system_prompt(True, 2048, "freeze")


def test_feedback_on_collecting():
    game = glyph.CraftaxGame(1000042, 50)
    # the player starts facing up; walk until it faces a tree, then chop it
    for _ in range(40):
        snap = game.snapshot()
        if snap["ahead"] == "tree":
            break
        game.step(glyph.ACTION_NAMES.index(["MOVE_LEFT", "MOVE_UP", "MOVE_RIGHT", "MOVE_DOWN"][_ % 4]))
    before = game.snapshot()
    assert before["ahead"] == "tree"
    do = glyph.ACTION_NAMES.index("DO")
    game.step(do)
    text = game.feedback(do, before, game.snapshot())
    assert text.startswith("DO: used on tree: wood +1"), text
    # a pickaxe craft with no table nearby says what is missing
    craft = glyph.ACTION_NAMES.index("MAKE_WOOD_PICKAXE")
    before = game.snapshot()
    game.step(craft)
    text = game.feedback(craft, before, game.snapshot())
    assert text == "MAKE_WOOD_PICKAXE: nothing made: needs a table within one step.", text
    # so does a table placed with 1 wood (it costs 2)
    place = glyph.ACTION_NAMES.index("PLACE_TABLE")
    before = game.snapshot()
    assert before["items"]["wood"] == 1
    game.step(place)
    text = game.feedback(place, before, game.snapshot())
    assert text == "PLACE_TABLE: nothing placed: needs 2 wood (you have 1).", text


def test_failed_build_reasons():
    snap = {
        "items": dict.fromkeys(["wood", "stone", "coal", "iron", "diamond", "sapling", "torches", "pickaxe", "sword"], 0),
        "near_table": True, "near_furnace": False, "ahead": "water", "mob_ahead": False,
    }
    assert glyph._failed_build("MAKE_IRON_PICKAXE", snap) == (
        "nothing made: needs 1 wood (you have 0), 1 stone (you have 0), 1 iron (you have 0), "
        "1 coal (you have 0), a furnace within one step"
    )
    snap["items"] |= {"wood": 3, "pickaxe": 1}
    assert glyph._failed_build("MAKE_WOOD_PICKAXE", snap) == "nothing made: you already have an equal or better pickaxe"
    assert glyph._failed_build("PLACE_TABLE", snap) == "nothing placed: the cell ahead (water) can't take it"


def test_multi_actions_and_feedback_in_a_run(tmp_path, monkeypatch):
    replies = [
        "<action>MOVE_LEFT</action>\n<action>MOVE_LEFT</action>\n<action>DO</action>",
        "no tags",
        "<action>DO</action>",
    ]
    monkeypatch.setattr(sys.modules[__name__], "REPLIES", replies)
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_actions_per_turn=3)).load()))
    assert "MULTIPLE ACTIONS" in task.data.system_prompt
    interaction = ScriptedInteraction(turns=6)

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CraftaxEnvConfig(episode_log=str(log), max_episode_steps=0, action_feedback=True))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    line = json.loads(log.read_text())
    assert line["turns"] == 6 and line["actions_played"] == 2 * 3 + 2 * 1
    assert line["game_steps"] == line["actions_played"] + 2  # two forfeited turns
    assert line["actions_per_turn"] == {"0": 2, "1": 2, "3": 2}
    after_multi, after_forfeit = interaction.received[1], interaction.received[2]
    assert "Played 3 actions, reward" in after_multi and "1. MOVE_LEFT: " in after_multi and "3. DO: " in after_multi
    assert "the turn was forfeited: the world did not move" in after_forfeit
    assert "then end with 1 to 3 <action> tags" in after_multi

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


def make_env(config):
    """A stand-in for the Env instance: the config plus CraftaxEnv's own methods."""
    env = SimpleNamespace(config=config)
    for name, member in vars(CraftaxEnv).items():  # every method, so the stand-in never lags the class
        if callable(member) and not name.startswith("__"):
            setattr(env, name, member.__get__(env))
    return env


def play(tmp_path, turns=None, respawn=False, **env_config):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048, respawn=respawn)).load()))
    assert task.data.prompt is None  # the env opens the episode
    interaction = ScriptedInteraction(turns)

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = make_env(CraftaxEnvConfig(episode_log=str(log), **env_config))
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
    assert all(isinstance(d["respawn_hostile"], bool) for d in line["deaths"])
    after = interaction.received[5]  # the observation after the first death
    assert "You died (death 1, on floor 1" in after and "You respawned on the Overworld" in after
    assert "Floor: 0 (Overworld)" in after and "HP: 9/9" in after
    assert ("Wood:0" in after) == (death_inventory == "drop_all")
    assert "Step: 5    Upstream timestep" in after  # no turn limit in the HUD
    assert interaction.trace.rewards["craftax_achievements"] == pytest.approx(line["achievement_score"] / 226 - 0.02)



def _with_zombie_at(game, cell):
    """``game`` with melee mob slot 0 on the Overworld alive at ``cell``, the others dead."""
    import numpy as np

    mobs = game.state.melee_mobs
    position = np.asarray(mobs.position).copy()
    mask = np.zeros_like(np.asarray(mobs.mask))
    position[0, 0] = cell
    mask[0, 0] = True
    jnp = game._rt["jax"].numpy
    game.state = game.state.replace(melee_mobs=mobs.replace(position=jnp.asarray(position), mask=jnp.asarray(mask)))
    for kind in ("ranged_mobs", "mob_projectiles"):
        other = getattr(game.state, kind)
        game.state = game.state.replace(**{kind: other.replace(mask=jnp.zeros_like(other.mask))})


def test_safe_respawn_keeps_away_from_hostiles():
    game = glyph.CraftaxGame(1000042, 50)
    nearest = game._free_cell(game.spawn)
    _with_zombie_at(game, nearest)  # a zombie on (well, at) the cell a plain respawn would pick
    assert game.hostiles(0) == [tuple(nearest)]
    assert game.hostile_near(nearest, 0) and not game.hostile_near((nearest[0] + 9, nearest[1]), 8)
    safe = game._free_cell(game.spawn, safe_radius=4)
    assert max(abs(safe[0] - nearest[0]), abs(safe[1] - nearest[1])) > 4
    assert not game.hostile_near(safe, 4)
    # Without safe_radius nothing changes but the mob map: the nearest free cell.
    plain = game._free_cell(game.spawn)
    assert max(abs(plain[0] - game.spawn[0]), abs(plain[1] - game.spawn[1])) <= max(abs(safe[0] - game.spawn[0]), abs(safe[1] - game.spawn[1]))
    at = game.respawn("keep", safe_radius=4)
    assert tuple(at) == tuple(safe) and tuple(int(v) for v in game.position) == tuple(safe)


def test_safe_respawn_falls_back_to_the_nearest_free_cell():
    game = glyph.CraftaxGame(1000042, 50)
    nearest = game._free_cell(game.spawn)
    _with_zombie_at(game, game.spawn)
    # No cell within the search radius is 40 tiles from the zombie: the nearest free cell.
    assert game._free_cell(game.spawn, radius=3, safe_radius=40) == game._free_cell(game.spawn, radius=3)
    assert nearest is not None

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
    env = make_env(CraftaxEnvConfig(episode_log=str(log), max_episode_steps=0, action_feedback=True))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    line = json.loads(log.read_text())
    assert line["turns"] == 6 and line["actions_played"] == 2 * 3 + 2 * 1
    assert line["game_steps"] == line["actions_played"] + 2  # two forfeited turns
    assert line["actions_per_turn"] == {"0": 2, "1": 2, "3": 2}
    after_multi, after_forfeit = interaction.received[1], interaction.received[2]
    assert "Played 3 actions, reward" in after_multi and "1. MOVE_LEFT: " in after_multi and "3. DO: " in after_multi
    assert "the turn was forfeited: the world did not move" in after_forfeit
    assert "then end with 1 to 3 <action> tags" in after_multi


def test_job_deadline_ends_the_episode(tmp_path):
    _, interaction, line = play(tmp_path, max_episode_steps=12, episode_deadline=1.0)  # long past
    assert line["ended_by"] == "deadline" and line["turns"] == 0
    assert len(interaction.received) == 1  # the opening; the reply to it is not played


def test_forfeit_classify():
    from craftax_env.forfeits import classify

    assert classify("", "long thinking", "length", "no_action_tag") == "length"
    assert classify("", "Then I `MA", "stop", "no_action_tag") == "reasoning_only"
    assert classify("", "so <action>DO</action>", "stop", "no_action_tag") == "action_in_reasoning"
    assert classify("890602", "", "stop", "no_action_tag") == "content_no_tag"
    assert classify("<action>", "", "stop", "no_action_tag") == "malformed_tag"
    assert classify("", "", "stop", "no_action_tag") == "empty"
    assert classify("<action>JUMP</action>", "", "stop", "unknown_name") == "unknown_name"


def test_forfeit_tracker_collapse():
    from craftax_env.forfeits import COLLAPSE_STREAK, ForfeitTracker

    t = ForfeitTracker()
    for _ in range(3):
        t.update(None)
    t.update("length")
    t.update(None)
    for _ in range(COLLAPSE_STREAK + 2):
        t.update("reasoning_only")
    m, log = t.metrics(), t.log()
    assert m["forfeit_length"] == 1 and m["forfeit_reasoning_only"] == COLLAPSE_STREAK + 2
    assert m["max_forfeit_streak"] == COLLAPSE_STREAK + 2 and m["forfeit_collapsed"] == 1.0
    assert log["collapse_onset"] == 5 and sum(log["forfeit_buckets"]) == COLLAPSE_STREAK + 3


def test_forfeit_kinds_logged(tmp_path):
    _, _, line = play(tmp_path, turns=10)
    assert line["forfeit_kinds"] == {"unknown_name": 2, "content_no_tag": 2}
    assert line["max_forfeit_streak"] == 2 and line["collapse_onset"] is None

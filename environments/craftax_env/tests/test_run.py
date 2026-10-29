"""CraftaxEnv.run against a scripted player: real Craftax worlds (JAX on CPU), no model."""

import asyncio
import json
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
    def __init__(self):
        self.received = []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={}, num_total_tokens=0,
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        reply = REPLIES[(len(self.received) - 1) % len(REPLIES)]
        return SimpleNamespace(last_reply=reply, terminated=False)


def play(tmp_path, **env_config):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048)).load()))
    assert task.data.prompt is None  # the env opens the episode
    interaction = ScriptedInteraction()

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

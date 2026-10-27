"""PokemonEnv.run against a scripted player: the real game, no model.

Needs $POKEMON_FIRERED_ROM and $POKEMON_FIRERED_STATE (a state from
`python -m pokemon_env.make_start_state`); skipped otherwise.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from pokemon_env.taskset import PokemonConfig, PokemonEnv, PokemonEnvConfig, PokemonTaskset, _parse_buttons

ROM, STATE = os.environ.get("POKEMON_FIRERED_ROM"), os.environ.get("POKEMON_FIRERED_STATE")
needs_rom = pytest.mark.skipif(not (ROM and STATE), reason="needs $POKEMON_FIRERED_ROM and $POKEMON_FIRERED_STATE")

# From the bedroom start state: walk to the stairs, step onto them from the right, go down.
REPLIES = [
    "<buttons>right right right right</buttons>",
    "<buttons>up up up up right</buttons>",
    "no tags",
    "<buttons>left left</buttons>",
]


class ScriptedInteraction:
    def __init__(self, replies):
        self.replies, self.received = replies, []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={}, num_total_tokens=0,
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        reply = self.replies[min(len(self.received) - 1, len(self.replies) - 1)]
        return SimpleNamespace(last_reply=reply, terminated=False)


def play(max_steps, replies=REPLIES, log=None):
    task = next(iter(PokemonTaskset(PokemonConfig(rom_path=ROM, start_state=STATE, num_examples=1)).load()))
    interaction = ScriptedInteraction(replies)

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    env = SimpleNamespace(config=PokemonEnvConfig(max_episode_steps=max_steps, episode_log=log))
    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(PokemonEnv.run(env, task, agents))
    return task, interaction


def test_parse_buttons():
    valid = ("up", "down", "left", "right", "a", "b", "start")
    assert _parse_buttons("go <buttons>Up, up  A jump</buttons>", valid, 6) == ["up", "up", "a"]
    assert _parse_buttons("<buttons>a</buttons> then <buttons>b b</buttons>", valid, 1) == ["b"]
    assert _parse_buttons("none", valid, 6) == []


@needs_rom
def test_tasks_are_prompt_less_and_spread_the_rng():
    tasks = list(PokemonTaskset(PokemonConfig(rom_path=ROM, start_state=STATE, num_examples=3)).load())
    assert all(t.data.prompt is None for t in tasks)
    assert [t.data.info["idle_frames"] for t in tasks] == [0, 1, 2]
    assert "<buttons>" in tasks[0].data.system_prompt


@needs_rom
def test_run_plays_an_episode(tmp_path):
    log = tmp_path / "episodes.jsonl"
    _task, interaction = play(max_steps=5, log=str(log))
    opening, *rest = interaction.received
    assert opening.startswith("Turn 1 of 5.") and "Players House 2F" in opening and "=== MAP:" in opening
    assert len(rest) == 4 and all("=== LAST ACTION ===" in m for m in rest)
    assert "no valid <buttons>" in rest[2]  # the "no tags" reply
    assert "entered PalletTown_PlayersHouse_1F" in rest[3]  # onto the stairs from the right, then down
    trace = interaction.trace
    assert trace.metrics["malformed_replies"] == 1.0
    assert 0.0 <= trace.rewards["milestones"] <= 1.0
    info = trace.info["pokemon"]
    assert info["ended_by"] == "max_steps" and info["steps"] == 5
    line = json.loads(log.read_text())
    assert line["end"] >= line["start"] and line["buttons"]["right"] == 5 and line["task"] == "firered#train0"

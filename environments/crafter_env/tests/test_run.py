"""CrafterEnv.run against a scripted player: real Crafter worlds, no model."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from crafter_env.taskset import CrafterConfig, CrafterEnv, CrafterEnvConfig, CrafterTaskset

REPLIES = ["<action>do</action>", "<action>west</action>", "<action>jump</action>", "no tags", "<action>noop</action>"]


class ScriptedInteraction:
    def __init__(self):
        self.received = []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={},
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        reply = REPLIES[(len(self.received) - 1) % len(REPLIES)]
        return SimpleNamespace(last_reply=reply, terminated=False)


def test_run_plays_an_episode(tmp_path):
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1)).load()))
    assert task.data.prompt is None  # the env opens the episode
    interaction = ScriptedInteraction()

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CrafterEnvConfig(max_episode_steps=12, episode_log=str(log)))
    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(CrafterEnv.run(env, task, agents))

    opening, *rest = interaction.received
    assert opening.startswith("Step: 0/12")
    assert rest and all(m.startswith("Step: ") for m in rest)
    trace = interaction.trace
    assert 0.0 <= trace.rewards["achievements"] <= 1.0
    # "jump" is invalid; "no tags" replays the last action written ("jump"), so is invalid too.
    assert trace.metrics["invalid_actions"] >= 2
    assert trace.info["crafter"]["seed"] == 42
    line = json.loads(log.read_text())
    assert line["steps"] == trace.info["crafter"]["steps"] > 1

"""action_retry: a reply without a valid action is re-asked once before the turn is forfeited."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from craftax_env.taskset import CraftaxConfig, CraftaxEnv, CraftaxEnvConfig, CraftaxTaskset

REPLIES = ["no tags", "<action>DO</action>", "<action>JUMP</action>", "nope", "<action>MOVE_LEFT</action>"]


class Scripted:
    def __init__(self):
        self.received = []
        self.trace = SimpleNamespace(
            rewards={}, metrics={}, info={}, num_total_tokens=0,
            record_reward=lambda name, value, weight: self.trace.rewards.update({name: value}),
            record_metrics=lambda metrics: self.trace.metrics.update(metrics),
        )

    async def turn(self, message=None):
        self.received.append(message)
        return SimpleNamespace(last_reply=REPLIES[(len(self.received) - 1) % len(REPLIES)], terminated=False)


def run(tmp_path, **env_config):
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, max_output_tokens=2048)).load()))
    interaction = Scripted()

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CraftaxEnvConfig(episode_log=str(log), max_episode_steps=6, **env_config))
    for name, member in vars(CraftaxEnv).items():
        if callable(member) and not name.startswith("__"):
            setattr(env, name, member.__get__(env))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    return interaction, json.loads(log.read_text())


def test_retry_rescues_or_forfeits(tmp_path):
    interaction, line = run(tmp_path, action_retry=1)
    if line["ended_by"] == "died":
        pytest.skip("the scripted player died")
    # opening->no tags, retry->DO | obs->JUMP, retry->nope (forfeit) | obs->MOVE_LEFT | obs->no tags, retry->DO | ...
    retries = [m for m in interaction.received if m.startswith("Your last reply had")]
    assert retries[0].startswith("Your last reply had no <action> tag, so no action was played")
    assert retries[1].startswith("Your last reply had an action name that is not in the action list")
    assert line["action_retries"] == len(retries) == 4
    assert line["action_retry_rescued"] == 2
    assert line["forfeits"] == {"no_action_tag": 2}  # the retries that failed too
    assert line["first_try_forfeits"] == {"content_no_tag": 2, "unknown_name": 2}
    assert line["turns"] == 6 and sum(line["actions"].values()) == 4  # forfeits still count as turns (freeze)
    assert line["actions"].get("DO") == 2
    assert interaction.trace.metrics["action_retries"] == 4.0


def test_retry_off_by_default(tmp_path):
    interaction, line = run(tmp_path)
    assert not any(m.startswith("Your last reply had") for m in interaction.received)
    assert line["action_retries"] == 0 and line["action_retry_rescued"] == 0
    assert sum(line["first_try_forfeits"].values()) == sum(line["forfeits"].values())

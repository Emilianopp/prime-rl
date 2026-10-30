"""CrafterEnv.run against a scripted player: real Crafter worlds, no model."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from crafter_env.taskset import CrafterEnv as _CrafterEnv
from crafter_env.taskset import CrafterConfig, CrafterEnv, CrafterEnvConfig, CrafterTaskset

REPLIES = ["<action>do</action>", "<action>west</action>", "<action>jump</action>", "no tags", "<action>noop</action>"]


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


def fake_env(config):
    """A CrafterEnv without its __init__ (no taskset or harness to load): the methods on a bare config."""
    env = _CrafterEnv.__new__(_CrafterEnv)
    env.config = config
    return env


def test_run_plays_an_episode(tmp_path):
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1)).load()))
    assert task.data.prompt is None  # the env opens the episode
    interaction = ScriptedInteraction()

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = fake_env(CrafterEnvConfig(max_episode_steps=12, episode_log=str(log)))
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
    assert line["end"] >= line["start"]  # wall clock
    assert line["steps"] == trace.info["crafter"]["steps"] > 1
    assert line["ended_by"] in {"max_episode_steps", "died", "world_length"}
    assert sum(line["actions"].values()) == line["steps"] - (line["ended_by"] == "max_episode_steps")
    assert set(line["unlocked_at"]) == set(trace.info["crafter"]["achievements"])


def play(config, monkeypatch=None, kill_at=None, reasoning=False):
    """One episode of `config` against the scripted player; `kill_at` zeroes the
    player's health right after that game step."""
    import crafter_env.taskset as ts

    if kill_at is not None:
        step, calls = ts._step, []

        def lethal_step(env, action):
            reward, done, info = step(env, action)
            calls.append(action)
            if len(calls) == kill_at:
                env._player.health = 0
                return reward, True, {**info, "inventory": env._player.inventory.copy()}
            return reward, done, info

        monkeypatch.setattr(ts, "_step", lethal_step)
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, enable_reasoning=reasoning)).load()))
    interaction = ScriptedInteraction()

    @asynccontextmanager
    async def open_interaction(_task):
        yield interaction

    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(CrafterEnv.run(fake_env(config), task, agents))
    return interaction


def test_death_ends_the_episode_by_default(monkeypatch):
    interaction = play(CrafterEnvConfig(max_episode_steps=12), monkeypatch, kill_at=3)
    info = interaction.trace.info["crafter"]
    assert info["ended_by"] == "died" and info["steps"] == 3 and info["deaths"] == []


def test_respawn_keeps_playing_the_same_world(monkeypatch):
    config = CrafterEnvConfig(max_episode_steps=12, on_death="respawn", death_inventory="drop_all", death_penalty=0.05)
    interaction = play(config, monkeypatch, kill_at=3)
    info, trace = interaction.trace.info["crafter"], interaction.trace
    assert info["ended_by"] == "max_episode_steps" and info["steps"] == 12
    (death,) = info["deaths"]
    assert death["step"] == 3 and death["vitals"]["health"] == 0
    assert max(abs(a - b) for a, b in zip(death["respawn"], (32, 32))) <= 16  # near the spawn point
    told = [m for m in interaction.received if m and m.startswith("You died")]
    assert len(told) == 1 and "lost your inventory" in told[0]
    assert "health=9, food=9, drink=9, energy=9" in told[0]
    assert trace.metrics["deaths"] == 1.0
    assert trace.rewards["achievements"] == len(info["achievements"]) / 22 and trace.rewards["deaths"] == -0.05


def test_respawn_cell_ignores_hostiles_unless_respawn_safe():
    import crafter
    from crafter import constants, objects

    from crafter_env.taskset import _free_cell_near, _hostile_near

    env = crafter.Env(seed=42, length=12)
    env.reset()
    world = env._world

    def free(cell):
        material, obj = world[cell]
        return material in constants.walkable and obj is None

    target, beside = next(  # a free cell with a free neighbour for the zombie, away from the player
        ((x, y), (x + 1, y)) for x in range(8, 56) for y in range(8, 56)
        if max(abs(x - 32), abs(y - 32)) > 6 and free((x, y)) and free((x + 1, y)) and not _hostile_near(world, (x, y))
    )
    world.add(objects.Zombie(world, beside, env._player))
    assert _free_cell_near(world, target) == target  # the original rule: the nearest free cell
    safe = _free_cell_near(world, target, safe=True)
    assert safe != target and not _hostile_near(world, safe)
    far = _free_cell_near(world, target, safe=True, safe_radius=8)  # out of the zombie's chase range
    assert far is not None and not _hostile_near(world, far, 8)
    for obj in [o for o in world.objects if type(o).__name__ in ("Zombie", "Skeleton")]:  # only one zombie, to test the radius
        world.remove(obj)
    world.add(objects.Zombie(world, beside, env._player))
    assert _hostile_near(world, (beside[0] + 8, beside[1]), 8) and not _hostile_near(world, (beside[0] + 9, beside[1]), 8)
    corner = next((x, y) for x in range(3) for y in range(3) if free((x, y)))  # near the edge, where world.nearby slices to nothing
    world.add(objects.Zombie(world, corner, env._player))
    assert _hostile_near(world, (corner[0] + 1, corner[1]), 8)


def test_a_reply_without_an_action_tag_repeats_the_last_action_and_says_so(monkeypatch):
    monkeypatch.setattr(__import__(__name__), "REPLIES", ["<action>do</action>", "<", "<action>noop</action>"])
    interaction = play(CrafterEnvConfig(max_episode_steps=6))
    told = [m for m in interaction.received if m and "had no <action> tag" in m]
    assert len(told) == 2  # turns 2 and 5 of the cycle
    assert "your previous action (do) was repeated" in told[0]
    assert interaction.trace.metrics["malformed_replies"] == 2.0



def test_reasoning_prompt_and_observations_agree():
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, enable_reasoning=True)).load()))
    assert "reason before you act" in task.data.system_prompt
    interaction = play(CrafterEnvConfig(max_episode_steps=4), reasoning=True)
    assert all("Reason in 2-4 sentences" in m and "Reply only as" not in m for m in interaction.received)



def test_relaxed_reply_format_control():
    task = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, relaxed_reply_format=True)).load()))
    assert "Reply with one action in XML:" in task.data.system_prompt
    assert "exactly one action" not in task.data.system_prompt and "Notebook" not in task.data.system_prompt



def test_the_malformed_notice_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(__import__(__name__), "REPLIES", ["<action>do</action>", "<"])
    interaction = play(CrafterEnvConfig(max_episode_steps=5, announce_malformed=False))
    assert not any(m and "had no <action> tag" in m for m in interaction.received)
    assert interaction.trace.metrics["malformed_replies"] == 2.0


def test_achievement_list_in_the_system_prompt_is_opt_in():
    from crafter_env.taskset import ACHIEVEMENTS, ACHIEVEMENT_LIST, with_achievement_list

    plain = next(iter(CrafterTaskset(CrafterConfig(num_examples=1)).load()))
    assert "22. collect_diamond" not in plain.data.system_prompt
    listed = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, list_achievements=True)).load()))
    sp = listed.data.system_prompt
    assert ACHIEVEMENT_LIST in sp and sp.index(ACHIEVEMENT_LIST) < sp.index("Reply with exactly one action")
    assert all(f" {name}\n" in sp or sp.endswith(f" {name}") for name in ACHIEVEMENTS)
    assert "1. collect_wood\n" in sp and "22. collect_diamond" in sp
    reasoning = next(iter(CrafterTaskset(CrafterConfig(num_examples=1, list_achievements=True, enable_reasoning=True)).load()))
    assert ACHIEVEMENT_LIST in reasoning.data.system_prompt
    assert with_achievement_list(sp) == sp  # idempotent

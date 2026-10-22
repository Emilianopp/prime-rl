import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from crafter_env.taskset import CrafterConfig, CrafterEnv, CrafterEnvConfig, CrafterTaskset
from crafter_env.workspace import Workspace


def test_write_append_read_and_search():
    space = Workspace()
    space.log("step 3: saw water at (35, 30)")
    space.log("step 9: saw iron at (60, 2)")
    out = space.apply('<write file="Plan.md">get wood</write><append file="plan.md">then table at (31, 29)</append>', 10)
    assert out == ["Wrote plan.md (1 lines).", "Appended to plan.md (2 lines)."]
    assert space.apply("<read>plan.md</read>", 11) == ["plan.md:", "  get wood", "  then table at (31, 29)"]
    hits = space.apply("<search>near:33,30</search>", 12)
    assert hits[0] == "search 'near:33,30': 2 matches"
    assert hits[1:] == ["  ledger: step 3: saw water at (35, 30)", "  plan.md:2: then table at (31, 29)"]
    assert space.apply("<search>re:(iron|diamond)</search>", 13)[1] == "  ledger: step 9: saw iron at (60, 2)"
    assert space.apply("<search>lava</search>", 14) == ["search 'lava': no matches."]
    assert space.apply("<search>re:(</search>", 15)[0].startswith("search 're:(': bad regular expression")
    assert space.apply('<write file="../x">no</write>', 16)[0].startswith('write "../x": not a valid file name')
    assert space.index() == "Workspace: files: plan.md (2 lines); ledger: 2 entries"
    assert space.summary()["counts"] == {"write": 2, "append": 1, "read": 1, "search": 4, "bad_path": 1}


def test_ablations_switch_a_store_off():
    ledger_only = Workspace(files_enabled=False)
    assert ledger_only.apply('<write file="a.md">x</write>', 1) == [] and ledger_only.files == {}
    assert ledger_only.index() == "Workspace: ledger: 0 entries"
    files_only = Workspace(ledger_enabled=False)
    files_only.log("step 1: saw water at (1, 1)")
    assert files_only.ledger == [] and files_only.index() == "Workspace: files: none"


def _play(config, replies, monkeypatch=None, kill_at=None, **env):
    import crafter_env.taskset as ts

    if kill_at is not None:
        step, calls = ts._step, []

        def lethal_step(e, action):
            reward, done, info = step(e, action)
            calls.append(action)
            if len(calls) == kill_at:
                e._player.health = 0
                return reward, True, {**info, "inventory": e._player.inventory.copy()}
            return reward, done, info

        monkeypatch.setattr(ts, "_step", lethal_step)
    task = next(iter(CrafterTaskset(config).load()))
    replies, received = iter(replies), []
    trace = SimpleNamespace(
        rewards={}, metrics={}, info={}, num_total_tokens=0,
        record_reward=lambda name, value, weight: trace.rewards.update({name: value}),
        record_metrics=lambda metrics: trace.metrics.update(metrics),
    )

    class Interaction:
        def __init__(self):
            self.trace = trace

        async def turn(self, message=None):
            received.append(message)
            return SimpleNamespace(last_reply=next(replies, "<action>noop</action>"), terminated=False)

    @asynccontextmanager
    async def open_interaction(_task):
        yield Interaction()

    agents = SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))
    asyncio.run(CrafterEnv.run(SimpleNamespace(config=CrafterEnvConfig(**env)), task, agents))
    return task, received, trace


def test_workspace_episode(monkeypatch):
    task, received, trace = _play(
        CrafterConfig(num_examples=1, workspace=True),
        ['<write file="plan.md">find wood</write><search>tree</search><action>do</action>', "<read>ledger</read><action>noop</action>"],
        monkeypatch,
        kill_at=4,
        max_episode_steps=8,
        on_death="respawn",
    )
    prompt = task.data.system_prompt
    assert "<search>water</search>" in prompt and "plus any workspace tools" in prompt
    assert received[0].startswith("Workspace: files: none; ledger: ") and "Reply only as" not in received[0]
    # The first turn's tools answer at the top of the second observation.
    assert received[1].startswith("Workspace: files: plan.md (1 lines); ledger: ")
    assert "Wrote plan.md (1 lines)." in received[1] and "search 'tree': " in received[1]
    assert "ledger (last " in received[2]
    summary = trace.info["crafter"]["workspace"]
    tail = "\n".join(summary["ledger_tail"])
    assert "died (food" in tail and "respawned at" in tail
    assert summary["files"] == {"plan.md": "find wood"} and summary["searches"][0]["query"] == "tree"
    assert trace.metrics["workspace_search"] == 1.0 and trace.metrics["workspace_read"] == 1.0


def test_ledger_logs_sightings_from_the_start():
    _, received, trace = _play(CrafterConfig(num_examples=1, workspace=True, workspace_files=False), [], max_episode_steps=3)
    summary = trace.info["crafter"]["workspace"]
    assert summary["ledger_entries"] > 0 and all(e.startswith("step ") for e in summary["ledger_tail"])
    assert "saw tree at (" in "\n".join(summary["ledger_tail"])  # seed 42 spawns next to trees
    assert received[0].startswith("Workspace: ledger: ")


def test_notebook_and_workspace_are_exclusive():
    with pytest.raises(ValueError):
        list(CrafterTaskset(CrafterConfig(num_examples=1, notebook=True, workspace=True)).load())


def test_observation_extras(monkeypatch):
    _, received, _ = _play(
        CrafterConfig(num_examples=1), ["<action>do</action>"] * 3, monkeypatch, kill_at=None, max_episode_steps=4,
        observation_extras=True,
    )
    first = received[0]
    assert "Creatures in view: " in first and "Nearest: tree " in first  # seed 42 spawns by trees
    assert "Can place/make now: " in first


def test_extras_report_health_loss_and_affordances():
    import crafter

    from crafter_env.taskset import _available, _format_observation, _player_info

    env = crafter.Env(seed=42, length=100)
    env.reset()
    env._player.inventory["wood"] = 2
    obs = _format_observation(env=env, info=_player_info(env), step=1, max_steps=100, view_radius=4, extras=True, health_lost=2)
    assert "place_table" in _available(env) and "You lost 2 health since your last action." in obs
    plain = _format_observation(env=env, info=_player_info(env), step=1, max_steps=100, view_radius=4)
    assert "Creatures in view" not in plain and "lost" not in plain



def test_extras_without_affordances(monkeypatch):
    _, received, _ = _play(
        CrafterConfig(num_examples=1), ["<action>do</action>"] * 2, monkeypatch, max_episode_steps=3,
        observation_extras=True, extras_affordances=False,
    )
    assert "Nearest: tree " in received[0] and "Can place/make now" not in received[0]

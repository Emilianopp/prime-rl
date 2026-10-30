"""Segments: one episode plays one segment of a game, writes its summary and saves the
world; the next segment restores it. The in-env beam (`tree_depth`) chains them."""

import asyncio
import dataclasses
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from crafter_env.state_store import DiskStateStore, MemoryStateStore
from crafter_env.taskset import (
    CONTINUATION_HEADER,
    DEATH_CAUSES,
    LESSON_HEADER,
    LESSON_PROTOCOL,
    LESSON_REQUEST,
    NOTES_PROTOCOL,
    extract_lesson,
    with_lesson_protocol,
    extract_notes,
    notes_problem,
    with_notes_protocol,
    SUMMARY_PROMPT,
    CrafterConfig,
    CrafterEnv,
    CrafterEnvConfig,
    CrafterTaskset,
    _turn_sampling,
    child_task,
    mask_notes_from_loss,
    RECALL_PROBE,
    RECALL_PROTOCOL,
    score_recall_probe,
    select_children,
)

ACTIONS = ["<action>do</action>", "<action>move_left</action>", "<action>move_up</action>", "<action>noop</action>"]


class FakeTrace:
    def __init__(self, id):
        self.id, self.ok, self.rewards, self.metrics, self.info, self.num_total_tokens = id, True, {}, {}, {}, 0
        self.nodes = []  # MessageNode-like: message, sampled, mask (what the loss sees)

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value * weight

    def record_metrics(self, metrics):
        self.metrics.update(metrics)

    @property
    def reward(self):
        return sum(self.rewards.values())


class ScriptedPlayer:
    """Opens scripted interactions: actions until the summary is asked for, then notes."""

    def __init__(self, tokens_per_turn: int = 0):
        self.interactions = []
        self.tokens_per_turn = tokens_per_turn

    def action_reply(self, interaction, reply):  # subclasses decorate the scripted action reply
        return reply

    @asynccontextmanager
    async def interaction(self, task):
        player = self
        interaction = SimpleNamespace(task=task, received=[], summaries=0)
        interaction.trace = FakeTrace(f"t{len(self.interactions)}")

        def record(message, reply):  # the trace graph: an input node, then the sampled reply
            nodes = interaction.trace.nodes
            if message is not None:
                nodes.append(SimpleNamespace(message={"role": "user", "content": message}, sampled=False, mask=[False] * 3))
            nodes.append(SimpleNamespace(message={"role": "assistant", "content": reply}, sampled=True, mask=[False, True, True, True]))
            return SimpleNamespace(last_reply=reply, terminated=False)

        async def turn(message=None):
            interaction.received.append(message)
            interaction.trace.num_total_tokens += player.tokens_per_turn
            if message is not None and (message.startswith(SUMMARY_PROMPT) or message.endswith(SUMMARY_PROMPT)):
                interaction.summaries += 1
                return record(message, "<notes>Notes: a tree lies 2W, water 3N; no table yet. Plan: collect two wood, place a table, make a wood pickaxe.</notes>")
            reply = player.action_reply(interaction, ACTIONS[(len(interaction.received) - 1) % len(ACTIONS)])
            return record(message, reply)

        interaction.turn = turn
        self.interactions.append(interaction)
        yield interaction


def fake_env(config):
    env = CrafterEnv.__new__(CrafterEnv)
    env.config = config
    return env


def root_task():
    return next(iter(CrafterTaskset(CrafterConfig(num_examples=1, max_episode_steps=10000)).load()))


def test_segment_cuts_summarizes_saves_and_the_next_segment_restores(tmp_path):
    log = tmp_path / "episodes.jsonl"
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=5, state_dir=str(tmp_path / "states"), episode_log=str(log),
                              on_death="respawn", respawn_at="last_sleep", death_inventory="drop_materials")
    env, player = fake_env(config), ScriptedPlayer()
    task = root_task()
    first = asyncio.run(env.play_segment(task, player))
    assert first["continues"] and first["ended_by"] == "segment_turns" and first["steps"] == 5
    assert first["summary"].startswith("Notes: a tree lies 2W") and "<notes>" not in first["summary"]
    assert first["depth"] == 1 and first["parent"] == "42-root"
    assert player.interactions[0].summaries == 1 and player.interactions[0].received[0].startswith("Step: 0/10000\nSegment turn: 0/5")
    assert len(player.interactions[0].received) == 6  # 5 actions sampled, then the notes with the last observation in view
    last = player.interactions[0].received[-1]
    assert last.startswith(SUMMARY_PROMPT) and "Step: 5/10000\nSegment turn: 5/5" in last and "Reply only as <action>" not in last
    assert any("This segment ends in about 2 turns" in m for m in player.interactions[0].received if m)
    assert (tmp_path / "states" / "t0.pkl").exists() and first["state_bytes"] > 1000
    trace = player.interactions[0].trace
    assert trace.info["crafter"]["state_path"] == first["state_path"] and trace.metrics["continues"] == 1.0
    assert "achievements" in trace.rewards  # the segment's delta / 22; deaths are a separate term

    # The child restores the saved world: the summary is in the system prompt, the game step continues.
    node = {"id": first["node_id"], "parent": first["parent"], "depth": first["depth"], "state_path": first["state_path"],
            "state_sha256": first["state_sha256"], "summary": first["summary"], "branch": 0}
    child = child_task(task, node)
    assert child.data.system_prompt.startswith(task.data.system_prompt) and CONTINUATION_HEADER in child.data.system_prompt
    assert child.data.system_prompt.endswith(first["summary"]) and child.data.prompt is None
    sp = player.interactions[0].task.data.system_prompt
    assert NOTES_PROTOCOL in sp and sp.startswith(task.data.system_prompt)
    assert child.data.name == "crafter#train0/d1/t0" and child.key != task.key
    second = asyncio.run(env.play_segment(child, player))
    assert second["game_step"] == first["game_step"] + 5 and second["depth"] == 2 and second["parent"] == first["node_id"]
    assert player.interactions[1].received[0].startswith(f"Step: {first['game_step']}/10000")
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["depth"] for l in lines] == [1, 2] and lines[1]["parent"] == lines[0]["node_id"]
    # A corrupt state is refused.
    bad = {**node, "state_sha256": "0" * 64}
    try:
        asyncio.run(env.play_segment(child_task(task, bad), player))
    except ValueError as exc:
        assert "corrupt" in str(exc)
    else:
        raise AssertionError("a sha256 mismatch must raise")


def test_token_cut_and_legacy_mode_unchanged(tmp_path):
    config = CrafterEnvConfig(max_episode_steps=10000, segment_tokens=1000, state_dir=str(tmp_path))
    result = asyncio.run(fake_env(config).play_segment(root_task(), ScriptedPlayer(tokens_per_turn=300)))
    assert result["ended_by"] == "segment_tokens" and result["steps"] == 4 and result["continues"]
    # No segment settings: one episode is one game, cut at max_episode_steps, no summary, no state.
    player = ScriptedPlayer()
    result = asyncio.run(fake_env(CrafterEnvConfig(max_episode_steps=4)).play_segment(root_task(), player))
    assert result["ended_by"] == "max_episode_steps" and not result["continues"] and result["state_path"] is None
    assert player.interactions[0].summaries == 0 and result["depth"] == 0


def test_select_children_and_sampling_swap():
    results = [
        {"continues": True, "reward": 0.2, "deaths": []},
        {"continues": True, "reward": 0.9, "deaths": [1]},
        {"continues": False, "reward": 1.0, "deaths": []},
        {"continues": True, "reward": 0.9, "deaths": []},
    ]
    kept = select_children(results, 2)
    assert [r["reward"] for r in kept] == [0.9, 0.9] and kept[0]["deaths"] == []
    # the sibling cap: at most 2 kept per parent, topped up when parents are scarce
    results = [{"continues": True, "reward": 0.9 - 0.1 * i, "deaths": [], "parent": "A" if i < 4 else "B", "i": i} for i in range(8)]
    results.append({"continues": False, "reward": 1.0, "deaths": [], "parent": "C", "i": 8})
    assert [r["i"] for r in select_children(results, 4)] == [0, 1, 2, 3]
    assert [r["i"] for r in select_children(results, 4, per_parent=2)] == [0, 1, 4, 5]
    assert [r["i"] for r in select_children(results, 4, per_parent=1)] == [0, 1, 2, 4]  # 2 parents only: 0 and 4 by the cap, topped up with 1 and 2; rank order
    assert [r["i"] for r in select_children(results, 3, per_parent=3)] == [0, 1, 2]
    one_parent = [{"continues": True, "reward": 0.5 - 0.01 * i, "deaths": [], "parent": None, "i": i} for i in range(16)]
    assert [r["i"] for r in select_children(one_parent, 4, per_parent=2)] == [0, 1, 2, 3]  # the root's level keeps 4

    @dataclasses.dataclass(frozen=True)
    class Ctx:
        sampling: object

    class Sampling:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

        def model_copy(self, update):
            return Sampling(update["max_tokens"])

    session = SimpleNamespace(ctx=Ctx(Sampling(1024)))
    interaction = SimpleNamespace(_run=SimpleNamespace(_session=session))

    async def check():
        async with _turn_sampling(interaction, 32768):
            assert session.ctx.sampling.max_tokens == 32768
        assert session.ctx.sampling.max_tokens == 1024
        async with _turn_sampling(SimpleNamespace(), 5):  # a scripted interaction: nothing to swap
            pass

    asyncio.run(check())


def test_in_env_beam_plays_the_tree(tmp_path):
    log = tmp_path / "episodes.jsonl"
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=3, tree_depth=3, branches=2, keep=1, episode_log=str(log))
    env, player = fake_env(config), ScriptedPlayer()
    asyncio.run(env.run(root_task(), SimpleNamespace(player=player)))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["depth"] for l in lines] == [1, 1, 2, 2, 3, 3]
    assert len({l["parent"] for l in lines if l["depth"] == 2}) == 1  # one kept branch per depth
    assert all(l["game_step"] == 3 * l["depth"] for l in lines)
    assert isinstance(env._store(), MemoryStateStore) and env._store().blobs == {}  # nothing left behind


def test_in_env_beam_fans_out_the_root_and_resumes_from_its_checkpoint(tmp_path):
    log, states = tmp_path / "episodes.jsonl", tmp_path / "states"
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=3, tree_depth=2, branches=2, keep=2, state_dir=str(states), episode_log=str(log))
    asyncio.run(fake_env(config).run(root_task(), SimpleNamespace(player=ScriptedPlayer())))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["depth"] for l in lines] == [1] * 4 + [2] * 4  # the root fans out to branches x keep, then 2 kept nodes x 2
    assert len({l["parent"] for l in lines if l["depth"] == 2}) == 2
    beam = json.loads((states / "beam" / "42.json").read_text())
    assert beam["depth"] == 2 and beam["done"] == "tree_depth" and len(beam["beam"]) == 2  # at the cap the kept nodes stay
    assert all((states / f"{n['id']}.pkl").exists() for n in beam["beam"])
    # the same cap again: nothing to do; a deeper cap: continues from the kept nodes
    asyncio.run(fake_env(config).run(root_task(), SimpleNamespace(player=ScriptedPlayer())))
    assert len(log.read_text().splitlines()) == 8
    asyncio.run(fake_env(config.model_copy(update={"tree_depth": 3})).run(root_task(), SimpleNamespace(player=ScriptedPlayer())))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["depth"] for l in lines] == [1] * 4 + [2] * 4 + [3] * 4 and {l["parent"] for l in lines if l["depth"] == 3} == {n["id"] for n in beam["beam"]}
    # a finished beam (the goal, or no branch left) is never resumed
    import crafter_env.taskset as ts
    ts._write_beam(str(states / "beam" / "42.json"), {"seed": 42, "depth": 3, "beam": [], "done": "final_goal"})
    asyncio.run(fake_env(config.model_copy(update={"tree_depth": 9})).run(root_task(), SimpleNamespace(player=ScriptedPlayer())))
    assert len(log.read_text().splitlines()) == 12
    # an interrupted run: the checkpoint names live nodes
    kept = [{"id": l["node_id"], "parent": l["parent"], "depth": 2, "state_path": l["state_path"], "state_sha256": None, "summary": l["summary"]}
            for l in lines if l["depth"] == 2][:2]
    for l in lines:
        if l["depth"] == 2:
            (states / f"{l['node_id']}.pkl").write_bytes(b"")  # placeholder files for the state_path check
    ts._write_beam(str(states / "beam" / "42.json"), {"seed": 42, "depth": 2, "beam": kept, "done": None})
    for n in kept:  # real worlds for the two kept nodes
        env2 = fake_env(config)
        world = __import__("crafter").Env(seed=42, length=10000)
        world.reset()
        path, sha, _ = env2._store().put(n["id"], {"env": world, "spawn": [32, 32], "last_sleep": [32, 32], "last_base": [32, 32], "seed": 42})
        n["state_path"], n["state_sha256"] = path, sha
    ts._write_beam(str(states / "beam" / "42.json"), {"seed": 42, "depth": 2, "beam": kept, "done": None})
    asyncio.run(fake_env(config.model_copy(update={"tree_depth": 3})).run(root_task(), SimpleNamespace(player=ScriptedPlayer())))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["depth"] for l in lines] == [1] * 4 + [2] * 4 + [3] * 4 + [3] * 4  # resumed at depth 3 from the 2 checkpointed nodes
    assert {l["parent"] for l in lines[-4:]} == {n["id"] for n in kept}
    assert json.loads((states / "beam" / "42.json").read_text())["done"] == "tree_depth"


def test_state_store_backends(tmp_path):
    for store in (DiskStateStore(tmp_path / "s"), MemoryStateStore()):
        path, sha, size = store.put("a", {"x": 1})
        assert store.get(path, sha) == {"x": 1} and size > 0
        store.put("b", {"y": 2})
        assert store.sweep(keep={path}, older_than_s=-1) == 1  # b is swept, a is kept
        assert store.delete(path) and not store.delete(path)


def test_death_inside_a_segment_respawns_at_the_base_and_the_next_segment_keeps_it(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 6:  # the sixth action of the first segment is fatal
            env._player.inventory["wood"] = 2
            env._player.inventory["wood_pickaxe"] = 1
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)
    base = (40, 40)
    original = ts._respawn

    def respawn_spy(env, at, death_inventory, safe=False, safe_radius=2):
        respawn_spy.at, respawn_spy.safe = at, safe
        return original(env, at, death_inventory, safe, safe_radius)

    monkeypatch.setattr(ts, "_respawn", respawn_spy)
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path), on_death="respawn",
                              respawn_at="last_base", respawn_safe=True, death_inventory="drop_materials", death_penalty=0.05)
    env, player = fake_env(config), ScriptedPlayer()
    task = root_task()
    first = asyncio.run(env.play_segment(task, player))
    assert first["ended_by"] == "segment_turns" and len(first["deaths"]) == 1 and first["deaths"][0]["step"] == 6
    assert respawn_spy.safe
    assert respawn_spy.at == first["deaths"][0]["respawn"] or max(abs(a - b) for a, b in zip(respawn_spy.at, first["deaths"][0]["respawn"])) <= 16
    death = first["deaths"][0]
    assert death["cause"] in DEATH_CAUSES and death["since"] == 6 and 0.0 <= death["daylight"] <= 1.0
    assert any("died at" in e and f"({death['cause']}" in e for e in first["events"])
    assert player.interactions[0].trace.metrics[f"deaths_{death['cause']}"] == 1.0
    assert first["inventory"].get("wood", 0) == 0 and first["inventory"].get("wood_pickaxe") == 1  # materials dropped, the tool kept
    assert first["inventory"]["health"] == 9 and first["inventory"]["food"] == 9  # full vitals after the respawn
    told = [m for m in player.interactions[0].received if m and "You died" in m]
    assert len(told) == 1 and "lost your materials (tools kept)" in told[0]
    assert player.interactions[0].trace.rewards["achievements"] == len(first["segment_achievements"]) / 22
    assert player.interactions[0].trace.rewards["deaths"] == -0.05  # its own term: it outlives achievement_weight = 0
    assert first["next_tier"].startswith("collect_wood") or first["next_tier"].startswith("place_table")
    assert first["world_nearby"] and first["cells_visited"] >= 1 and first["inventory_before"] == {} and first["position_before"] == [32, 32]

    # The next segment restores the world after the respawn: achievements, the tool and the base survive.
    node = {"id": first["node_id"], "depth": 1, "state_path": first["state_path"], "state_sha256": first["state_sha256"], "summary": first["summary"]}
    second = asyncio.run(env.play_segment(child_task(task, node), player))
    assert second["achievements_before"] == first["achievements"] and second["inventory_before"].get("wood_pickaxe") == 1
    assert second["game_step"] == first["game_step"] + 8 and second["deaths"] == []
    assert second["position_before"] == first["position"]


def test_death_cause_reads_the_killer_off_the_world():
    import crafter
    import crafter.objects as objects

    from crafter_env.taskset import _death_cause

    env = crafter.Env(seed=3)
    env.reset()
    world, player = env._world, env._player
    for obj in [o for o in world.objects if o is not player]:
        world.remove(obj)
    vitals = {"health": 0, "food": 5, "drink": 5, "energy": 5}
    assert _death_cause(env, vitals) == "unknown"
    assert _death_cause(env, {**vitals, "drink": 0}) == "starved"
    x, y = (int(v) for v in player.pos)
    cell = next(c for c in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if world[c][0] in ("grass", "sand", "path"))
    world.add(objects.Zombie(world, cell, player))
    assert _death_cause(env, {**vitals, "drink": 0}) == "zombie"  # the attacker wins over a vital at 0
    world[tuple(int(v) for v in player.pos)] = "lava"
    assert _death_cause(env, vitals) == "lava"


def test_cut_on_death_ends_the_segment_after_the_respawn(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 3:
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=30, state_dir=str(tmp_path), on_death="respawn", cut_on_death=True)
    env, player = fake_env(config), ScriptedPlayer()
    result = asyncio.run(env.play_segment(root_task(), player))
    assert result["ended_by"] == "death_cut" and result["continues"] and result["steps"] == 3 and len(result["deaths"]) == 1
    assert result["state_path"] and player.interactions[0].summaries == 1
    assert "You died (death 1" in player.interactions[0].received[-1]  # the respawn notice is in the state shown with the notes request


def test_in_env_beam_group_judge_one_call_per_node(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    calls = []

    async def fake_group(judge, infos, **kw):
        calls.append(len(infos))
        return [{"score": 0.1 * i, "failed": False, "rationale": "x", "progress": 0.1 * i, "plan": 0.0, "consistency": 1.0} for i in range(len(infos))], 0.01, None

    monkeypatch.setattr(ts, "group_verdicts", fake_group)
    log = tmp_path / "episodes.jsonl"
    import verifiers.v1 as vf

    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=3, tree_depth=2, branches=3, keep=1, episode_log=str(log),
                              judge=vf.JudgeConfig(base_url="http://localhost:1/v1", api_key_var="X", model="m"), judge_mode="group")
    env, player = fake_env(config), ScriptedPlayer()
    asyncio.run(env.run(root_task(), SimpleNamespace(player=player)))
    assert calls == [3, 3]  # one comparative call per node, one node per depth with keep 1
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    judged = [l for l in lines if l["task"] == "crafter-judge"]
    assert len(judged) == 6 and all(l["mode"] == "group" and l["group_size"] == 3 and l["wait_s"] >= 0 for l in judged)
    seg2 = [l for l in lines if l["task"] == "crafter" and l["depth"] == 2]
    best = max((l for l in judged if l["depth"] == 1), key=lambda l: l["judge_score"])
    assert {l["parent"] for l in seg2} == {best["node_id"]}  # the best-scored branch was expanded


def test_summary_retry_asks_again_when_the_notes_turn_gets_an_action(tmp_path):
    class ActionOnlyPlayer(ScriptedPlayer):
        """Answers the first notes request with an action, the retry with notes."""

        @asynccontextmanager
        async def interaction(self, task):
            async with super().interaction(task) as it:
                inner = it.turn

                async def turn(message=None):
                    if message is not None and (message.startswith(SUMMARY_PROMPT) or message.endswith(SUMMARY_PROMPT)):
                        it.received.append(message)
                        return SimpleNamespace(last_reply="<action>do</action>", terminated=False)
                    if message is not None and message.startswith("END OF SEGMENT, still"):
                        it.received.append(message)
                        return SimpleNamespace(last_reply="1. Near a tree at 2W. 2. Inventory empty. 3. Wandered. 4. Collect wood, then place a table.", terminated=False)
                    return await inner(message)

                it.turn = turn
                yield it

    for retry in (False, True):
        config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=4, state_dir=str(tmp_path), summary_retry=retry, notes_protocol=False)
        player = ActionOnlyPlayer()
        result = asyncio.run(fake_env(config).play_segment(root_task(), player))
        request = [m for m in player.interactions[0].received if m and m.endswith(SUMMARY_PROMPT)][0]
        if retry:
            assert result["summary"].startswith("1. Near a tree") and "Reply only as <action>" not in request
            assert result["summary_retry_reason"] == "action" and player.interactions[0].trace.metrics["summary_retry_action"] == 1.0
        else:
            assert result["summary"] == "" and "Reply only as <action>" in request


    # The notes turns' output cap stops at what the context still holds; no room, no retry.
    import crafter_env.taskset as ts
    import verifiers.v1 as vf

    caps, original = [], ts._turn_sampling

    def spy(interaction, max_tokens):
        caps.append(max_tokens)
        return original(interaction, max_tokens)

    ts._turn_sampling = spy
    try:
        for limit, expected, retried in ((40000, [32768, 32768], True), (1200, [176], False)):
            caps.clear()
            config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=4, state_dir=str(tmp_path), notes_protocol=False,
                                      player=vf.AgentConfig(harness={"id": "inline-chat"}, max_total_tokens=limit))
            result = asyncio.run(fake_env(config).play_segment(root_task(), ActionOnlyPlayer()))
            assert caps == expected and bool(result["summary_retry_reason"]) == retried
    finally:
        ts._turn_sampling = original


def test_notes_protocol_helpers():
    base = "You are playing Crafter."
    sp = with_notes_protocol(f"{base}\n\n{CONTINUATION_HEADER}\n\nold notes")
    assert sp.index(NOTES_PROTOCOL) < sp.index(CONTINUATION_HEADER) and sp.endswith("old notes")
    assert with_notes_protocol(sp) == sp and with_notes_protocol(base).endswith(NOTES_PROTOCOL)
    assert extract_notes("<notes>\n1. tree 2W\n</notes>") == "1. tree 2W"
    assert extract_notes("<notes>1. unterminated") == "1. unterminated"
    assert extract_notes("1. plain <action>do</action>") == "1. plain"
    long = "1. At (31, 37); trees at (33, 34) and (35, 38). 2. Nothing. 3. Died to a zombie. 4. Collect wood."
    assert notes_problem("<action>do</action>", "") == "action"
    assert notes_problem(f"<notes>{long}</notes><action>do</action>", long) is None  # full notes: the action is ignored
    assert notes_problem("1. tree <action>do</action>", "1. tree") == "action"
    assert notes_problem("<notes>1. tree</notes><action>do</action>", "1. tree") == "short"
    assert notes_problem(long, long) == "no_notes_tag" and notes_problem(long, long, protocol=False) is None
    assert notes_problem("<notes>1. tree</notes>", "1. tree") == "short"
    assert notes_problem(f"<notes>{long}</notes>", long) is None
    assert "world coordinates" in SUMMARY_PROMPT and "columns right of @" in SUMMARY_PROMPT


def test_death_with_revert_reloads_the_segment_start(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 6:  # the sixth action is fatal, with loot that the revert must undo
            env._player.inventory["wood"] = 2
            env._player.inventory["wood_pickaxe"] = 1
            env._player.achievements["collect_wood"] = 1
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy(), "achievements": env._player.achievements.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path), on_death="revert", death_penalty=0.05)
    env, player = fake_env(config), ScriptedPlayer()
    first = asyncio.run(env.play_segment(root_task(), player))
    assert first["ended_by"] == "segment_turns" and len(first["deaths"]) == 1
    death = first["deaths"][0]
    assert death["step"] == 6 and death["revert_step"] == 0 and death["lost_turns"] == 6 and death["cause"] in DEATH_CAUSES
    assert death["respawn"] == [32, 32] == first["position_before"]  # back at the spawn, where the segment began
    assert first["game_step"] == 2  # 8 turns played, 6 undone
    assert first["inventory"].get("wood_pickaxe", 0) == 0 and first["inventory"]["health"] == 9  # the loot of the lost life is gone
    assert first["segment_achievements"] == [] and first["unlocked_at"] == {}
    assert any("died at" in e and "restored to step 0" in e for e in first["events"])
    told = [m for m in player.interactions[0].received if m and "You died" in m]
    assert len(told) == 1 and "restored to game step 0 (the start of this segment)" in told[0] and "Last action" not in told[0].split("\n\n", 1)[1].split("Local map")[0]
    assert player.interactions[0].trace.rewards["deaths"] == -0.05
    assert player.interactions[0].trace.metrics["reverted_turns"] == 6.0
    assert first["on_death"] == "revert" and first["revert_to"] == "segment_start"


def test_death_with_revert_to_last_base_reloads_the_sleep(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def scripted_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 3:  # the player falls asleep on the third action: a save point
            env._player.sleeping = True
        if len(calls) == 6:
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", scripted_step)
    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path), on_death="revert", revert_to="last_base")
    env, player = fake_env(config), ScriptedPlayer()
    first = asyncio.run(env.play_segment(root_task(), player))
    death = first["deaths"][0]
    assert death["revert_step"] == 3 and death["lost_turns"] == 3
    assert any("save point (sleep)" in e for e in first["events"])
    told = [m for m in player.interactions[0].received if m and "You died" in m]
    assert "restored to game step 3 (your last save point)" in told[0]
    assert first["game_step"] == 5  # 3 kept + 2 after the revert


def test_respawn_line_names_the_cause_when_asked(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 6:
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)
    for announce in (False, True):
        calls.clear()  # the sixth action of each run is the fatal one
        config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path), on_death="respawn",
                                  respawn_at="last_base", announce_death_cause=announce)
        env, player = fake_env(config), ScriptedPlayer()
        first = asyncio.run(env.play_segment(root_task(), player))
        told = [m for m in player.interactions[0].received if m and "You died" in m][0]
        death = first["deaths"][0]
        hows = {"zombie": "killed by a zombie", "skeleton": "shot by a skeleton", "lava": "fell into lava", "starved": "starved:", "unknown": "cause unknown"}
        if announce:
            assert hows[death["cause"]] in told and f"game step {death['step']}" in told and "You respawned at" in told
            assert (" at night" in told) == (death["daylight"] < 0.5)
        else:
            assert "game step" not in told and "cause unknown" not in told and "killed by" not in told
        assert first["announce_death_cause"] is announce


def test_death_lesson_is_asked_once_and_pinned(tmp_path, monkeypatch):
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 4:
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)

    class LessonPlayer(ScriptedPlayer):
        @asynccontextmanager
        async def interaction(self, task):
            async with super().interaction(task) as interaction:
                scripted = interaction.turn

                async def turn(message=None):
                    reply = await scripted(message)
                    if message and LESSON_REQUEST in message:
                        reply.last_reply = "<lesson>A zombie killed me at night; next time I will\n  stay away from zombies and sleep earlier.</lesson>\n<action>move_right</action>"
                    return reply

                interaction.turn = turn
                yield interaction

    for lesson in (False, True):
        calls.clear()
        config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=9, state_dir=str(tmp_path), on_death="respawn",
                                  respawn_at="last_base", death_lesson=lesson, lessons_kept=6)
        env, player = fake_env(config), LessonPlayer()
        first = asyncio.run(env.play_segment(root_task(), player))
        received = player.interactions[0].received
        asked = [m for m in received if m and LESSON_REQUEST in m]
        system = player.interactions[0].task.data.system_prompt
        assert first["death_lesson"] is lesson and len(first["deaths"]) == 1
        if not lesson:
            assert not asked and LESSON_PROTOCOL not in system and not any(LESSON_HEADER in (m or "") for m in received)
            assert first["lessons"] == [] and first["lessons_missing"] == 0
            continue
        assert LESSON_PROTOCOL in system and NOTES_PROTOCOL in system and system.index(NOTES_PROTOCOL) < system.index(LESSON_PROTOCOL)
        assert len(asked) == 1 and "You died" in asked[0] and asked[0].rstrip().endswith("Then choose the next action as <action>action_name</action>.")
        assert "Reply only as" not in asked[0].rsplit("\n", 1)[-1]
        assert LESSON_HEADER not in asked[0]  # no lesson yet at the death turn
        after = [m for m in received[received.index(asked[0]) + 1:] if m and not m.startswith(SUMMARY_PROMPT)]
        assert after and all(LESSON_HEADER in m for m in after)  # the end-of-segment warning may sit above the block
        assert "- after death 1: A zombie killed me at night; next time I will stay away from zombies and sleep earlier." in after[0]
        assert first["lessons"] == [{"death": 1, "turn": 5, "chars": 87, "text": "A zombie killed me at night; next time I will stay away from zombies and sleep earlier."}]
        assert first["lessons_missing"] == 0
        assert any("lesson after death 1" in e for e in first["events"])
        assert player.interactions[0].trace.metrics["lessons"] == 1.0 and player.interactions[0].trace.metrics["lesson_chars"] == 87.0
        assert first["summary"].startswith("Notes:")  # the notes turn still comes at the cut


def test_lesson_helpers():
    assert extract_lesson("<lesson> one\n two </lesson><action>do</action>") == "one two"
    assert extract_lesson("<lesson>unclosed lesson\n<action>do</action>") == "unclosed lesson"
    assert extract_lesson("<action>do</action>") == "" and extract_lesson("") == ""
    sp = with_lesson_protocol(with_notes_protocol("rules"))
    assert sp == with_lesson_protocol(sp) and sp.startswith("rules") and sp.index(NOTES_PROTOCOL) < sp.index(LESSON_PROTOCOL)
    cont = with_lesson_protocol(f"rules\n\n{CONTINUATION_HEADER}\n\nold notes")
    assert cont.index(LESSON_PROTOCOL) < cont.index(CONTINUATION_HEADER) and cont.endswith("old notes")


def test_notes_masked_out_of_the_loss(tmp_path):
    """train_on_notes = False masks the notes turn and any mid-segment <notes> reply out of
    the loss; the default leaves every sampled node trainable."""

    class ChattyPlayer(ScriptedPlayer):
        def action_reply(self, interaction, reply):
            if len(interaction.received) == 3:  # the third action turn thinks out loud first
                return "<notes>\nStuck going north; turn back south.\n</notes>\n\n" + reply
            return reply

    for train_on_notes in (True, False):
        player = ChattyPlayer()
        config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=6, state_dir=str(tmp_path), train_on_notes=train_on_notes)
        env = fake_env(config)
        result = asyncio.run(env.play_segment(root_task(), player))
        trace = player.interactions[0].trace
        sampled = [n for n in trace.nodes if n.sampled]
        assert len(sampled) == 7 and "<notes>" in sampled[2].message["content"] and "<action>" in sampled[2].message["content"]
        assert sampled[-1].message["content"].startswith("<notes>")  # the notes turn
        trainable = [any(n.mask) for n in sampled]
        if train_on_notes:
            assert all(trainable) and trace.metrics["notes_masked_tokens"] == 0.0 and result["notes_masked"] == {"turn": 0, "mid": 0, "tokens": 0}
        else:
            assert trainable == [True, True, False, True, True, True, False]
            assert trace.metrics["notes_masked_replies"] == 2.0 and trace.metrics["notes_masked_mid"] == 1.0 and trace.metrics["notes_masked_tokens"] == 6.0
            assert result["train_on_notes"] is False and result["notes_masked"] == {"turn": 1, "mid": 1, "tokens": 6}
            assert "<notes>" in sampled[2].message["content"]  # the text stays in the context, only the loss mask changes
            dropped = [i for i, n in enumerate(trace.nodes) if n.sampled and not any(n.mask)]
            assert trace.info["loss_mask"] == {"drop_nodes": dropped} and len(dropped) == 2  # what the orchestrator applies to its copy


def test_mask_notes_helper_without_a_summary_turn():
    """A segment that ends the lineage has no notes turn: only mid-segment <notes> replies are masked."""
    nodes = [
        SimpleNamespace(message={"role": "assistant", "content": "<action>do</action>"}, sampled=True, mask=[True, True]),
        SimpleNamespace(message={"role": "assistant", "content": "<notes>plan</notes><action>do</action>"}, sampled=True, mask=[True, True, True]),
        SimpleNamespace(message={"role": "user", "content": "<notes> in an observation"}, sampled=False, mask=[False]),
    ]
    trace = SimpleNamespace(nodes=nodes, info={})
    out = mask_notes_from_loss(trace, 10**9)
    assert out == {"turn": 0, "mid": 1, "tokens": 3} and nodes[0].mask == [True, True] and nodes[1].mask == [False] * 3 and nodes[2].mask == [False]
    assert trace.info == {"loss_mask": {"drop_nodes": [1]}}


def test_recall_probe_asks_before_the_notes_and_scores_the_reply(tmp_path, monkeypatch):
    """recall_probe: a MEMORY CHECK turn right before the notes request, declared in the system
    prompt, scored against the segment's record and masked out of the loss."""
    import crafter_env.taskset as ts

    step, calls = ts._step, []

    def lethal_step(env, action):
        reward, done, info = step(env, action)
        calls.append(action)
        if len(calls) == 6:
            env._player.health = 0
            return reward, True, {**info, "inventory": env._player.inventory.copy()}
        return reward, done, info

    monkeypatch.setattr(ts, "_step", lethal_step)

    class ProbedPlayer(ScriptedPlayer):
        def action_reply(self, interaction, reply):
            if interaction.received[-1].startswith(RECALL_PROBE):
                return ("deaths: 2\ndeath: step=7, cause=zombie\ndeath: step=3, cause=lava\n"
                        "unlocked: collect_wood, step=2\nunlocked: place_table, step=5")
            return reply

    config = CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path), on_death="respawn", recall_probe=True)
    env, player = fake_env(config), ProbedPlayer()
    result = asyncio.run(env.play_segment(root_task(), player))
    interaction = player.interactions[0]
    probes = [i for i, m in enumerate(interaction.received) if m and m.startswith(RECALL_PROBE)]
    notes = [i for i, m in enumerate(interaction.received) if m and m.startswith(SUMMARY_PROMPT)]
    assert len(probes) == 1 and len(notes) == 1 and probes[0] == notes[0] - 1
    assert RECALL_PROTOCOL in interaction.task.data.system_prompt and "Step:" in interaction.received[probes[0]]
    recall, death = result["recall"], result["deaths"][0]
    assert death["game_step"] == 6 and death["step"] == 6
    assert recall["parsed"] and recall["deaths_true"] == 1 and recall["deaths_claimed"] == 2 and not recall["count_correct"]
    assert recall["deaths"] == [{"game_step": 6, "cause": death["cause"], "turns_ago": 2, "matched": True, "cause_ok": death["cause"] == "zombie"}]
    assert recall["false_deaths"] == 1  # step 3 is within tolerance too, but step 7 is closer and a claim is used once
    assert recall["achievements_true"] == len(result["unlocked_at"])
    assert recall["false_achievements"] == sum(n not in result["unlocked_at"] for n in ("collect_wood", "place_table"))
    trace = interaction.trace
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    probe_index = sampled[-2]
    assert not any(trace.nodes[probe_index].mask) and any(trace.nodes[sampled[-1]].mask) and any(trace.nodes[sampled[-3]].mask)
    assert trace.info["loss_mask"] == {"drop_nodes": [probe_index]}
    assert trace.metrics["recall_parsed"] == 1.0 and trace.metrics["recall_deaths_matched"] == 1.0 and trace.metrics["recall_false_deaths"] == 1.0
    assert trace.metrics["recall_deaths_true"] == 1.0 and trace.metrics["recall_count_correct"] == 0.0
    # the default: no probe, no recall field, no mask request
    env, player = fake_env(CrafterEnvConfig(max_episode_steps=10000, segment_turns=8, state_dir=str(tmp_path))), ScriptedPlayer()
    result = asyncio.run(env.play_segment(root_task(), player))
    assert result["recall"] is None and not any(m and m.startswith(RECALL_PROBE) for m in player.interactions[0].received)
    assert "loss_mask" not in player.interactions[0].trace.info and "recall_parsed" not in player.interactions[0].trace.metrics


def test_score_recall_probe_helper():
    deaths = [{"step": 30, "game_step": 430, "cause": "skeleton"}, {"step": 120, "game_step": 520, "cause": "zombie"}]
    unlocked = {"collect_stone": 450}
    r = score_recall_probe("deaths: 2\ndeath: step=425, cause=skeleton arrow\ndeath: step=600, cause=zombie\n"
                           "unlocked: collect_stone, step=455\nunlocked: eat_cow, step=500", deaths, unlocked, 200)
    assert r["parsed"] and r["count_correct"] and [d["matched"] for d in r["deaths"]] == [True, False]
    assert r["deaths"][0]["cause_ok"] and r["deaths"][0]["turns_ago"] == 170 and r["false_deaths"] == 1
    assert r["achievements_step_ok"] == 1 and r["false_achievements"] == 1
    r = score_recall_probe("<action>noop</action>", deaths, unlocked, 200)
    assert not r["parsed"] and r["deaths_claimed"] is None and not r["count_correct"] and r["false_deaths"] == 0
    assert [d["matched"] for d in r["deaths"]] == [False, False]
    r = score_recall_probe("deaths: 0", [], {}, 50)
    assert r["parsed"] and r["count_correct"] and r["false_deaths"] == 0 and r["deaths"] == []
    r = score_recall_probe("deaths: unknown\ndeath: step=unknown, cause=lava", deaths, unlocked, 200)
    assert r["parsed"] and r["deaths_claimed"] == 1 and not r["count_correct"] and r["false_deaths"] == 1

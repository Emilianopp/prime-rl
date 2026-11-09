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


@pytest.mark.parametrize("death_inventory", ["drop_all", "keep", "drop_grave"])
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
    assert ("Wood:0" in after) == (death_inventory in ("drop_all", "drop_grave"))
    if death_inventory == "drop_grave":
        assert "dropped your inventory (wood 5) at (" in after and "on floor 1: walk onto that cell by step 305" in after
        assert line["graves"]["dropped"] == 2 and all(d["dropped"] for d in line["deaths"])
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


def test_a_game_resumes_from_its_saved_state(tmp_path, monkeypatch):
    """Two jobs of a chain: the first stops at its time limit after asking for a handoff
    summary and saves the game; the second opens with that summary and plays on."""
    import craftax_env.taskset as taskset
    import verifiers.v1 as vf
    from verifiers.v1.configs.agent import SegmentsConfig

    clock = iter(range(10**6))  # one second per reading
    monkeypatch.setattr(taskset, "time", SimpleNamespace(time=lambda: float(next(clock))))
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, respawn=True)).load()))
    player = vf.AgentConfig(harness={"id": "inline-chat"}, segments=SegmentsConfig(new_tokens=1000, summary_tokens=100))
    opened, asked = [], []

    class Resumable(ScriptedInteraction):
        def __init__(self):
            super().__init__()
            self.trace.nodes = []

        async def summarize(self, message):
            asked.append(message)
            return f"summary after {len(self.received)} replies"

    def job(log):
        interaction = Resumable()

        @asynccontextmanager
        async def open_interaction(_task, **kwargs):
            opened.append(kwargs)
            yield interaction

        env = SimpleNamespace(config=CraftaxEnvConfig(
            player=player, episode_log=str(log), state_dir=str(tmp_path), save_every=2,
            max_episode_steps=0, max_episode_seconds=12, on_death="respawn",
        ))
        asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
        return interaction, json.loads(log.read_text())

    first, line1 = job(tmp_path / "a.jsonl")
    assert line1["ended_by"] == "time_limit" and line1["block"] == 0 and line1["turns"] > 0
    assert opened[0] == {} and len(asked) == 1
    assert asked[0].startswith(first.received[-1])  # the pending observation, shown again
    second, line2 = job(tmp_path / "b.jsonl")
    assert opened[1] == {"summary": f"summary after {len(first.received)} replies"}
    assert second.received[0] == first.received[-1]  # the same observation opens the next job
    assert line2["block"] == 1 and line2["turns"] > line1["turns"]
    assert line2["game_steps"] > line1["game_steps"] and line2["played_seconds"] > line1["played_seconds"]


def test_the_player_memory_is_saved_and_resumed(tmp_path, monkeypatch):
    """With player.memory, each job hands the interaction the game's Memory (fresh, then
    the saved one), and the episode log reports its entries and edits."""
    import craftax_env.taskset as taskset
    import verifiers.v1 as vf
    from verifiers.v1.configs.agent import MemoryConfig, SegmentsConfig

    clock = iter(range(10**6))
    monkeypatch.setattr(taskset, "time", SimpleNamespace(time=lambda: float(next(clock))))
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, respawn=True)).load()))
    player = vf.AgentConfig(
        harness={"id": "inline-chat"}, memory=MemoryConfig(),
        segments=SegmentsConfig(new_tokens=1000, summary_tokens=100),
    )
    memories = []

    class Resumable(ScriptedInteraction):
        def __init__(self):
            super().__init__()
            self.trace.nodes = []

        async def summarize(self, message):
            return "s"

    def job(log):
        interaction = Resumable()

        @asynccontextmanager
        async def open_interaction(_task, **kwargs):
            memory = kwargs["memory"]
            memories.append(dict(memory.entries))
            memory.apply(f'<write key="map">job {len(memories)}</write>')
            yield interaction

        env = SimpleNamespace(config=CraftaxEnvConfig(
            player=player, episode_log=str(log), state_dir=str(tmp_path), save_every=2,
            max_episode_steps=0, max_episode_seconds=12, on_death="respawn",
        ))
        asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
        return json.loads(log.read_text())

    line1 = job(tmp_path / "a.jsonl")
    line2 = job(tmp_path / "b.jsonl")
    assert memories == [{}, {"map": "job 1"}]
    assert line1["memory"] == {"tokens": {"map": 2}, "ops": {"write ok": 1}}
    assert line2["memory"]["tokens"] == {"map": 3}  # "job 1\njob 2"


def test_episodes_and_variants():
    tasks = list(CraftaxTaskset(CraftaxConfig(split="eval", episodes=[3, 7], variants=2)).load())
    assert [t.data.name for t in tasks] == ["craftax#eval3v0", "craftax#eval3v1", "craftax#eval7v0", "craftax#eval7v1"]
    assert [(t.data.info["seed"], t.data.info["variant"]) for t in tasks] == [
        (1_000_045, 0), (1_000_045, 1), (1_000_049, 0), (1_000_049, 1)]
    plain = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1)).load()))
    assert "variant" not in plain.data.info


def test_a_variant_saves_on_its_own_and_stops_after_its_segments(tmp_path):
    """A variant keeps its own save, and ``max_segments`` stops the run at the next
    segment's first turn without playing that reply."""
    task = list(CraftaxTaskset(CraftaxConfig(num_examples=1, variants=2, respawn=True)).load())[1]

    class TwoSegments(ScriptedInteraction):
        @property
        def segments(self):
            return 1 + (len(self.received) > 3)  # the 4th turn opens segment 2

    interaction = TwoSegments()

    @asynccontextmanager
    async def open_interaction(_task, **kwargs):
        yield interaction

    log = tmp_path / "episodes.jsonl"
    env = SimpleNamespace(config=CraftaxEnvConfig(
        episode_log=str(log), state_dir=str(tmp_path), max_segments=1, max_episode_steps=0, on_death="respawn",
    ))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    line = json.loads(log.read_text())
    assert line["ended_by"] == "segment_limit" and line["turns"] == 3
    assert (tmp_path / f"{task.data.info['seed']}-1.pkl").exists()
    assert not (tmp_path / f"{task.data.info['seed']}.pkl").exists()


def test_segment_boundaries_are_written_for_the_judge(tmp_path):
    """With ``state_log`` the env writes the true state where each segment starts and
    ends, the end naming the next segment, for craftax_env.qv.score_segment."""
    task = next(iter(CraftaxTaskset(CraftaxConfig(num_examples=1, respawn=True)).load()))

    class TwoTraces(ScriptedInteraction):
        @property
        def trace(self):
            return self.traces[0] if len(self.received) <= 3 else self.traces[1]  # the 4th turn opens seg-b

        @trace.setter
        def trace(self, first):
            first.id = "seg-a"
            self.traces = [first, SimpleNamespace(**{**vars(first), "id": "seg-b"})]

        @property
        def segments(self):
            return 1 + (len(self.received) > 3)

    interaction = TwoTraces()

    @asynccontextmanager
    async def open_interaction(_task, **kwargs):
        yield interaction

    folder = tmp_path / "states"
    env = SimpleNamespace(config=CraftaxEnvConfig(
        episode_log=str(tmp_path / "e.jsonl"), state_log=str(folder), max_segments=1, max_episode_steps=0,
        on_death="respawn",
    ))
    asyncio.run(CraftaxEnv.run(env, task, SimpleNamespace(player=SimpleNamespace(interaction=open_interaction))))
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["seg-a.end.json", "seg-a.start.json", "seg-b.end.json", "seg-b.start.json"]
    a_end = json.loads((folder / "seg-a.end.json").read_text())
    b_start = json.loads((folder / "seg-b.start.json").read_text())
    assert a_end["next"] == "seg-b" and a_end["state"] == b_start["state"]
    assert json.loads((folder / "seg-b.end.json").read_text())["next"] is None
    state = b_start["state"]
    assert state["hud"].startswith("Step:") and "Legend" not in state["map"] and len(state["map"].splitlines()) == 50


def test_respawn_prompt_with_drops():
    system = glyph.system_prompt(True, 2048, "freeze", respawn=True, graves=(300, 3))
    assert f"shows on the map as {glyph.GRAVE_GLYPH}" in system and "within 300 steps" in system and "3 most recent" in system


def _armed(game, **items):
    st = game.state
    inv = st.inventory.replace(**{k: getattr(st.inventory, k) * 0 + v for k, v in items.items()})
    game.state = st.replace(inventory=inv)


def _walkable_near(game, r, c):
    level = glyph.np.asarray(game.state.map[0])
    mobs = glyph.np.asarray(game.state.mob_map[0])
    for dr, dc in ((0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, -1)):
        if not game._rt["not_walkable"][level[r + dr, c + dc]] and not mobs[r + dr, c + dc]:
            return r + dr, c + dc
    raise AssertionError("no walkable cell near the spawn")


def test_a_death_drops_the_inventory_where_it_happened_and_walking_there_picks_it_up():
    game = glyph.CraftaxGame(seed=7, max_turns=None)
    jnp = game._rt["jax"].numpy
    spot = _walkable_near(game, *game.spawn)
    game.state = game.state.replace(player_position=jnp.asarray(spot, dtype=game.state.player_position.dtype))
    _armed(game, wood=5, pickaxe=2, sword=1)
    game.turn = 40
    game.respawn("drop_grave")
    assert game.dropped["pos"] == spot and game.dropped["floor"] == 0 and game.dropped["at"] == 40
    assert int(game.state.inventory.wood) == 0 and int(game.state.inventory.pickaxe) == 0
    legend, grid, _, _ = game.render()  # the drop is 2 cells from the respawn point: in view
    assert f"{glyph.GRAVE_GLYPH} — {glyph.GRAVE_LEGEND}" in legend
    assert f"dropped items (yours: walk onto the cell to pick them up) ({spot[0]},{spot[1]})" in glyph.visible(legend, grid, game.position)
    rows = grid.split("\n")
    r, c = game.position
    assert rows[spot[0] - r + len(rows) // 2][spot[1] - c + len(rows[0]) // 2] == glyph.GRAVE_GLYPH
    # picked up by standing on the cell after a step; counts add, tools keep the better one
    _armed(game, wood=2, sword=2)
    game.state = game.state.replace(player_position=jnp.asarray(spot, dtype=game.state.player_position.dtype))
    step = game.step(glyph.NOOP)
    assert "You picked up the items you dropped at" in step.note and "wood 5" in step.note
    inv = game.state.inventory
    assert (int(inv.wood), int(inv.pickaxe), int(inv.sword)) == (7, 2, 2)
    assert game.graves == [] and game.grave_stats["recovered"] == 1
    assert glyph.GRAVE_GLYPH not in game.render()[1]


def test_drops_expire_are_capped_and_survive_a_save():
    game = glyph.CraftaxGame(seed=7, max_turns=None)
    game.grave_ttl, game.grave_limit = 50, 3
    for k in range(4):
        _armed(game, wood=k + 1)
        game.turn = 10 * k
        game.respawn("drop_grave")
    assert [g["items"]["wood"] for g in game.graves] == [2, 3, 4] and game.grave_stats["replaced"] == 1
    game.respawn("drop_grave")  # nothing carried: no drop
    assert game.dropped is None and game.grave_stats["dropped"] == 4
    other = glyph.CraftaxGame(seed=7, max_turns=None)
    other.restore(game.dump())
    assert other.graves == game.graves and other.grave_stats == game.grave_stats
    other.grave_ttl = 50
    other.turn = 69  # the drop made at turn 10 is 59 turns old, the others at most 49
    other.forfeit()
    assert [g["items"]["wood"] for g in other.graves] == [3, 4] and other.grave_stats["expired"] == 1


def test_ladder_feedback_says_why_descend_or_ascend_did_nothing():
    game = glyph.CraftaxGame(seed=7, max_turns=None)
    rt, jnp = game._rt, game._rt["jax"].numpy
    r, c = game.position
    def at(item, floor=0, kills=0):
        st = game.state
        game.state = st.replace(
            player_level=st.player_level * 0 + floor,
            item_map=st.item_map.at[floor, r, c].set(item),
            monsters_killed=st.monsters_killed.at[floor].set(kills),
        )
        return game.snapshot()
    down, up = glyph.action_index("DESCEND"), glyph.action_index("ASCEND")
    snap = at(0)
    assert "not on a ladder down (▼)" in game.feedback(down, snap, snap)
    assert "no floor above the Overworld" in game.feedback(up, snap, snap)
    snap = at(rt["ladder_up"], floor=1)
    assert "you are on the ladder up (△), which only goes up" in game.feedback(down, snap, snap)
    snap = at(min(rt["ladder_down"]), floor=1, kills=5)
    assert "closed: kill 3 more monsters on this floor" in game.feedback(down, snap, snap)
    assert "you are on the ladder down (▼), which only goes down" in game.feedback(up, snap, snap)

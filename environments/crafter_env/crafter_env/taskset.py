"""crafter-env — Crafter, the survival and crafting game, as text; one episode per rollout.

A verifiers v1 port of the kv-eviction Crafter env (a v0 ``MultiTurnEnv``) with the
same prompts, observation rendering, action parsing and reward, so results are
comparable. The env plays the user: the model's ``<action>`` is stepped in the
game and the rendered observation (vitals, inventory, achievements, a local map)
comes back as the next user turn, until the episode ends or ``max_episode_steps``
model turns have been taken. Reward: achievements unlocked / 22.

Uses the lightweight ``crafter`` package directly (no BALROG: its full install
pulls in NLE/MiniHack native builds Crafter does not need).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from typing import Any, Literal

import crafter
from crafter import constants
from pydantic import Field

import verifiers.v1 as vf

from crafter_env import notebook as nb
from crafter_env import workspace as ws

logger = logging.getLogger(__name__)

ACTION_NAMES: tuple[str, ...] = tuple(constants.actions)
ACTION_TO_INDEX: dict[str, int] = {
    action: idx for idx, action in enumerate(ACTION_NAMES)
}
ACHIEVEMENTS: tuple[str, ...] = tuple(constants.achievements)
VITAL_ITEMS: tuple[str, ...] = ("health", "food", "drink", "energy")

SYSTEM_PROMPT = """\
You are playing Crafter, a survival and crafting game.

Goal: unlock as many achievements as possible before the episode ends. Explore,
collect resources, craft tools, place useful objects, manage food/drink/energy,
and avoid dying.

Reply with exactly one action in XML:
<action>action_name</action>

Valid action_name values:
- noop
- move_left, move_right, move_up, move_down
- do
- sleep
- place_stone, place_table, place_furnace, place_plant
- make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe
- make_wood_sword, make_stone_sword, make_iron_sword

Action notes:
- move_up is north, move_down is south, move_left is west, move_right is east.
- do interacts with the tile in front of you: collect resources, drink water,
  eat plants/cows, or attack creatures.
- craft pickaxes and swords near a table; iron tools also need a furnace.
- place_table uses 2 wood. place_furnace uses 4 stone.

Output the XML action."""

# Reasoning variant: the model reasons before acting. Written as a standalone
# prompt so the XMLParser can extract the <action> tag from the end of the
# response.
REASONING_SYSTEM_PROMPT = """\
You are playing Crafter, a survival and crafting game.

Goal: unlock as many achievements as possible before the episode ends. Explore,
collect resources, craft tools, place useful objects, manage food/drink/energy,
and avoid dying.

Valid action names:
- noop
- move_left, move_right, move_up, move_down
- do
- sleep
- place_stone, place_table, place_furnace, place_plant
- make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe
- make_wood_sword, make_stone_sword, make_iron_sword

Action notes:
- move_up is north, move_down is south, move_left is west, move_right is east.
- do interacts with the tile in front of you: collect resources, drink water,
  eat plants/cows, or attack creatures.
- craft pickaxes and swords near a table; iron tools also need a furnace.
- place_table uses 2 wood. place_furnace uses 4 stone.

On every turn, reason before you act. First write 2-4 sentences of reasoning:
read your vitals, the tile ahead, the local map and inventory, recall your
current goal, and decide the single best next action. Then, on the final line,
output that action and nothing else, in this exact format:
<action>action_name</action>"""

# With memory tools on, the reply-format lines must allow them next to the action:
# "exactly one action" / "reply only as <action>" otherwise tell the model not to use them.
def _tools_reply_format(tools: str) -> dict[str, str]:
    return {
        "Reply with exactly one action in XML:": f"Reply with exactly one action in XML (plus any {tools} tools, below):",
        "Output the XML action.": f"Output the XML action, and any {tools} tools.",
        "output that action and nothing else, in this exact format:": (
            f"output that action in this exact format ({tools} tools, if any, go before it):"
        ),
    }


def _admit_tools(system: str, tools: str) -> str:
    replaced = 0
    for old, new in _tools_reply_format(tools).items():
        replaced += old in system
        system = system.replace(old, new)
    assert replaced, f"the system prompt's reply format must admit the {tools} tools"
    return system


_CLOSINGS = {
    None: "Choose the next action. Reply only as <action>action_name</action>.",
    "notebook": (
        "Choose the next action as <action>action_name</action>, and write to your notebook "
        "if you learned something worth keeping."
    ),
    "workspace": (
        "Choose the next action as <action>action_name</action>, and use your workspace "
        "tools if they help."
    ),
}

_ACTION_ALIASES: dict[str, str] = {
    "wait": "noop",
    "no_op": "noop",
    "move_west": "move_left",
    "west": "move_left",
    "left": "move_left",
    "move_east": "move_right",
    "east": "move_right",
    "right": "move_right",
    "move_north": "move_up",
    "north": "move_up",
    "up": "move_up",
    "move_south": "move_down",
    "south": "move_down",
    "down": "move_down",
    "place_table": "place_table",
    "place_workbench": "place_table",
}

_ID_TO_CHAR: dict[int, str] = {
    0: " ",
    1: "~",  # water
    2: ".",  # grass
    3: "s",  # stone
    4: "=",  # path
    5: ",",  # sand
    6: "T",  # tree
    7: "^",  # lava
    8: "c",  # coal
    9: "i",  # iron
    10: "d",  # diamond
    11: "B",  # table
    12: "F",  # furnace
    13: "@",  # player
    14: "C",  # cow
    15: "Z",  # zombie
    16: "S",  # skeleton
    17: "*",  # arrow
    18: "P",  # plant
}
_MAP_LEGEND = (
    "@ player, . grass, ~ water, T tree, s stone, = path, , sand, "
    "c coal, i iron, d diamond, B table, F furnace, P plant, C cow, "
    "Z zombie, S skeleton, ^ lava"
)


def _canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _normalize_action(text: str | None) -> str | None:
    if not text:
        return None
    key = _canonical_key(text)
    if key in ACTION_TO_INDEX:
        return key
    return _ACTION_ALIASES.get(key)


def _player_info(env: crafter.Env, reward: float = 0.0) -> dict[str, Any]:
    player = getattr(env, "_player", None)
    if player is None:
        return {
            "inventory": {},
            "achievements": {},
            "player_pos": [0, 0],
            "facing": [0, 1],
            "semantic": None,
            "reward": float(reward),
        }
    semantic = None
    sem_view = getattr(env, "_sem_view", None)
    if sem_view is not None:
        try:
            semantic = sem_view()
        except Exception:
            semantic = None
    return {
        "inventory": {
            str(k): int(v) for k, v in getattr(player, "inventory", {}).items()
        },
        "achievements": {
            str(k): int(v) for k, v in getattr(player, "achievements", {}).items()
        },
        "player_pos": [int(x) for x in getattr(player, "pos", [0, 0])],
        "facing": [int(x) for x in getattr(player, "facing", [0, 1])],
        "semantic": semantic,
        "reward": float(reward),
    }


def _direction_name(facing: list[int]) -> str:
    return {
        (-1, 0): "west",
        (1, 0): "east",
        (0, -1): "north",
        (0, 1): "south",
    }.get(tuple(facing), str(tuple(facing)))


def _describe_ahead(env: crafter.Env) -> str:
    player = getattr(env, "_player", None)
    world = getattr(env, "_world", None)
    if player is None or world is None:
        return "unknown"
    try:
        target = player.pos + player.facing
        material, obj = world[target]
    except Exception:
        return "unknown"
    obj_name = type(obj).__name__.lower() if obj is not None else "nothing"
    return f"{material or 'void'} / {obj_name}"


def _format_local_map(info: dict[str, Any], radius: int) -> str:
    semantic = info.get("semantic")
    pos = info.get("player_pos")
    if pos is None:
        pos = [0, 0]
    if semantic is None:
        return "(semantic map unavailable)"
    px, py = int(pos[0]), int(pos[1])
    rows: list[str] = []
    width, height = semantic.shape
    for y in range(py - radius, py + radius + 1):
        chars: list[str] = []
        for x in range(px - radius, px + radius + 1):
            if x == px and y == py:
                chars.append("@")
            elif x < 0 or y < 0 or x >= width or y >= height:
                chars.append(" ")
            else:
                chars.append(_ID_TO_CHAR.get(int(semantic[x, y]), "?"))
        rows.append("".join(chars))
    return "\n".join(rows)


def _format_inventory(inventory: dict[str, int]) -> tuple[str, str]:
    vitals = ", ".join(
        f"{name}={int(inventory.get(name, 0))}" for name in VITAL_ITEMS
    )
    resources = {
        key: value
        for key, value in inventory.items()
        if key not in VITAL_ITEMS and int(value) != 0
    }
    if resources:
        resource_text = ", ".join(
            f"{key}={int(resources[key])}" for key in sorted(resources)
        )
    else:
        resource_text = "none"
    return vitals, resource_text


def _format_observation(
    *,
    env: crafter.Env,
    info: dict[str, Any],
    step: int,
    max_steps: int,
    view_radius: int,
    last_action: str | None = None,
    invalid_action: str | None = None,
    done: bool = False,
    tools: str | None = None,
) -> str:
    inventory = info.get("inventory") or {}
    achievements = info.get("achievements") or {}
    vitals, resources = _format_inventory(inventory)
    unlocked = sorted(
        name for name, count in achievements.items() if int(count) > 0
    )
    unlocked_text = ", ".join(unlocked) if unlocked else "none"
    facing = _direction_name(info.get("facing") or [0, 1])
    pos = info.get("player_pos")
    if pos is None:
        pos = [0, 0]

    lines = [
        f"Step: {step}/{max_steps}",
        f"Position: ({int(pos[0])}, {int(pos[1])}), facing {facing}",
        f"Ahead: {_describe_ahead(env)}",
        f"Vitals: {vitals}",
        f"Inventory: {resources}",
        f"Unlocked achievements ({len(unlocked)}/{len(ACHIEVEMENTS)}): {unlocked_text}",
        f"Last reward: {float(info.get('reward', 0.0)):.3f}",
    ]
    if last_action is not None:
        lines.append(f"Last action: {last_action}")
    if invalid_action is not None:
        lines.append(
            "Invalid action received: "
            f"{invalid_action!r}. It was treated as noop. Use one valid XML action."
        )
    if done:
        lines.append("Episode ended.")
    lines.extend(
        [
            "",
            "Local map, north is up:",
            _format_local_map(info, view_radius),
            f"Legend: {_MAP_LEGEND}",
            "",
            _CLOSINGS[tools],
        ]
    )
    return "\n".join(lines)


_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


class CrafterData(vf.TaskData):
    info: dict
    """The episode's world seed."""


class CrafterTask(vf.Task[CrafterData]):
    pass


class CrafterConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    num_examples: int = Field(5000, ge=1)
    seed: int = 42
    """Episode i of the train split uses world seed ``seed + i`` (eval: ``seed + 1_000_000 + i``)."""
    max_episode_steps: int = Field(100, ge=1)
    """Game length; shown to the model in every observation (match the env's cap)."""
    view_radius: int = Field(4, ge=1)
    include_initial_obs_in_system: bool = False
    """Put the opening observation in the system prompt (the user turn is then "What action do you take?")."""
    enable_reasoning: bool = False
    """Use the prompt that asks for 2-4 sentences of reasoning before the action."""
    notebook: bool = False
    """Give the model notebook tools (memories + scratchpad, crafter_env.notebook),
    explained in the system prompt and shown at the top of every observation."""
    workspace: bool = False
    """Give the model a queryable workspace (crafter_env.workspace): markdown files it
    writes and a ledger the env writes for it, one search tool over both. Excludes
    `notebook`."""
    workspace_files: bool = True
    """The model-written files (off: ledger only)."""
    workspace_ledger: bool = True
    """The env-written ledger (off: files only)."""
    workspace_file_chars: int = Field(2000, ge=100)
    notebook_slots: int = Field(10, ge=1)
    memory_chars: int = Field(120, ge=1)
    scratchpad_chars: int = Field(400, ge=1)


class CrafterEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "inline-chat"})
    max_episode_steps: int = Field(100, ge=1)
    """Model turns (game actions) before the episode is cut off; also the world's length."""
    view_radius: int = Field(4, ge=1)
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, reward, achievements) to this file."""
    on_death: Literal["end", "respawn"] = "end"
    """`end`: death ends the episode, as in the Crafter benchmark. `respawn`: the player
    comes back with full vitals in the same world and the same trace, its achievements
    still unlocked; the episode runs until the world's length, `max_episode_steps` or a
    turn/token limit of the player."""
    respawn_at: Literal["spawn", "last_sleep", "death_site"] = "spawn"
    """Where a respawn puts the player (the nearest free walkable cell): the world's
    spawn point, where it last fell asleep (Minecraft's bed; the spawn point until it
    has slept) or where it died."""
    death_inventory: Literal["drop_all", "drop_materials", "keep"] = "drop_all"
    """What a death takes: every item (Minecraft, minus the pickup: Crafter has no item
    drops), the materials but not the tools, or nothing."""
    death_penalty: float = Field(0.0, ge=0.0)
    """Subtracted from the episode reward per death."""


_VITALS = ("health", "food", "drink", "energy")
_MATERIALS = ("sapling", "wood", "stone", "coal", "iron", "diamond")


def _free_cell_near(world: Any, target: tuple[int, int], radius: int = 16) -> tuple[int, int] | None:
    """The walkable, unoccupied cell nearest `target` (rings of growing radius)."""
    x0, y0 = target
    for r in range(radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                material, obj = world[(x0 + dx, y0 + dy)]
                if material in constants.walkable and obj is None:
                    return (x0 + dx, y0 + dy)
    return None


def _respawn(env: crafter.Env, at: tuple[int, int], death_inventory: str) -> tuple[int, int]:
    """Bring the dead player back: full vitals, the survival counters cleared, the
    inventory rule applied, moved to the free cell nearest `at`. Achievements stay."""
    player = env._player
    for item in _VITALS:
        player.inventory[item] = constants.items[item]["max"]
    for item in player.inventory:
        if item not in _VITALS and (death_inventory == "drop_all" or (death_inventory == "drop_materials" and item in _MATERIALS)):
            player.inventory[item] = 0
    player._hunger = player._thirst = player._fatigue = player._recover = 0
    player.sleeping = False
    if tuple(int(x) for x in player.pos) != at:
        cell = _free_cell_near(env._world, at)
        if cell is not None:
            env._world.move(player, cell)
    env._last_health = player.health  # the refill is no health reward
    return tuple(int(x) for x in player.pos)


# The ledger: what the env writes into a workspace for the model, one line per event.
_SIGHTED = {1: "water", 3: "stone", 6: "tree", 7: "lava", 8: "coal", 9: "iron", 10: "diamond", 11: "table", 12: "furnace", 14: "cow"}
_PLACED = {"place_table": "table", "place_furnace": "furnace", "place_stone": "stone", "place_plant": "sapling"}


def _ledger_sightings(env: crafter.Env, step: int, seen: set, radius: int) -> list[str]:
    """First sightings in view, one per kind and 5x5 block (the cell nearest the player)."""
    semantic = _player_info(env).get("semantic")
    if semantic is None:
        return []
    px, py = (int(v) for v in env._player.pos)
    width, height = semantic.shape
    nearest: dict[tuple[str, int, int], tuple[int, int, int]] = {}
    for x in range(max(0, px - radius), min(width, px + radius + 1)):
        for y in range(max(0, py - radius), min(height, py + radius + 1)):
            kind = _SIGHTED.get(int(semantic[x, y]))
            if kind is None or (key := (kind, x // 5, y // 5)) in seen:
                continue
            d = max(abs(x - px), abs(y - py))
            if key not in nearest or d < nearest[key][0]:
                nearest[key] = (d, x, y)
    seen.update(nearest)
    return [f"step {step}: saw {kind} at ({x}, {y})" for (kind, _, _), (_, x, y) in sorted(nearest.items(), key=lambda kv: kv[1])]


def _ledger_events(env: crafter.Env, step: int, before: dict[str, int], after: dict[str, int]) -> list[str]:
    """What the last step achieved (collected, made, placed, ate, defeated, woke up)."""
    px, py = (int(v) for v in env._player.pos)
    lines = []
    for name in sorted(after):
        if int(after[name]) <= int(before.get(name, 0)):
            continue
        first = " (first time)" if int(before.get(name, 0)) == 0 else ""
        if name in _PLACED:
            tx, ty = (int(v) for v in env._player.pos + env._player.facing)
            lines.append(f"step {step}: placed {_PLACED[name]} at ({tx}, {ty}){first}")
        else:
            lines.append(f"step {step} at ({px}, {py}): {name.replace('_', ' ')}{first}")
    return lines


def _ledger_death(env: crafter.Env, step: int, current: dict[str, Any]) -> str:
    inventory = current.get("inventory") or {}
    px, py = (int(v) for v in env._player.pos)
    semantic = _player_info(env).get("semantic")
    zombies = 0
    if semantic is not None:
        width, height = semantic.shape
        zombies = sum(
            int(semantic[x, y]) == 15
            for x in range(max(0, px - 1), min(width, px + 2))
            for y in range(max(0, py - 1), min(height, py + 2))
        )
    vitals = ", ".join(f"{k} {int(inventory.get(k, 0))}" for k in ("food", "drink", "energy"))
    return f"step {step} at ({px}, {py}): died ({vitals}; zombies adjacent: {zombies})"


def _initial_observation(seed: int, max_steps: int, view_radius: int, tools: str | None = None) -> str:
    env = crafter.Env(seed=seed, length=max_steps)
    env.reset()
    return _format_observation(
        env=env, info=_player_info(env), step=0, max_steps=max_steps, view_radius=view_radius, tools=tools
    )


def _step(env: crafter.Env, action: str) -> tuple[float, bool, dict[str, Any]]:
    _, reward, done, info = env.step(ACTION_TO_INDEX[action])
    return float(reward), bool(done), info


def _step_info(env: crafter.Env, info: Any, reward: float) -> dict[str, Any]:
    """The player's state after a step: the step's public info where it has it, as
    plain ints (Crafter returns numpy values, e.g. `player_pos` is an array)."""
    current = _player_info(env, reward)
    if isinstance(info, dict):
        for key in ("inventory", "achievements", "player_pos"):
            if key in info:
                current[key] = info[key]
    current["inventory"] = {str(k): int(v) for k, v in (current.get("inventory") or {}).items()}
    current["achievements"] = {str(k): int(v) for k, v in (current.get("achievements") or {}).items()}
    pos = current.get("player_pos")
    current["player_pos"] = [int(x) for x in (pos if pos is not None else (0, 0))]
    return current


class CrafterEnv(vf.Env[CrafterEnvConfig]):
    async def run(self, task, agents):
        seed = int(task.data.info["seed"])
        max_steps = self.config.max_episode_steps
        started = time.time()
        env = crafter.Env(seed=seed, length=max_steps)
        await asyncio.to_thread(env.reset)
        book = nb.Notebook(**task.data.info["notebook"]) if task.data.info.get("notebook") else None
        space = ws.Workspace(**task.data.info["workspace"]) if task.data.info.get("workspace") else None
        tools = "notebook" if book is not None else "workspace" if space is not None else None
        seen: set[tuple[str, int, int]] = set()  # ledger sightings: (kind, 5x5 block)
        if space is not None:
            for line in _ledger_sightings(env, 0, seen, self.config.view_radius):
                space.log(line)
        opening = None
        if task.data.prompt is None:
            opening = _format_observation(
                env=env,
                info=_player_info(env),
                step=0,
                max_steps=max_steps,
                view_radius=self.config.view_radius,
                tools=tools,
            )
            if space is not None:
                opening = f"{space.index()}\n\n{opening}"
            if book is not None:
                opening = f"{book.render()}\n\n{opening}"
        total_reward, invalid, steps, done = 0.0, 0, 0, False
        actions: Counter[str] = Counter()
        unlocked_at: dict[str, int] = {}  # achievement -> game step it was first unlocked
        end = "terminated"  # the run stopped answering (a turn or token limit)
        spawn = tuple(int(x) for x in env._player.pos)
        last_sleep = spawn
        deaths: list[dict[str, Any]] = []  # one entry per death: step, vitals, where it came back
        respawned: str | None = None  # told to the model in the next observation
        raw_action: str | None = None  # the last action the model wrote, as in v0's parse_answer
        current = _player_info(env)
        try:
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn(opening)
                while not segment.terminated:
                    steps += 1
                    if steps >= max_steps:
                        end = "max_episode_steps"
                        break  # as in v0: the capped turn's action is not played
                    notes = book.apply(segment.last_reply) if book is not None else []
                    results = space.apply(segment.last_reply, steps) if space is not None else []
                    before = dict(current["achievements"])
                    found = _ACTION.findall(segment.last_reply)
                    if found:
                        raw_action = found[0]
                    action = _normalize_action(raw_action)
                    invalid_action = None
                    if action is None:
                        invalid_action, action = raw_action or "", "noop"
                        invalid += 1
                    try:
                        reward, done, info = await asyncio.to_thread(_step, env, action)
                    except Exception as exc:  # a broken step must not kill the rollout
                        logger.warning("Crafter step error: %s", exc)
                        reward, done, info = 0.0, False, {}
                        invalid_action = raw_action or action
                    total_reward += reward
                    actions[action] += 1
                    current = _step_info(env, info, reward)
                    if space is not None:
                        for line in _ledger_events(env, steps, before, current["achievements"]):
                            space.log(line)
                        for line in _ledger_sightings(env, steps, seen, self.config.view_radius):
                            space.log(line)
                    for name, count in current["achievements"].items():
                        if count > 0 and name not in unlocked_at:
                            unlocked_at[name] = steps
                    if env._player.sleeping:
                        last_sleep = tuple(int(x) for x in env._player.pos)
                    died = current["inventory"].get("health", 0) <= 0
                    world_over = env._step >= env._length
                    if died and space is not None:
                        space.log(_ledger_death(env, steps, current))
                    if died and not world_over and self.config.on_death == "respawn":
                        vitals = {k: int(current["inventory"].get(k, 0)) for k in _VITALS}
                        at = {"spawn": spawn, "last_sleep": last_sleep}.get(
                            self.config.respawn_at, tuple(int(x) for x in env._player.pos)
                        )
                        pos = _respawn(env, at, self.config.death_inventory)
                        if space is not None:
                            space.log(f"step {steps}: respawned at {pos} ({self.config.death_inventory.replace('_', ' ')})")
                        deaths.append({"step": steps, "vitals": vitals, "pos": list(current["player_pos"]), "respawn": list(pos)})
                        current = _step_info(env, {}, 0.0)
                        lost = {"drop_all": "lost your inventory", "drop_materials": "lost your materials (tools kept)", "keep": "kept your inventory"}
                        respawned = (
                            f"You died (death {len(deaths)}, at {tuple(deaths[-1]['pos'])}). You respawned at {pos} "
                            f"with full health, food, drink and energy, and {lost[self.config.death_inventory]}. "
                            "Unlocked achievements stay unlocked."
                        )
                        done = False
                    elif done:
                        end = "died" if died else "world_length"
                        break
                    observation = _format_observation(
                        env=env,
                        info=current,
                        step=steps,
                        max_steps=max_steps,
                        view_radius=self.config.view_radius,
                        last_action=action,
                        invalid_action=invalid_action,
                        tools=tools,
                    )
                    if respawned is not None:
                        observation, respawned = f"{respawned}\n\n{observation}", None
                    if book is not None:
                        observation = "\n".join([book.render(), *notes, "", observation])
                    if space is not None:
                        observation = "\n".join([space.index(), *results, "", observation])
                    segment = await interaction.turn(observation)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        trace = interaction.trace
        unlocked = sorted(name for name, count in (current.get("achievements") or {}).items() if int(count) > 0)
        reward = len(unlocked) / max(len(ACHIEVEMENTS), 1) - self.config.death_penalty * len(deaths)
        trace.record_reward("achievements", reward, 1.0)
        survived = 1.0 if int((current.get("inventory") or {}).get("health", 0)) > 0 else 0.0
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(unlocked)),
                "total_env_reward": total_reward,
                "invalid_actions": float(invalid),
                "survived": survived,
                "deaths": float(len(deaths)),
                **({"memory_writes": float(book.writes), "scratchpad_updates": float(book.scratchpad_updates)} if book else {}),
                **(
                    {f"workspace_{k}": float(space.counts.get(k, 0)) for k in ("search", "read", "write", "append")}
                    if space
                    else {}
                ),
            }
        )
        trace.info["crafter"] = {
            "achievements": unlocked,
            "unlocked_at": unlocked_at,
            "steps": steps,
            "done": done,
            "ended_by": end,
            "seed": seed,
            "deaths": deaths,
            "notebook": book.summary() if book is not None else None,
            "workspace": space.summary() if space is not None else None,
        }
        if self.config.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "reward": reward,
                "achievements": len(unlocked),
                "invalid_actions": invalid,
                "steps": steps,
                "done": done,
                "survived": survived,
                "ended_by": end,
                "deaths": deaths,
                "notebook": book.summary() if book is not None else None,
                "workspace": space.summary() if space is not None else None,
                "seed": seed,
                "unlocked_at": unlocked_at,
                "actions": dict(actions),
                "inventory": current.get("inventory") or {},
                "trace_tokens": trace.num_total_tokens,
                "task": "crafter",
            }
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class CrafterTaskset(vf.Taskset[CrafterTask, CrafterConfig]):
    def load(self) -> Iterator[CrafterTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = REASONING_SYSTEM_PROMPT if cfg.enable_reasoning else SYSTEM_PROMPT
        if cfg.notebook and cfg.workspace:
            raise ValueError("notebook and workspace are alternative memories: enable one")
        if cfg.workspace and not (cfg.workspace_files or cfg.workspace_ledger):
            raise ValueError("a workspace needs files, a ledger or both")
        notebook = workspace = None
        tools = None
        if cfg.notebook:
            notebook = {"slots": cfg.notebook_slots, "memory_chars": cfg.memory_chars, "scratchpad_chars": cfg.scratchpad_chars}
            system, tools = f"{_admit_tools(system, 'notebook')}\n\n{nb.instructions(**notebook)}", "notebook"
        if cfg.workspace:
            workspace = {
                "files_enabled": cfg.workspace_files,
                "ledger_enabled": cfg.workspace_ledger,
                "file_chars": cfg.workspace_file_chars,
            }
            guide = ws.instructions(cfg.workspace_files, cfg.workspace_ledger, cfg.workspace_file_chars)
            system, tools = f"{_admit_tools(system, 'workspace')}\n\n{guide}", "workspace"
        for i in range(cfg.num_examples):
            seed = base + i
            if cfg.include_initial_obs_in_system:
                # The system prompt carries the opening observation, so this renders a
                # world per task up front (about a second each).
                observation = _initial_observation(seed, cfg.max_episode_steps, cfg.view_radius, tools)
                system_prompt, prompt = f"{system}\n\nINITIAL OBSERVATION:\n{observation}", "What action do you take?"
            else:
                # Prompt-less: the env opens the episode with the observation of the
                # world it builds (as v0's setup_state did), so loading stays instant.
                system_prompt, prompt = system, None
            yield CrafterTask(
                CrafterData(
                    name=f"crafter#{cfg.split}{i}",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    info={
                        "seed": seed,
                        "episode_idx": i,
                        "split": cfg.split,
                        "notebook": notebook,
                        "workspace": workspace,
                    },
                ),
                cfg.task,
            )

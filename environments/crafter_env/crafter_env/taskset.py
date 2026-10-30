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
import dataclasses
import gc
import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import crafter
import verifiers.v1 as vf
from crafter import constants
from pydantic import Field, field_validator, model_validator

from crafter_env import notebook as nb
from crafter_env import workspace as ws
from crafter_env.judge import (
    CrafterGroupJudge,
    CrafterJudge,
    CrafterJudgeConfig,
    GroupRendezvous,
    apply_verdict,
    group_verdicts,
    judge_segment,
    judge_segment_group,
    scale_cost,
    usage_cost,
)
from crafter_env.state_store import StateStore, open_store, retry_io

logger = logging.getLogger(__name__)

ACTION_NAMES: tuple[str, ...] = tuple(constants.actions)
ACTION_TO_INDEX: dict[str, int] = {
    action: idx for idx, action in enumerate(ACTION_NAMES)
}
ACHIEVEMENTS: tuple[str, ...] = tuple(constants.achievements)
VITAL_ITEMS: tuple[str, ...] = ("health", "food", "drink", "energy")

# The 22 achievements by the names the observation uses, in BALROG's order (its Crafter
# instruction prompt lists them numbered, roughly along the tech tree). Opt-in via
# CrafterConfig.list_achievements: the base prompt only says "unlock as many as possible".
ACHIEVEMENT_ORDER: tuple[str, ...] = (
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie", "collect_stone",
    "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword",
    "wake_up", "place_furnace", "collect_coal", "collect_iron", "make_iron_pickaxe",
    "make_iron_sword", "collect_diamond",
)
assert set(ACHIEVEMENT_ORDER) == set(ACHIEVEMENTS) and len(ACHIEVEMENT_ORDER) == len(ACHIEVEMENTS)
ACHIEVEMENT_LIST = "The 22 achievements, by name, roughly in tech-tree order:\n" + "\n".join(
    f"{i}. {name}" for i, name in enumerate(ACHIEVEMENT_ORDER, 1)
)
_GOAL_PARAGRAPH_END = "and avoid dying.\n"


def with_achievement_list(system: str) -> str:
    """The system prompt with the numbered achievement list after the goal paragraph."""
    if ACHIEVEMENT_LIST in system:
        return system
    assert _GOAL_PARAGRAPH_END in system, "the system prompt's goal paragraph is missing"
    return system.replace(_GOAL_PARAGRAPH_END, f"{_GOAL_PARAGRAPH_END}\n{ACHIEVEMENT_LIST}\n", 1)

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


# The reply format of the memory arms without their tools (an ablation control): the
# memory arms drop the base prompt's "exactly one action" / "reply only as" wording.
_RELAXED_REPLY_FORMAT = {
    "Reply with exactly one action in XML:": "Reply with one action in XML:",
    "Output the XML action.": "Output the XML action.",
}
_RELAXED_CLOSING = "Choose the next action as <action>action_name</action>."


def _closing(tools: str | None, reasoning: bool, relaxed: bool = False) -> str:
    """The observation's last line: what to reply with, consistent with the system prompt."""
    if relaxed and tools is None and not reasoning:
        return _RELAXED_CLOSING
    if not reasoning:
        return _CLOSINGS[tools]
    line = "Reason in 2-4 sentences, then give the action on the last line as <action>action_name</action>."
    if tools == "notebook":
        line += " Write to your notebook if you learned something worth keeping."
    elif tools == "workspace":
        line += " Use your workspace tools if they help."
    return line


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


# Observation extras (CrafterEnvConfig.observation_extras): facts the 9x9 map shows
# only implicitly - what is around and where, what the player can craft or place
# right now, and that it is being hurt.
_CREATURES = {15: "zombie", 16: "skeleton", 14: "cow"}
_RESOURCES = {6: "tree", 1: "water", 3: "stone", 8: "coal", 9: "iron", 10: "diamond", 7: "lava", 11: "table", 12: "furnace"}


def _offset(dx: int, dy: int) -> str:
    parts = ([f"{abs(dx)}{'E' if dx > 0 else 'W'}"] if dx else []) + ([f"{abs(dy)}{'S' if dy > 0 else 'N'}"] if dy else [])
    return " ".join(parts) or "here"


def _nearby(env: crafter.Env, radius: int) -> tuple[str, str]:
    """(creatures in view, nearest of each resource), with offsets from the player."""
    semantic = _player_info(env).get("semantic")
    if semantic is None:
        return "unknown", "unknown"
    px, py = (int(v) for v in env._player.pos)
    width, height = semantic.shape
    creatures, nearest = [], {}
    for x in range(max(0, px - radius), min(width, px + radius + 1)):
        for y in range(max(0, py - radius), min(height, py + radius + 1)):
            sid, d = int(semantic[x, y]), max(abs(x - px), abs(y - py))
            if sid in _CREATURES:
                creatures.append((d, _CREATURES[sid], x - px, y - py))
            elif sid in _RESOURCES and (sid not in nearest or d < nearest[sid][0]):
                nearest[sid] = (d, x - px, y - py)
    creatures_text = ", ".join(f"{name} {_offset(dx, dy)}" for _, name, dx, dy in sorted(creatures)[:5]) or "none"
    order = sorted(nearest.items(), key=lambda kv: kv[1][0])
    resources_text = ", ".join(f"{_RESOURCES[sid]} {_offset(dx, dy)}" for sid, (_, dx, dy) in order) or "none"
    return creatures_text, resources_text


def _available(env: crafter.Env) -> list[str]:
    """The place/make actions whose requirements are met right now."""
    player, world = env._player, env._world
    out = []
    target = tuple(int(v) for v in player.pos + player.facing)
    material, obj = world[target]
    for name, info in constants.place.items():
        if obj is None and material in info["where"] and all(player.inventory[k] >= v for k, v in info["uses"].items()):
            out.append(f"place_{name}")
    nearby, _ = world.nearby(player.pos, 1)
    for name, info in constants.make.items():
        if all(u in nearby for u in info["nearby"]) and all(player.inventory[k] >= v for k, v in info["uses"].items()):
            out.append(f"make_{name}")
    return out


_TIER_ACTIONS = (
    ("collect_wood", "collect wood: `do` facing a tree"),
    ("place_table", "place_table"),
    ("make_wood_pickaxe", "make_wood_pickaxe"),
    ("collect_stone", "collect stone: `do` facing stone"),
    ("make_stone_pickaxe", "make_stone_pickaxe"),
    ("place_furnace", "place_furnace"),
    ("collect_coal", "collect coal: `do` facing coal"),
    ("collect_iron", "collect iron: `do` facing iron"),
    ("make_iron_pickaxe", "make_iron_pickaxe"),
    ("collect_diamond", "collect the diamond: `do` facing a diamond"),
)


def _next_tier(env: crafter.Env) -> str:
    """The first step of the tech tree not yet unlocked and what it still needs, from
    Crafter's own recipes: a deterministic yardstick the judge can hold a plan against."""
    player = env._player
    unlocked = {k for k, v in player.achievements.items() if v > 0}
    for name, how in _TIER_ACTIONS:
        if name in unlocked:
            continue
        needs: list[str] = []
        kind, _, item = name.partition("_")
        if kind == "place":
            uses = constants.place[item]["uses"]
        elif kind == "make":
            info = constants.make[item]
            uses = info["uses"]
            needs += [f"near a {u}" for u in info["nearby"]]
        else:
            uses = (constants.collect.get(item) or {}).get("require") or {}
        for k, v in uses.items():
            have = int(player.inventory.get(k, 0))
            if have < v:
                needs.append(f"{v} {k} (has {have})")
        return f"{name} ({how})" + (": needs " + ", ".join(needs) if needs else ": requirements met right now")
    return "every tier is unlocked"


def _ahead_requirement(env: crafter.Env) -> str:
    """What collecting the tile ahead needs and the player lacks, if anything."""
    player = env._player
    material, _ = env._world[tuple(int(v) for v in player.pos + player.facing)]
    need = (constants.collect.get(material) or {}).get("require") or {}
    missing = [k for k, v in need.items() if player.inventory[k] < v]
    return f" (collecting it needs {', '.join(missing)})" if missing else ""


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
    extras: bool = False,
    affordances: bool = True,
    health_lost: int = 0,
    repeated_action: str | None = None,
    reasoning: bool = False,
    relaxed: bool = False,
    segment: str | None = None,
    closing: str | None = None,
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
        *([segment] if segment else []),
        f"Position: ({int(pos[0])}, {int(pos[1])}), facing {facing}",
        f"Ahead: {_describe_ahead(env)}" + (_ahead_requirement(env) if extras else ""),
        f"Vitals: {vitals}",
        f"Inventory: {resources}",
        f"Unlocked achievements ({len(unlocked)}/{len(ACHIEVEMENTS)}): {unlocked_text}",
        f"Last reward: {float(info.get('reward', 0.0)):.3f}",
    ]
    if extras:
        creatures, resources = _nearby(env, view_radius)
        lines += [f"Creatures in view: {creatures}", f"Nearest: {resources}"]
        if affordances:
            available = _available(env)
            lines.append(f"Can place/make now: {', '.join(available) if available else 'nothing'}")
        if health_lost > 0:
            lines.append(f"You lost {health_lost} health since your last action.")
    if last_action is not None:
        lines.append(f"Last action: {last_action}")
    if repeated_action is not None:
        lines.append(
            f"Your last reply had no <action> tag, so your previous action ({repeated_action}) was "
            "repeated. Reply with <action>action_name</action>."
        )
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
            closing or _closing(tools, reasoning, relaxed),
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
    list_achievements: bool = False
    """List the 22 achievements by name in the system prompt, as BALROG's Crafter
    instruction prompt does (off: the base prompt names none of them)."""
    """Use the prompt that asks for 2-4 sentences of reasoning before the action."""
    notebook: bool = False
    """Give the model notebook tools (memories + scratchpad, crafter_env.notebook),
    explained in the system prompt and shown at the top of every observation."""
    relaxed_reply_format: bool = False
    """Ablation control for the memory arms: their reply format (no "exactly one action" /
    "reply only as" wording) without their tools. No effect with a notebook, a
    workspace or reasoning, which set their own reply format."""
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
    on_death: Literal["end", "respawn", "revert"] = "end"
    """`end`: death ends the episode, as in the Crafter benchmark. `respawn`: the player
    comes back with full vitals in the same world and the same trace, its achievements
    still unlocked; the episode runs until the world's length, `max_episode_steps` or a
    turn/token limit of the player. `revert`: the whole world (player, inventory,
    achievements, creatures, time of day) is restored to a save point (`revert_to`) in the
    same trace, like reloading a saved game: what the player did since is undone, the turns
    spent are not given back, and the context still holds them."""
    revert_to: Literal["segment_start", "last_base"] = "segment_start"
    """With on_death = revert: the save point. `segment_start`: the world as this segment
    began (the parent's saved state, or the fresh seed). `last_base`: refreshed whenever
    the player falls asleep or places a table (a bed or a workbench is the save), the
    segment start until then."""
    respawn_at: Literal["spawn", "last_sleep", "last_base", "death_site"] = "spawn"
    """Where a respawn puts the player (the nearest free walkable cell): the world's
    spawn point, where it last fell asleep (Minecraft's bed; the spawn point until it
    has slept) or where it died. New strategy, opt-in: `last_base`, the most recent of
    where it last slept and where it last placed a table (a policy that never sleeps
    still keeps its workbench)."""
    respawn_safe: bool = False
    """New strategy, opt-in: prefer the nearest free cell with no zombie or skeleton
    within `respawn_safe_radius` tiles (else the nearest free cell). Off: the nearest free
    cell, as before."""
    respawn_safe_radius: int = Field(2, ge=1)
    """With respawn_safe: how far (in tiles, a square around the cell) no hostile may be. A
    zombie chases from Manhattan distance 8, so 8 puts the player out of every chase."""
    death_inventory: Literal["drop_all", "drop_materials", "keep"] = "drop_all"
    """What a death takes: every item (Minecraft, minus the pickup: Crafter has no item
    drops), the materials but not the tools, or nothing."""
    announce_malformed: bool = True
    """Tell the model when a reply without an action tag made its last action repeat
    (off: the repeat is silent, as in v0 - for comparisons with runs made before)."""
    observation_extras: bool = False
    """Add to each observation the creatures in view and the nearest of each resource
    (with offsets), the place/make actions available right now, what collecting the
    tile ahead still needs, and how much health the last step cost."""
    extras_affordances: bool = True
    """With observation_extras: include the "Can place/make now" line (off: perception
    only - creatures, nearest resources, the tile ahead's requirement, damage)."""
    death_penalty: float = Field(0.0, ge=0.0)
    """Subtracted from the reward per death, as its own `deaths` term (weight 1, next to
    the achievement delta and the judge, so it stays with achievement_weight = 0)."""
    announce_death_cause: bool = False
    """With respawn: the death line names the cause (zombie, skeleton, lava, starved),
    says "at night" when it was dark, and gives the game step, as the revert line always
    does. Off: the line only says where the player died and came back (as before)."""
    death_lesson: bool = False
    """The observation after a death (respawn or revert) asks, once, for a lesson before
    the action, in the same reply: one or two sentences inside <lesson>...</lesson> on
    what killed the player and what to do differently (the <plan>-before-action format of
    Paglieri et al. 2509.03581). The segment's lessons are then shown at the top of every
    later observation of the segment (the turn window cannot evict them), and the system
    prompt declares the protocol (LESSON_PROTOCOL). Off: no request, nothing shown."""
    lesson_max_chars: int = Field(300, ge=1)
    """With death_lesson: a lesson is cut to this many characters when it is pinned."""
    lessons_kept: int = Field(6, ge=1)
    """With death_lesson: how many of the latest lessons are pinned."""
    train_on_notes: bool = True
    """False: the notes turn (the summary reply and its retry) and every earlier reply that
    holds <notes> stay in the context but are masked out of the loss (their nodes' mask set
    False), so the gradient sees the action replies only. A mid-segment <notes> reply loses
    its action too. True: everything the model sampled is trained on, as upstream."""
    recall_probe: bool = False
    """At the cut, before the notes request, a MEMORY CHECK turn (RECALL_PROBE, declared in
    the system prompt by RECALL_PROTOCOL) asks for the segment's deaths (count, game step,
    cause) and the game step of each achievement unlocked, in a fixed line form. The reply
    is scored against the segment's own record (`score_recall_probe`: `recall` in the record
    and the episode log, `recall_*` metrics) and is never trained on (the same loss-mask
    request as train_on_notes = False). An eval knob: what the policy can still recall of
    what the turn window evicted (2026-10-28)."""
    cut_on_death: bool = False
    """With segments and respawn: a death ends the segment right after the respawn
    (summary, saved state, judge), so the next segment starts from the respawn point
    with a fresh context instead of carrying the death inside a long segment."""
    # Segments: the game continues across episodes (docs/crafter-kv-streams-summary-segments.md).
    # With either cut set, one episode is one segment: it restores the world the task
    # points at (`task.data.info["node"]`), plays to the cut, asks the policy for a
    # summary in the same session, saves the world to `state_dir` and leaves the
    # summary and the state path in `trace.info["crafter"]`. `max_episode_steps` is
    # then the world's own length (Crafter's 10,000), not a cut.
    segment_turns: int | None = Field(None, ge=1)
    """Model turns per segment; the reply that crosses it is not played."""
    segment_tokens: int | None = Field(None, ge=1)
    """Logical positions per segment (`trace.num_total_tokens`, evicted tokens
    included): the cut leaves room for the summary under the model's limit."""
    summary_max_tokens: int | None = Field(32768, ge=1)
    """Output cap for the summary turn only (the per-turn sampling cap stays)."""
    summary_prompt: str | None = None
    """Overrides the default request for the summary."""
    notes_protocol: bool = True
    """Declare the notes turn in the system prompt (NOTES_PROTOCOL), send the last
    observation without its "reply with an action" line, ask for <notes>...</notes>, and
    warn the policy a few turns before the cut."""
    notes_warning_turns: int = Field(3, ge=0)
    """With notes_protocol: say in the observation that the segment ends in N turns."""
    summary_retry: bool = True
    """Ask once more in the same session (SUMMARY_RETRY) when the notes reply holds no
    notes of 80+ characters (inside <notes> with notes_protocol; a trailing action is
    ignored): `notes_problem`, the reason logged as `summary_retry_reason`. Also
    (without notes_protocol) drops the observation's "reply with an action" closing line
    from the summary request. (After 400 action turns the policy answered the request
    with an action in 17% of segments, job 11393114.)"""
    state_dir: str | None = None
    """Where saved worlds go (`<state_dir>/<trace_id>.pkl`); unset keeps them in
    memory, which only works when one process plays the whole tree (`tree_depth`)."""
    final_goal: str = "collect_diamond"
    """The achievement that ends a lineage; the segment that unlocks it writes no summary."""
    achievement_weight: float = Field(1.0, ge=0.0)
    """Weight of the segment's achievement delta / 22 in the reward (0 for the judge alone)."""
    judge: CrafterJudgeConfig | None = None
    """The segment judge's backend, endpoint and model (OpenAI-compatible backends
    read `api_key_var`; CAPI uses COPILOT_GITHUB_TOKEN or the active saved CLI login);
    unset: no judge, the achievement delta is the reward."""
    judge_weight: float = Field(1.0, ge=0.0)
    judge_concurrency: int = Field(8, ge=1)
    """Concurrent judge calls per env worker."""
    judge_mode: Literal["single", "group"] = "single"
    """`single`: each segment is judged alone as soon as it ends. `group`: the branches
    of one node are judged together in one comparative call, once the last of them ends
    (RL pipeline: they meet in `<state_dir>/judge_groups`; the eval beam: in process)."""
    judge_group_size: int = Field(4, ge=2)
    """Branches per node in group mode (the orchestrator's `group_size`; the eval beam uses `branches`)."""
    judge_group_timeout_s: float = Field(1800.0, gt=0)
    """How long a finished branch waits for its siblings before it is judged alone."""
    judge_price: tuple[float, float, float] | None = None
    """USD per 1M tokens (input, output incl. thinking, cached input) for the cost metrics;
    unset: crafter_env.judge.PRICES for the judge's model."""
    judge_mix: tuple[float, float, float] = (0.5, 0.3, 0.2)
    """Weights of the judge's progress, plan and consistency sub-scores in its score."""
    judge_tier_weight: float = Field(0.0, ge=0.0, le=1.0)
    """> 0: the judge's score becomes (1 - w) x its mixed sub-scores + w x the lineage's
    cumulative tech tier (the deepest ladder step unlocked so far / 8: wood 1, table 2,
    wood tools 3, stone and coal 4, stone tools and furnace 5, iron 6, iron tools 7,
    diamond 8), so the beam keeps advanced lineages that plateau for a segment. The
    judge's own score is logged as `llm_score`, the tier as `tier`."""
    judge_world_radius: int = Field(24, ge=0)
    """The judge is told the nearest resources within this many tiles of the player,
    from the whole map (the policy sees `view_radius`): privileged, judge-only."""
    tree_depth: int = Field(0, ge=0)
    """The eval driver (arm A): > 0 plays a beam of `branches` per node, `keep` per
    depth, to this depth inside one run, with the judge after each branch; 0 plays one
    segment per episode and leaves the tree to the orchestrator's tree sampler."""
    branches: int = Field(4, ge=1)
    keep: int = Field(1, ge=1)
    keep_per_parent: int | None = Field(None, ge=1)
    """Exploration, the sibling cap: at most this many of a node's branches are kept per
    level, so several lineages stay alive (2 with keep 4 = at least two parents); unset:
    the `keep` best regardless of parent. See `select_children` for the top-up rule."""
    root_branches: int | None = Field(None, ge=1)
    """Branches of the root (the only node of depth 1); unset: `branches x keep`, so every
    level plays the same number of segments. With `state_dir` the beam checkpoints its
    kept nodes to `<state_dir>/beam/<seed>.json` after each level and a later run of the
    same seed resumes from there (a 5-day job chain can go thousands of levels deep)."""

    @field_validator("judge", mode="before")
    @classmethod
    def _judge_config(cls, value):
        if isinstance(value, vf.JudgeConfig) and not isinstance(value, CrafterJudgeConfig):
            return value.model_dump()
        return value

    @model_validator(mode="after")
    def _check_segments(self):
        if self.tree_depth > 0 and self.segment_turns is None and self.segment_tokens is None:
            raise ValueError("tree_depth needs segments: set segment_turns and/or segment_tokens")
        return self


_VITALS = ("health", "food", "drink", "energy")
_MATERIALS = ("sapling", "wood", "stone", "coal", "iron", "diamond")


_HOSTILE = ("Zombie", "Skeleton")


def _hostile_near(world: Any, pos: tuple[int, int], distance: int = 2) -> bool:
    """A zombie or skeleton in the square of half-width `distance` around `pos` (the area
    of `world.nearby`, which near the map's edge slices a negative start to nothing)."""
    x, y = pos
    return any(
        type(o).__name__ in _HOSTILE and max(abs(int(o.pos[0]) - x), abs(int(o.pos[1]) - y)) <= distance
        for o in world.objects
    )


def _free_cell_near(
    world: Any, target: tuple[int, int], radius: int = 16, safe: bool = False, safe_radius: int = 2
) -> tuple[int, int] | None:
    """The walkable, unoccupied cell nearest `target` (rings of growing radius); with
    `safe`, first one with no zombie or skeleton within `safe_radius` tiles, else any."""
    x0, y0 = target
    fallback = None
    for r in range(radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cell = (x0 + dx, y0 + dy)
                material, obj = world[cell]
                if material in constants.walkable and obj is None:
                    if not safe or not _hostile_near(world, cell, safe_radius):
                        return cell
                    fallback = fallback or cell
    return fallback


DEATH_CAUSES = ("zombie", "skeleton", "lava", "starved", "unknown")


def _death_cause(env: crafter.Env, vitals: dict[str, int]) -> str:
    """What most likely took the last health, read off the world right after the step
    (crafter/objects.py): lava under the player; a zombie at Manhattan distance 1 (where it
    hits); a skeleton within bow range (it shoots at 5, the arrow is gone after a hit) or an
    arrow next to the player; else a food/drink/energy at 0 (the slow health drain); else unknown."""
    player = env._player
    if env._world[tuple(int(v) for v in player.pos)][0] == "lava":
        return "lava"
    near = [(type(o).__name__, int(player.distance(o))) for o in env._world.objects if o is not player]
    if any(name == "Zombie" and d <= 1 for name, d in near):
        return "zombie"
    if any((name == "Skeleton" and d <= 6) or (name == "Arrow" and d <= 2) for name, d in near):
        return "skeleton"
    if any(vitals.get(k, 1) == 0 for k in ("food", "drink", "energy")):
        return "starved"
    return "unknown"


def _death_how(cause: str, starved: list[str], daylight: float) -> str:
    """The cause as told to the model: "killed by a zombie at night", "starved: food at 0"."""
    how = {
        "zombie": "killed by a zombie", "skeleton": "shot by a skeleton", "lava": "fell into lava",
        "starved": "starved: " + ", ".join(starved) + " at 0", "unknown": "cause unknown",
    }[cause]
    return how + (" at night" if daylight < 0.5 else "")


def _respawn(
    env: crafter.Env, at: tuple[int, int], death_inventory: str, safe: bool = False, safe_radius: int = 2
) -> tuple[int, int]:
    """Bring the dead player back: full vitals, the survival counters cleared, the
    inventory rule applied, moved to the free cell nearest `at` (with `safe`, away
    from hostiles if it can, also when it died on `at` itself). Achievements stay."""
    player = env._player
    for item in _VITALS:
        player.inventory[item] = constants.items[item]["max"]
    for item in player.inventory:
        if item not in _VITALS and (death_inventory == "drop_all" or (death_inventory == "drop_materials" and item in _MATERIALS)):
            player.inventory[item] = 0
    player._hunger = player._thirst = player._fatigue = player._recover = 0
    player.sleeping = False
    here = tuple(int(x) for x in player.pos)
    if here != at or (safe and _hostile_near(env._world, here, safe_radius)):
        cell = _free_cell_near(env._world, at, safe=safe, safe_radius=safe_radius)
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


def _initial_observation(
    seed: int, max_steps: int, view_radius: int, tools: str | None = None, reasoning: bool = False
) -> str:
    env = crafter.Env(seed=seed, length=max_steps)
    env.reset()
    return _format_observation(
        env=env, info=_player_info(env), step=0, max_steps=max_steps, view_radius=view_radius, tools=tools,
        reasoning=reasoning,
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


# Segments: the game continues across episodes. A segment is one episode, one trace
# and one SGLang session (so its RoPE positions start at 0): it restores the world a
# previous segment saved, plays until the cut, asks the policy for a summary in the
# same session, saves the world to the state store and leaves the summary and the
# state path in `trace.info["crafter"]` for whoever continues it (the tree sampler
# in the orchestrator, or the in-env beam of `tree_depth > 0`). The next segment's
# protected first message is the system prompt plus that summary.
CONTINUATION_HEADER = (
    "You are continuing a game already in progress. Below are the notes you wrote at the end "
    "of the previous segment of this same game; the world, your inventory and your achievements "
    "are exactly as you left them."
)

# The notes turn is declared up front, in the protected system prompt, so the request
# at the cut is an announced mode switch rather than a late instruction that contradicts
# the 400 turns of "reply with one action" before it.
NOTES_PROTOCOL = """\
This game is played in segments of a few hundred turns. On every normal turn, reply \
with one action as described above. When a message starts with "END OF SEGMENT", the \
game is paused and no action will be played: reply only with your notes for the next \
segment, inside <notes>...</notes>, and no <action> tag. The next segment starts from \
the same world with a fresh context that holds only this system prompt and those notes."""

SUMMARY_PROMPT = """\
END OF SEGMENT. The game is paused and no action will be played. The game continues \
from exactly this state, but in a fresh context that will hold only the system prompt \
and the notes you write now, so write down everything the next segment needs, inside \
<notes>...</notes>. Cover:
1. Where you are and what is around you: known trees, water, stone, coal, iron, diamonds, \
lava, caves and dark areas; where your table and furnace are. Give every place in world \
coordinates (x, y), as in the Position line: x grows to the east, y grows to the south, and \
a map cell k columns right of @ and m rows below it is at (your x + k, your y + m) (left \
and above: minus). Do not use row or column numbers of the map: they change as you move.
2. Your inventory and tools and the achievements unlocked, as in the last observation (if \
you died, as they are after the respawn, not as remembered).
3. What worked and what failed in this segment: dangers, deaths, wasted turns.
4. Your plan: the concrete next steps toward collect_diamond (the next resource, the next \
tool, where to go, how to get through the night).
Be factual and specific. Reply with <notes>...</notes> only, no <action> tag."""

SUMMARY_RETRY = """\
END OF SEGMENT, still: the game is paused and no action will be played now. Write the \
notes for the next segment inside <notes>...</notes>, numbered 1 to 4 as asked above \
(where you are and what is around you in world coordinates, inventory and achievements, \
what worked and failed, the plan). Do not output an <action> tag."""

_NOTES = re.compile(r"<notes>\s*(.*?)\s*(?:</notes>|$)", re.DOTALL)

# The recall probe (recall_probe = True): a paused turn before the notes request that asks
# for facts the record can check, so what the policy recalls of the evicted part of the
# segment is measured directly instead of read off the free-form notes.
RECALL_PROTOCOL = """\
A message that starts with "MEMORY CHECK" is also a paused turn: no action will be played; \
answer its questions from memory, in the line form it gives, and no <action> tag."""

RECALL_PROBE = """\
MEMORY CHECK. The game is paused and no action will be played. Answer from memory about \
this segment only (since the start of this segment, or since your previous notes). Do not \
guess: write unknown for a value you do not remember.
1. How many times did you die in this segment? For each death: at which game step (the \
"Step:" counter of the observation) and from what cause (zombie, skeleton, lava or starved).
2. For each achievement you unlocked in this segment: at which game step did you unlock it?
Reply in exactly this form, one item per line and nothing else:
deaths: <number>
death: step=<game step>, cause=<cause>
unlocked: <achievement_name>, step=<game step>
Write "deaths: 0" and no death line if you did not die. Do not output an <action> tag."""

_PROBE_COUNT = re.compile(r"deaths?\s*:\s*(\d+|unknown)", re.I)
_PROBE_DEATH = re.compile(r"death\s*\d*\s*:\s*step\s*[=:]?\s*(\d+|unknown)\s*,?\s*cause\s*[=:]?\s*([a-z_]+)", re.I)
_PROBE_ACH = re.compile(r"unlocked\s*:\s*([a-z_]+)\s*,?\s*step\s*[=:]?\s*(\d+|unknown)", re.I)
_PROBE_CAUSES = {"zombie": "zombie", "skeleton": "skeleton", "arrow": "skeleton", "lava": "lava", "starv": "starved",
                 "hunger": "starved", "food": "starved", "thirst": "starved", "drink": "starved", "energy": "starved"}


def _probe_cause(word: str) -> str:
    w = word.lower()
    return next((c for k, c in _PROBE_CAUSES.items() if w.startswith(k)), "unknown")


def score_recall_probe(reply: str, deaths: list[dict[str, Any]], unlocked_at: dict[str, int], steps: int,
                       tolerance: int = 10) -> dict[str, Any]:
    """The probe reply against the segment record. Each real death is matched to the closest
    claimed death within `tolerance` game steps (each claim used once); unmatched claims are
    false deaths. A claimed achievement is a false one when the segment did not unlock it, and
    its step is right within `tolerance`. `turns_ago` = segment turns between the death and the
    cut (what the turn window evicts is a function of it)."""
    text = _strip_action(reply or "")
    count_match = _PROBE_COUNT.search(text)
    claimed = [(int(step) if step.isdigit() else None, _probe_cause(cause)) for step, cause in _PROBE_DEATH.findall(text)]
    claimed_ach = [(name.lower(), int(step) if step.isdigit() else None) for name, step in _PROBE_ACH.findall(text)]
    parsed = bool(count_match or claimed or claimed_ach)
    count = int(count_match.group(1)) if count_match and count_match.group(1).isdigit() else (len(claimed) if claimed else None)
    free = list(range(len(claimed)))
    matched = []
    for d in deaths:
        truth = int(d.get("game_step", d["step"]))
        best = None
        for j in free:
            step = claimed[j][0]
            if step is not None and abs(step - truth) <= tolerance and (best is None or abs(step - truth) < abs(claimed[best][0] - truth)):
                best = j
        if best is not None:
            free.remove(best)
        matched.append({"game_step": truth, "cause": d.get("cause"), "turns_ago": int(steps - int(d["step"])),
                        "matched": best is not None, "cause_ok": best is not None and claimed[best][1] == d.get("cause")})
    ach = [{"name": name, "step": step, "known": name in unlocked_at,
            "step_ok": name in unlocked_at and step is not None and abs(step - int(unlocked_at[name])) <= tolerance}
           for name, step in claimed_ach]
    return {
        "reply": text[:1500], "parsed": parsed,
        "deaths_true": len(deaths), "deaths_claimed": count, "count_correct": count == len(deaths),
        "deaths": matched, "false_deaths": len(free),
        "achievements_true": len(unlocked_at), "achievements": ach,
        "achievements_step_ok": sum(a["step_ok"] for a in ach), "false_achievements": sum(not a["known"] for a in ach),
    }


def recall_metrics(recall: dict[str, Any] | None) -> dict[str, float]:
    if not recall:
        return {}
    return {
        "recall_parsed": float(recall["parsed"]),
        "recall_deaths_true": float(recall["deaths_true"]),
        "recall_deaths_claimed": float(recall["deaths_claimed"] or 0),
        "recall_count_correct": float(recall["count_correct"]),
        "recall_deaths_matched": float(sum(d["matched"] for d in recall["deaths"])),
        "recall_deaths_cause_ok": float(sum(d["cause_ok"] for d in recall["deaths"])),
        "recall_false_deaths": float(recall["false_deaths"]),
        "recall_achievements_true": float(recall["achievements_true"]),
        "recall_achievements_step_ok": float(recall["achievements_step_ok"]),
        "recall_false_achievements": float(recall["false_achievements"]),
    }


def with_recall_protocol(system_prompt: str | None) -> str:
    """The system prompt with RECALL_PROTOCOL after the game rules (and the notes protocol), before any notes of a previous segment."""
    sp = system_prompt or ""
    if RECALL_PROTOCOL in sp:
        return sp
    if CONTINUATION_HEADER in sp:
        head, tail = sp.split(CONTINUATION_HEADER, 1)
        return f"{head.rstrip()}\n\n{RECALL_PROTOCOL}\n\n{CONTINUATION_HEADER}{tail}"
    return f"{sp}\n\n{RECALL_PROTOCOL}"


def request_loss_drop(trace: Any, node_indexes: list[int]) -> None:
    """Ask the orchestrator to mask these nodes out of the loss on its copy of the graph.
    The orchestrator assembles its own graph from the model calls as they happen; a node
    crosses the wire once, so an env-side mask edit never reaches it. `info` is re-sent
    whenever it changes: `prime_rl.orchestrator.algo.routing.apply_loss_mask_overrides`
    applies `trace.info["loss_mask"]["drop_nodes"]` before scoring."""
    info = getattr(trace, "info", None)
    if not node_indexes or not isinstance(info, dict):
        return
    request = dict(info.get("loss_mask") or {})
    request["drop_nodes"] = sorted(set(request.get("drop_nodes") or []) | set(node_indexes))
    info["loss_mask"] = request


def drop_sampled_range(trace: Any, first: int, last: int) -> list[int]:
    """Mask the sampled nodes with ordinal in (first, last] (1-based, as `sampled_before_*`
    count them) out of the loss, here and on the orchestrator's copy; returns their indexes."""
    drop, k = [], 0
    for index, node in enumerate(getattr(trace, "nodes", None) or []):
        if not getattr(node, "sampled", False):
            continue
        k += 1
        if first < k <= last:
            node.mask = [False] * len(list(getattr(node, "mask", None) or []))
            drop.append(index)
    request_loss_drop(trace, drop)
    return drop

# The death lesson: asked once, in the observation after a death, as a block before the
# action in the same reply; the segment's lessons are pinned at the top of every later
# observation, so the turn window does not evict them the way it evicts the death itself.
LESSON_PROTOCOL = """\
If you die, the observation after the death asks you, once, for a lesson: in that reply, \
before the action, write one or two sentences inside <lesson>...</lesson> on what killed \
you and what you will do differently. The lessons you write are shown at the top of \
every later observation of the segment."""

LESSON_REQUEST = (
    "First write what you learned from this death inside <lesson>...</lesson>: one or two "
    "sentences on what killed you and what you will do differently."
)

LESSON_HEADER = "Lessons from your deaths in this segment (you wrote these after dying):"

_LESSON = re.compile(r"<lesson>\s*(.*?)\s*(?:</lesson>|<action>|$)", re.DOTALL)


def with_lesson_protocol(system_prompt: str | None) -> str:
    """The system prompt with LESSON_PROTOCOL after the game rules (and the notes protocol),
    before any notes of a previous segment."""
    sp = system_prompt or ""
    if LESSON_PROTOCOL in sp:
        return sp
    if CONTINUATION_HEADER in sp:
        head, tail = sp.split(CONTINUATION_HEADER, 1)
        return f"{head.rstrip()}\n\n{LESSON_PROTOCOL}\n\n{CONTINUATION_HEADER}{tail}"
    return f"{sp}\n\n{LESSON_PROTOCOL}"


def extract_lesson(reply: str) -> str:
    """The lesson of a reply to a death observation: the text inside <lesson> (a missing
    closing tag ends at the action), whitespace collapsed; "" when there is none."""
    m = _LESSON.search(reply or "")
    return " ".join(m.group(1).split()) if m else ""


def render_lessons(lessons: list[dict[str, Any]], kept: int) -> str:
    """The pinned block: the latest `kept` lessons, numbered by death."""
    shown = lessons[-kept:]
    return "\n".join([LESSON_HEADER, *(f"- after death {l['death']}: {l['text']}" for l in shown)])


def _lesson_closing(tools: str | None, reasoning: bool, relaxed: bool = False) -> str:
    """The closing line of a death observation with death_lesson: the lesson request, then
    the usual closing turned into its second step."""
    closing = _closing(tools, reasoning, relaxed)
    closing = closing.replace("Choose the next action. Reply only as", "Then choose the next action as", 1)
    if not closing.startswith("Then "):
        closing = "Then " + closing[0].lower() + closing[1:]
    return f"{LESSON_REQUEST} {closing}"


def with_notes_protocol(system_prompt: str | None) -> str:
    """The system prompt with NOTES_PROTOCOL after the game rules, before any notes of a previous segment."""
    sp = system_prompt or ""
    if NOTES_PROTOCOL in sp:
        return sp
    if CONTINUATION_HEADER in sp:
        head, tail = sp.split(CONTINUATION_HEADER, 1)
        return f"{head.rstrip()}\n\n{NOTES_PROTOCOL}\n\n{CONTINUATION_HEADER}{tail}"
    return f"{sp}\n\n{NOTES_PROTOCOL}"


NOTES_RETRY_REASONS = ("action", "no_notes_tag", "short")


def notes_problem(reply: str, notes: str, protocol: bool = True) -> str | None:
    """Why a notes-turn reply gets the retry, else None. Notes of 80+ characters are
    accepted, inside <notes> with the notes protocol, and an action after them is
    ignored (`extract_notes` keeps only the notes; job 11399874: 15 of 17 replies retried
    for their action already held full notes). Otherwise: "action" (an action instead of
    notes), "no_notes_tag" (text without the <notes> tag the protocol asks for) or
    "short" (notes under 80 characters)."""
    tagged = _NOTES.search(reply or "") is not None
    if len(notes) >= 80 and (tagged or not protocol):
        return None
    if not tagged and _ACTION.search(reply or ""):
        return "action"
    if protocol and not tagged:
        return "no_notes_tag"
    return "short"


def extract_notes(reply: str) -> str:
    """The notes from a notes-turn reply: inside <notes>, else the reply minus any action tags."""
    m = _NOTES.search(reply or "")
    return m.group(1).strip() if m else _strip_action(reply)

FINAL_GOAL = "collect_diamond"


def _node_text(node: Any) -> str:
    m = getattr(node, "message", None)
    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    if isinstance(content, str):
        return content
    return " ".join(x.get("text", "") or "" for x in (content or []) if isinstance(x, dict))


def mask_notes_from_loss(trace: Any, summary_from: int) -> dict[str, int]:
    """train_on_notes = False: set mask False on the sampled nodes from the `summary_from`-th
    sampled one on (the notes turn and its retry) and on earlier sampled replies that hold
    <notes>, and ask the orchestrator to do the same on its copy through
    `trace.info["loss_mask"]["drop_nodes"]`. Returns how many notes-turn replies, mid-segment
    replies and tokens were masked."""
    out = {"turn": 0, "mid": 0, "tokens": 0}
    drop: list[int] = []
    k = 0
    for index, node in enumerate(getattr(trace, "nodes", None) or []):
        if not getattr(node, "sampled", False):
            continue
        k += 1
        turn = k > summary_from
        if not turn and "<notes>" not in _node_text(node):
            continue
        mask = list(getattr(node, "mask", None) or [])
        if not any(mask):
            continue
        out["turn" if turn else "mid"] += 1
        out["tokens"] += sum(mask)
        node.mask = [False] * len(mask)  # this process's copy (an in-process eval)
        drop.append(index)
    request_loss_drop(trace, drop)  # the orchestrator's copy of the graph (see there)
    return out


def continuation_system_prompt(base: str, summary: str | None) -> str:
    """The protected prefix of a continuing segment: the base system prompt plus the
    previous segment's notes (the turn window never evicts the first message)."""
    notes = summary.strip() if summary and summary.strip() else "(no notes were written)"
    return f"{base}\n\n{CONTINUATION_HEADER}\n\n{notes}"


def child_task(task: vf.Task, node: dict[str, Any], base_system_prompt: str | None = None) -> vf.Task:
    """The task of a segment that continues `node` (the env or a sampler mints it):
    the root task with the summary in the system prompt and the node in the info."""
    base = base_system_prompt if base_system_prompt is not None else (task.data.info.get("base_system_prompt") or task.data.system_prompt or "")
    info = {**task.data.info, "base_system_prompt": base, "node": node}
    if int(node.get("depth", 0)) == 0:  # a root's first segment: the base prompt, no notes yet
        return task.with_data(system_prompt=base, info=info)
    name = f"{task.data.info.get('root_name') or task.data.name}/d{node.get('depth', 0)}/{node.get('id')}"
    return task.with_data(system_prompt=continuation_system_prompt(base, node.get("summary")), prompt=None, info=info, name=name)


def _read_beam(path: str | None) -> dict[str, Any] | None:
    """The beam checkpoint of a seed (`<state_dir>/beam/<seed>.json`), if any."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("unreadable beam checkpoint %s (%s): starting over", path, exc)
        return None


def _write_beam(path: str | None, record: dict[str, Any]) -> None:
    if not path:
        return

    def write():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(record, f)
        os.replace(tmp, path)

    retry_io(write, what=f"writing the beam checkpoint {path}")


def _append_line(path: str, line: dict[str, Any]) -> None:
    """One JSON line onto the episode log, retried on a transient scratch error."""
    def write():
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")

    retry_io(write, what=f"appending to {path}")


def select_children(results: list[dict[str, Any]], keep: int, per_parent: int | None = None) -> list[dict[str, Any]]:
    """The beam rule shared by the in-env driver and the tree sampler: of the branches
    that can continue, the `keep` highest-rewarded (ties by fewer deaths, then order).
    `per_parent` caps the kept children of one parent (the sibling cap: at least
    ceil(keep / per_parent) lineages stay alive); when fewer parents qualify (the root's
    level, or a level where few branches continue) the rest of `keep` is filled from the
    best remaining branches, so the beam never shrinks because of the cap."""
    live = [r for r in results if r.get("continues")]
    ranked = [r for _, r in sorted(enumerate(live), key=lambda ir: (-float(ir[1].get("reward", 0.0)), len(ir[1].get("deaths") or []), ir[0]))]
    if per_parent is None:
        return ranked[:keep]
    kept: list[dict[str, Any]] = []
    siblings: Counter[Any] = Counter()
    for r in ranked:
        if len(kept) >= keep:
            break
        if siblings[r.get("parent")] < per_parent:
            kept.append(r)
            siblings[r.get("parent")] += 1
    if len(kept) < keep:  # not enough distinct parents: top up in rank order
        chosen = {id(r) for r in kept}
        kept += [r for r in ranked if id(r) not in chosen][: keep - len(kept)]
    return sorted(kept, key=ranked.index)


@asynccontextmanager
async def _turn_sampling(interaction: Any, max_tokens: int | None):
    """Raise the output cap for one turn (the summary). Sampling is applied per
    request from the rollout session's ModelContext (frozen), which only this rollout
    reads, so swapping the session's context touches no other branch in flight."""
    run = getattr(interaction, "_run", None)
    session = getattr(run, "_session", None)
    if session is None or max_tokens is None:
        yield
        return
    ctx = session.ctx
    session.ctx = dataclasses.replace(ctx, sampling=ctx.sampling.model_copy(update={"max_tokens": max_tokens}))
    try:
        yield
    finally:
        session.ctx = ctx


def _strip_action(text: str) -> str:
    return _ACTION.sub("", text or "").strip()


def _node_of(task: vf.Task) -> dict[str, Any]:
    node = dict(task.data.info.get("node") or {})
    node.setdefault("id", f"{task.data.info.get('seed')}-root")
    node.setdefault("depth", 0)
    node.setdefault("parent", None)
    node.setdefault("state_path", None)
    node.setdefault("summary", None)
    return node


class CrafterEnv(vf.Env[CrafterEnvConfig]):
    # --- resources ---------------------------------------------------------------
    def _store(self) -> StateStore:
        store = self.__dict__.get("_state_store")
        if store is None:
            store = self.__dict__["_state_store"] = open_store(self.config.state_dir)
        return store

    def _judge_gate(self) -> asyncio.Semaphore:
        gate = self.__dict__.get("_judge_semaphore")
        if gate is None:
            gate = self.__dict__["_judge_semaphore"] = asyncio.Semaphore(max(1, self.config.judge_concurrency))
        return gate

    async def start(self) -> None:
        self._store()
        self._judge_gate()

    def segmented(self) -> bool:
        return self.config.segment_turns is not None or self.config.segment_tokens is not None

    # --- one episode -------------------------------------------------------------
    async def run(self, task, agents):
        if self.config.tree_depth > 0:
            await self._run_tree(task, agents)
            return
        await self.play_segment(task, agents.player)

    async def finalize(self, task, episode) -> None:
        """Arm B's judge: score each segment the run left unjudged (the in-env beam
        judges its branches as they close, so this is a no-op there)."""
        cfg = self.config
        if cfg.judge is None:
            return
        judge = CrafterJudge(cfg.judge)
        common = dict(goal=cfg.final_goal, weight=cfg.judge_weight, mix=cfg.judge_mix, world_radius=cfg.judge_world_radius,
                      gate=self._judge_gate(), episode_log=cfg.episode_log, price=cfg.judge_price, tier_weight=cfg.judge_tier_weight)
        for trace in episode.traces:
            info = trace.info.get("crafter") or {}
            if not info or "judge" in info or not getattr(trace, "ok", True):
                continue
            if cfg.judge_mode == "group" and info.get("parent"):
                await judge_segment_group(self._rendezvous(), judge, CrafterGroupJudge(cfg.judge), trace, **common)
            else:
                await judge_segment(judge, trace, **common)

    def _rendezvous(self) -> GroupRendezvous:
        rv = self.__dict__.get("_judge_rendezvous")
        if rv is None:
            if not self.config.state_dir:
                raise ValueError("judge_mode = 'group' in the RL pipeline needs state_dir (the branches meet on disk)")
            rv = self.__dict__["_judge_rendezvous"] = GroupRendezvous(
                os.path.join(self.config.state_dir, "judge_groups"), self.config.judge_group_size, self.config.judge_group_timeout_s)
        return rv

    # --- arm A: the beam inside one run ---------------------------------------------
    async def _run_tree(self, task, agents) -> None:
        cfg = self.config
        store = self._store()
        judge = CrafterJudge(cfg.judge) if cfg.judge is not None else None
        seed = int(task.data.info["seed"])
        root = {"id": f"{seed}-root", "parent": None, "depth": 0, "state_path": None, "summary": None}
        beam = [root]
        found: dict[str, Any] | None = None
        root_fanout = cfg.root_branches or cfg.branches * cfg.keep
        beam_file = os.path.join(cfg.state_dir, "beam", f"{seed}.json") if cfg.state_dir else None
        start_depth = 1
        saved = _read_beam(beam_file)
        if saved:
            # a beam that stopped at its depth cap keeps its nodes: a deeper cap continues it
            if saved.get("done") and not (saved["done"] == "tree_depth" and int(saved.get("depth") or 0) < cfg.tree_depth):
                logger.info("seed %d: the beam already ended (%s) at depth %s; nothing to resume", seed, saved["done"], saved.get("depth"))
                return
            nodes = [n for n in saved.get("beam") or [] if n.get("state_path") and os.path.exists(n["state_path"])]
            if nodes:
                beam, start_depth = nodes, int(saved["depth"]) + 1
                logger.info("seed %d: resuming the beam from depth %d with %d nodes", seed, saved["depth"], len(nodes))

        common = dict(weight=cfg.judge_weight, episode_log=cfg.episode_log)

        async def node_group(node: dict[str, Any], fanout: int) -> list[Any]:
            """Group mode: a node's branches play together, then one comparative call."""
            outs = await asyncio.gather(*(self.play_segment(child_task(task, {**node, "branch": b}), agents.player)
                                          for b in range(fanout)), return_exceptions=True)
            done = [o for o in outs if not isinstance(o, BaseException) and o.get("trace") is not None]
            if judge is not None and done:
                infos = [{k: v for k, v in o["trace"].info["crafter"].items() if k != "judge"} for o in done]
                started, costs = time.time(), []
                verdicts, call_s, _ = await group_verdicts(CrafterGroupJudge(cfg.judge), infos, goal=cfg.final_goal, mix=cfg.judge_mix,
                                                           world_radius=cfg.judge_world_radius, trace=done[0]["trace"], gate=self._judge_gate(),
                                                           price=cfg.judge_price, costs=costs, tier_weight=cfg.judge_tier_weight)
                share = scale_cost(costs[0], 1.0 / len(done)) if costs else usage_cost(None, cfg.judge.model)
                for i, o in enumerate(done):
                    tr = o["trace"]
                    ended = float(tr.info["crafter"].get("ended_at") or started)
                    timing = {"group_wait_s": started - ended + call_s, "call_s": call_s, "group_size": float(len(done)),
                              "group_incomplete": float(len(done) < fanout)}
                    cost = dict(share)
                    if verdicts is None:
                        await judge_segment(judge, tr, goal=cfg.final_goal, mix=cfg.judge_mix, world_radius=cfg.judge_world_radius,
                                            gate=self._judge_gate(), price=cfg.judge_price, tier_weight=cfg.judge_tier_weight, **common)
                        single = tr.info["crafter"]["judge"]
                        cost = {k: cost[k] + float(single.get(k, 0.0)) for k in cost}
                        verdict = {k: single[k] for k in ("score", "progress", "plan", "consistency", "rationale", "failed", "llm_score", "tier") if k in single}
                        timing["group_fallback"] = 1.0
                    else:
                        verdict, timing["group_fallback"] = verdicts[i], 0.0
                    timing["wait_s"] = time.time() - ended
                    apply_verdict(tr, verdict, timing=timing, mode="group", cost=cost, **common)
                    o["reward"] = float(tr.reward)
            return list(outs)

        async def branch(node: dict[str, Any], index: int) -> dict[str, Any]:
            child = child_task(task, {**node, "branch": index})
            result = await self.play_segment(child, agents.player)
            if judge is not None and result.get("trace") is not None:
                await judge_segment(judge, result["trace"], goal=cfg.final_goal, weight=cfg.judge_weight, mix=cfg.judge_mix, world_radius=cfg.judge_world_radius, gate=self._judge_gate(), episode_log=cfg.episode_log, price=cfg.judge_price, tier_weight=cfg.judge_tier_weight)
                result["reward"] = float(result["trace"].reward)
            return result

        for depth in range(start_depth, cfg.tree_depth + 1):
            level_started = time.time()
            fanout = root_fanout if depth == 1 else cfg.branches
            jobs = [(node, b) for node in beam for b in range(fanout)]
            if cfg.judge_mode == "group":
                per_node = await asyncio.gather(*(node_group(node, fanout) for node in beam))
                outcomes = [o for outs in per_node for o in outs]
            else:
                outcomes = await asyncio.gather(*(branch(node, b) for node, b in jobs), return_exceptions=True)
            for node in beam:  # every branch restored it; the beam never revisits
                if node.get("state_path"):
                    store.delete(node["state_path"])
            results = []
            for (node, b), out in zip(jobs, outcomes):
                if isinstance(out, BaseException):
                    logger.warning("branch %s/%d at depth %d failed: %s", node["id"], b, depth, out)
                    continue
                results.append(out)
            reached = [r for r in results if r.get("final_goal_reached")]
            if reached:
                found = max(reached, key=lambda r: float(r.get("reward", 0.0)))
                for r in results:
                    if r.get("state_path"):
                        store.delete(r["state_path"])
                break
            kept = select_children(results, cfg.keep, cfg.keep_per_parent)
            kept_paths = {r.get("state_path") for r in kept}
            for r in results:
                if r.get("state_path") and r["state_path"] not in kept_paths:
                    store.delete(r["state_path"])
            beam = [
                {"id": r["node_id"], "parent": r["parent"], "depth": depth, "state_path": r["state_path"],
                 "state_sha256": r.get("state_sha256"), "summary": r.get("summary")}
                for r in kept
            ]
            best = max((float(r.get("reward", 0.0)) for r in kept), default=float("nan"))
            logger.info("seed %d depth %d: %d branches, kept %d (best reward %.3f, achievements %s), %.1f min",
                        seed, depth, len(results), len(kept), best,
                        max((len(r.get("achievements") or []) for r in kept), default=0), (time.time() - level_started) / 60)
            _write_beam(beam_file, {"seed": seed, "depth": depth, "beam": beam, "done": None, "at": time.time()})
            # Every finished segment's trace stays in the episode, so the worker's heap grows a
            # level at a time; freezing it keeps the cyclic GC from re-walking old levels on
            # every full collection (a 0.7 s pause every ~20 s at depth 1, growing with depth).
            gc.collect()
            gc.freeze()
            if not beam:
                break
        done = "final_goal" if found is not None else ("exhausted" if not beam else "tree_depth")
        if done != "tree_depth" or not beam_file:  # at the cap, the nodes stay on disk for a deeper run
            for node in beam:
                if node.get("state_path"):
                    store.delete(node["state_path"])
            beam = []
        _write_beam(beam_file, {"seed": seed, "depth": found["depth"] if found is not None else min(depth, cfg.tree_depth),
                                "beam": beam, "done": done, "at": time.time()})
        if found is not None:
            logger.info("seed %d reached %s at depth %d (node %s)", seed, cfg.final_goal, found["depth"], found["node_id"])

    # --- one segment -------------------------------------------------------------
    async def play_segment(self, task, agent) -> dict[str, Any]:
        """Play one segment of `task` with `agent` and return what the search needs:
        the node ids, whether the game continues, the saved state's path, the
        summary and the reward so far (the judge, if any, is applied by the caller)."""
        cfg = self.config
        segmented = cfg.segment_turns is not None or cfg.segment_tokens is not None
        node = _node_of(task)
        seed = int(task.data.info["seed"])
        max_steps = cfg.max_episode_steps
        started = time.time()
        store = self._store() if segmented else None
        unlocked_at: dict[str, int] = {}  # achievement -> game step it was first unlocked, this segment
        spawn = last_sleep = last_base = None
        if node.get("state_path"):
            saved = await asyncio.to_thread(store.get, node["state_path"], node.get("state_sha256"))
            env = saved["env"]
            spawn, last_sleep = tuple(saved["spawn"]), tuple(saved["last_sleep"])
            last_base = tuple(saved.get("last_base") or last_sleep)
        else:
            env = crafter.Env(seed=seed, length=max_steps)
            await asyncio.to_thread(env.reset)
        book = nb.Notebook(**task.data.info["notebook"]) if task.data.info.get("notebook") else None
        space = ws.Workspace(**task.data.info["workspace"]) if task.data.info.get("workspace") else None
        tools = "notebook" if book is not None else "workspace" if space is not None else None
        reasoning = bool(task.data.info.get("reasoning"))
        relaxed = bool(task.data.info.get("relaxed_reply_format"))
        seen: set[tuple[str, int, int]] = set()  # ledger sightings: (kind, 5x5 block)
        if space is not None:
            for line in _ledger_sightings(env, 0, seen, cfg.view_radius):
                space.log(line)
        current = _player_info(env)
        achievements_before = {name for name, count in current["achievements"].items() if int(count) > 0}
        inventory_before = {k: int(v) for k, v in current["inventory"].items() if k not in _VITALS and int(v)}
        position_before = list(current["player_pos"])
        events: list[str] = []  # the segment's timeline for the judge: unlocks, deaths, table/furnace, sleep
        visited: set[tuple[int, int]] = {tuple(position_before)}
        world_length = int(getattr(env, "_length", max_steps) or max_steps)
        segment_line = (lambda t: f"Segment turn: {t}/{cfg.segment_turns}" if cfg.segment_turns else f"Segment turn: {t}") if segmented else (lambda t: None)
        opening = None
        if task.data.prompt is None:
            opening = _format_observation(
                env=env,
                info=current,
                step=int(getattr(env, "_step", 0) or 0),
                max_steps=world_length,
                view_radius=cfg.view_radius,
                tools=tools,
                extras=cfg.observation_extras,
                affordances=cfg.extras_affordances,
                reasoning=reasoning,
                relaxed=relaxed,
                segment=segment_line(0),
            )
            if space is not None:
                opening = f"{space.index()}\n\n{opening}"
            if book is not None:
                opening = f"{book.render()}\n\n{opening}"
        total_reward, invalid, steps, done = 0.0, 0, 0, False
        malformed = 0  # replies without an <action> tag
        actions: Counter[str] = Counter()
        end = "terminated"  # the run stopped answering (a turn or token limit)
        if spawn is None:
            spawn = tuple(int(x) for x in env._player.pos)
            last_sleep = last_base = spawn
        checkpoint: bytes | None = None  # on_death = revert: the pickled save point and the marks that go with it
        checkpoint_step = segment_start_step = int(getattr(env, "_step", 0) or 0)
        if cfg.on_death == "revert":
            checkpoint = StateStore.dumps({"env": env, "spawn": spawn, "last_sleep": last_sleep, "last_base": last_base})
        deaths: list[dict[str, Any]] = []  # one entry per death: step, vitals, where it came back
        respawned: str | None = None  # told to the model in the next observation
        raw_action: str | None = None  # the last action the model wrote, as in v0's parse_answer
        final_goal_reached = False
        summary: str | None = None
        summary_tokens = 0
        summary_retried = False
        retry_reason: str | None = None  # notes_problem() of the first notes reply
        continues = False
        lessons: list[dict[str, Any]] = []  # death_lesson: what the model wrote after each death
        lesson_asked = False  # the observation just sent asked for a lesson
        lessons_missing = 0  # replies to that request without a <lesson>
        sampled_before_summary = 10**9  # sampled nodes before the notes request (train_on_notes = False)
        if segmented and cfg.notes_protocol:
            task = task.with_data(system_prompt=with_notes_protocol(task.data.system_prompt))
        if segmented and cfg.recall_probe:
            task = task.with_data(system_prompt=with_recall_protocol(task.data.system_prompt))
        recall = None  # the recall probe's scored reply (recall_probe = True)
        sampled_before_probe = 10**9
        if cfg.death_lesson:
            task = task.with_data(system_prompt=with_lesson_protocol(task.data.system_prompt))
        try:
            async with agent.interaction(task) as interaction:
                segment = await interaction.turn(opening)
                while not segment.terminated:
                    steps += 1
                    if not segmented and steps >= max_steps:
                        end = "max_episode_steps"
                        break  # as in v0: the capped turn's action is not played
                    notes = book.apply(segment.last_reply) if book is not None else []
                    results = space.apply(segment.last_reply, steps) if space is not None else []
                    if lesson_asked:
                        lesson_asked = False
                        text = extract_lesson(segment.last_reply)
                        if text:
                            lessons.append({"death": len(deaths), "turn": steps, "chars": len(text), "text": text[:cfg.lesson_max_chars]})
                            events.append(f"step {int(getattr(env, '_step', 0) or 0)}: lesson after death {len(deaths)}: {text[:160]}")
                        else:
                            lessons_missing += 1
                    before = dict(current["achievements"])
                    health_before = int(current["inventory"].get("health", 0))
                    found = _ACTION.findall(segment.last_reply)
                    repeated = None
                    if found:
                        raw_action = found[0]
                    else:
                        # No action tag: the last action written stands in (v0's rule), and
                        # the model is told - silently repeating it made "<" look like a
                        # working reply, which the model then copied turn after turn.
                        malformed += 1
                        repeated = _normalize_action(raw_action)
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
                        for line in _ledger_sightings(env, steps, seen, cfg.view_radius):
                            space.log(line)
                    game_step = int(getattr(env, "_step", steps) or steps)
                    for name, count in current["achievements"].items():
                        if count > 0 and name not in unlocked_at and name not in achievements_before:
                            unlocked_at[name] = game_step
                            events.append(f"step {game_step}: unlocked {name}")
                    if int(current["achievements"].get("place_table", 0)) > int(before.get("place_table", 0)):
                        last_base = tuple(int(v) for v in env._player.pos)
                        events.append(f"step {game_step}: placed a table at {tuple(int(v) for v in env._player.pos + env._player.facing)}")
                        if checkpoint is not None and cfg.revert_to == "last_base" and int(current["inventory"].get("health", 0)) > 0:
                            checkpoint = StateStore.dumps({"env": env, "spawn": spawn, "last_sleep": last_sleep, "last_base": last_base})
                            checkpoint_step = game_step
                            events.append(f"step {game_step}: save point (table)")
                    if int(current["achievements"].get("place_furnace", 0)) > int(before.get("place_furnace", 0)):
                        events.append(f"step {game_step}: placed a furnace at {tuple(int(v) for v in env._player.pos + env._player.facing)}")
                    if env._player.sleeping:
                        if last_sleep != tuple(int(x) for x in env._player.pos):
                            events.append(f"step {game_step}: slept at {tuple(int(x) for x in env._player.pos)}")
                            if checkpoint is not None and cfg.revert_to == "last_base":
                                checkpoint = StateStore.dumps({"env": env, "spawn": spawn, "last_sleep": last_sleep, "last_base": last_base})
                                checkpoint_step = game_step
                                events.append(f"step {game_step}: save point (sleep)")
                        last_sleep = last_base = tuple(int(x) for x in env._player.pos)
                    visited.add(tuple(int(x) for x in env._player.pos))
                    died = current["inventory"].get("health", 0) <= 0
                    world_over = env._step >= env._length
                    if died and space is not None:
                        space.log(_ledger_death(env, steps, current))
                    reverted = False
                    if died and not world_over and cfg.on_death == "revert" and checkpoint is not None:
                        vitals = {k: int(current["inventory"].get(k, 0)) for k in _VITALS}
                        starved = [k for k in ("food", "drink", "energy") if vitals.get(k, 1) == 0]
                        cause = _death_cause(env, vitals)
                        daylight = round(float(getattr(env._world, "daylight", 1.0)), 2)
                        died_at = tuple(int(v) for v in current["player_pos"])
                        close = getattr(env, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                pass
                        saved = StateStore.loads(checkpoint, "checkpoint", None)
                        env = saved["env"]
                        spawn, last_sleep, last_base = saved["spawn"], saved["last_sleep"], saved["last_base"]
                        current = _player_info(env)
                        world_length = int(getattr(env, "_length", max_steps) or max_steps)
                        pos = tuple(int(x) for x in env._player.pos)
                        kept = {name for name, count in current["achievements"].items() if int(count) > 0}
                        lost_achievements = sorted(name for name in unlocked_at if name not in kept)
                        unlocked_at = {name: at_step for name, at_step in unlocked_at.items() if name in kept}
                        events.append(
                            f"step {game_step}: died at {died_at} ({cause}{', at 0: ' + ', '.join(starved) if starved else ''}); "
                            f"the world was restored to step {checkpoint_step}"
                            + (f", losing {', '.join(lost_achievements)}" if lost_achievements else "")
                        )
                        if space is not None:
                            space.log(f"step {steps}: the world was restored to step {checkpoint_step} ({cfg.revert_to.replace('_', ' ')})")
                        deaths.append({
                            "step": steps, "vitals": vitals, "pos": list(died_at), "respawn": list(pos),
                            "cause": cause, "daylight": daylight, "game_step": game_step,
                            "respawn_hostile": _hostile_near(env._world, pos, 8),
                            "since": steps - (deaths[-1]["step"] if deaths else 0),
                            "revert_step": checkpoint_step, "lost_turns": game_step - checkpoint_step,
                            "lost_achievements": lost_achievements,
                        })
                        how = _death_how(cause, starved, daylight)
                        where = "the start of this segment" if checkpoint_step == segment_start_step else "your last save point"
                        respawned = (
                            f"You died (death {len(deaths)}, at {died_at}, {how}, game step {game_step}). "
                            f"The game was restored to game step {checkpoint_step} ({where}): your position, inventory, "
                            f"achievements, health and the time of day are as they were then. Everything you did after "
                            f"that is undone"
                            + (f" (you no longer have: {', '.join(lost_achievements)})" if lost_achievements else "")
                            + ". You still remember it, but the turns you spent are not given back."
                        )
                        reverted = True
                        done = False
                        if segmented and cfg.cut_on_death:
                            end = "death_cut"
                            observation = f"{respawned}\n\n" + _format_observation(
                                env=env, info=current, step=int(getattr(env, "_step", steps) or steps), max_steps=world_length,
                                view_radius=cfg.view_radius, tools=tools, extras=cfg.observation_extras,
                                affordances=cfg.extras_affordances, reasoning=reasoning, relaxed=relaxed, segment=segment_line(steps),
                            )
                            break
                    elif died and not world_over and cfg.on_death == "respawn":
                        vitals = {k: int(current["inventory"].get(k, 0)) for k in _VITALS}
                        at = {"spawn": spawn, "last_sleep": last_sleep, "last_base": last_base}.get(
                            cfg.respawn_at, tuple(int(x) for x in env._player.pos)
                        )
                        starved = [k for k in ("food", "drink", "energy") if vitals.get(k, 1) == 0]
                        cause = _death_cause(env, vitals)
                        daylight = round(float(getattr(env._world, "daylight", 1.0)), 2)
                        events.append(
                            f"step {game_step}: died at {tuple(int(v) for v in current['player_pos'])} "
                            f"({cause}{', at 0: ' + ', '.join(starved) if starved else ''}), respawned at {cfg.respawn_at}"
                        )
                        pos = _respawn(env, at, cfg.death_inventory, cfg.respawn_safe, cfg.respawn_safe_radius)
                        if space is not None:
                            space.log(f"step {steps}: respawned at {pos} ({cfg.death_inventory.replace('_', ' ')})")
                        deaths.append({
                            "step": steps, "vitals": vitals, "pos": list(current["player_pos"]), "respawn": list(pos),
                            "cause": cause, "daylight": daylight, "game_step": game_step,
                            "respawn_hostile": _hostile_near(env._world, pos, 8),  # a zombie or skeleton within 8 of the respawn cell (any arm)
                            "since": steps - (deaths[-1]["step"] if deaths else 0),  # turns since the last respawn (or the segment start)
                        })
                        current = _step_info(env, {}, 0.0)
                        lost = {"drop_all": "lost your inventory", "drop_materials": "lost your materials (tools kept)", "keep": "kept your inventory"}
                        died_line = (
                            f"You died (death {len(deaths)}, at {tuple(deaths[-1]['pos'])}, {_death_how(cause, starved, daylight)}, game step {game_step})."
                            if cfg.announce_death_cause
                            else f"You died (death {len(deaths)}, at {tuple(deaths[-1]['pos'])})."
                        )
                        respawned = (
                            f"{died_line} You respawned at {pos} "
                            f"with full health, food, drink and energy, and {lost[cfg.death_inventory]}. "
                            "Unlocked achievements stay unlocked."
                        )
                        done = False
                        if segmented and cfg.cut_on_death:
                            end = "death_cut"
                            observation = f"{respawned}\n\n" + _format_observation(
                                env=env, info=current, step=int(getattr(env, "_step", steps) or steps), max_steps=world_length,
                                view_radius=cfg.view_radius, last_action=action, tools=tools, extras=cfg.observation_extras,
                                affordances=cfg.extras_affordances, reasoning=reasoning, relaxed=relaxed, segment=segment_line(steps),
                            )
                            break
                    elif done:
                        end = "died" if died else "world_length"
                        break
                    if segmented and int(current["achievements"].get(cfg.final_goal, 0)) > 0 and cfg.final_goal not in achievements_before:
                        final_goal_reached = True
                        end = "final_goal"
                        break
                    observation = _format_observation(
                        env=env,
                        info=current,
                        step=int(getattr(env, "_step", steps) or steps),
                        max_steps=world_length,
                        view_radius=cfg.view_radius,
                        last_action=None if reverted else action,
                        invalid_action=None if reverted else invalid_action,
                        tools=tools,
                        extras=cfg.observation_extras,
                        affordances=cfg.extras_affordances,
                        reasoning=reasoning,
                        relaxed=relaxed,
                        repeated_action=repeated if cfg.announce_malformed else None,
                        health_lost=max(0, health_before - int(current["inventory"].get("health", 0))) if respawned is None else 0,
                        segment=segment_line(steps),
                        closing=_lesson_closing(tools, reasoning, relaxed) if cfg.death_lesson and respawned is not None else None,
                    )
                    lesson_asked = cfg.death_lesson and respawned is not None
                    if respawned is not None:
                        observation, respawned = f"{respawned}\n\n{observation}", None
                    if cfg.death_lesson and lessons:
                        observation = f"{render_lessons(lessons, cfg.lessons_kept)}\n\n{observation}"
                    if book is not None:
                        observation = "\n".join([book.render(), *notes, "", observation])
                    if space is not None:
                        observation = "\n".join([space.index(), *results, "", observation])
                    if segmented and cfg.notes_protocol and cfg.notes_warning_turns:
                        left = []
                        if cfg.segment_turns is not None:
                            left.append(cfg.segment_turns - steps)
                        if cfg.segment_tokens is not None and steps:
                            per_turn = interaction.trace.num_total_tokens / steps
                            left.append(int((cfg.segment_tokens - interaction.trace.num_total_tokens) // max(per_turn, 1)))
                        if left and 0 < min(left) <= cfg.notes_warning_turns:
                            observation = (f"This segment ends in about {min(left)} turn{'s' * (min(left) > 1)}; then you will be "
                                           f"asked for your notes (END OF SEGMENT).\n{observation}")
                    # The cut comes before the next action is sampled: the last observation
                    # goes to the model with the summary request instead.
                    if segmented and cfg.segment_turns is not None and steps >= cfg.segment_turns:
                        end = "segment_turns"
                        break
                    if segmented and cfg.segment_tokens is not None and interaction.trace.num_total_tokens >= cfg.segment_tokens:
                        end = "segment_tokens"
                        break
                    segment = await interaction.turn(observation)
                # The cut: the game goes on, so the policy writes its notes in this same
                # session (prefix + live window + the recurrent state) and the world is saved.
                continues = segmented and end in {"segment_turns", "segment_tokens", "death_cut"} and not final_goal_reached
                if continues:
                    request = cfg.summary_prompt or SUMMARY_PROMPT
                    shown = observation
                    if cfg.notes_protocol or cfg.summary_retry:  # the observation's last line asks for an action: drop it
                        shown = observation.rsplit("\n", 1)[0] if "<action>" in observation.rsplit("\n", 1)[-1] else observation
                    message = f"{request}\n\nThe state at the end of the segment:\n{shown}" if cfg.notes_protocol else f"{shown}\n\n{request}"

                    def notes_cap() -> int | None:
                        # The output cap of a notes turn: summary_max_tokens, cut to what the context
                        # still holds. SGLang refuses a request whose logical length plus max_new_tokens
                        # passes the model's limit (job 11399874: a retry asked for 32768 at 99.7k of 131072).
                        limit = cfg.player.max_total_tokens
                        if limit is None:
                            return cfg.summary_max_tokens
                        room = limit - interaction.trace.num_total_tokens - 1024  # the request message itself
                        return max(1, min(cfg.summary_max_tokens or room, room))

                    reply = None
                    if cfg.recall_probe:
                        # The probe comes first: the notes request would show the model what to write.
                        sampled_before_probe = sum(1 for n in (getattr(interaction.trace, "nodes", None) or []) if getattr(n, "sampled", False))
                        probe = await interaction.turn(f"{RECALL_PROBE}\n\nThe state at the end of the segment:\n{shown}")
                        if probe.terminated:
                            reply = probe  # the session ended at the probe: no notes turn
                        else:
                            recall = score_recall_probe(probe.last_reply, deaths, unlocked_at, steps)
                    sampled_before_summary = sum(1 for n in (getattr(interaction.trace, "nodes", None) or []) if getattr(n, "sampled", False))
                    if reply is None:
                        async with _turn_sampling(interaction, notes_cap()):
                            reply = await interaction.turn(message)
                    if not reply.terminated:
                        summary = extract_notes(reply.last_reply)
                        retry_reason = notes_problem(reply.last_reply, summary, cfg.notes_protocol)
                        cap = notes_cap()
                        if cfg.summary_retry and retry_reason is not None and (cap is None or cap >= 256):
                            summary_retried = True
                            async with _turn_sampling(interaction, cap):
                                reply = await interaction.turn(SUMMARY_RETRY)
                            if not reply.terminated:
                                summary = extract_notes(reply.last_reply) or summary
                    if not reply.terminated:
                        summary_tokens = len(summary) // 4
                    else:
                        end = "terminated_at_summary"
        finally:
            if not continues:
                close = getattr(env, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        trace = interaction.trace
        notes_masked = {"turn": 0, "mid": 0, "tokens": 0}
        if not cfg.train_on_notes:
            notes_masked = mask_notes_from_loss(trace, sampled_before_summary)
        if cfg.recall_probe and sampled_before_probe < 10**9:
            drop_sampled_range(trace, sampled_before_probe, sampled_before_summary)  # the probe reply is never trained on
        trace_id = getattr(trace, "id", None) or uuid.uuid4().hex
        state_path = state_sha = None
        state_bytes = 0
        if continues:
            saved = {"env": env, "spawn": list(spawn), "last_sleep": list(last_sleep), "last_base": list(last_base), "seed": seed}
            state_path, state_sha, state_bytes = await asyncio.to_thread(store.put, trace_id, saved)
        unlocked = sorted(name for name, count in (current.get("achievements") or {}).items() if int(count) > 0)
        delta = sorted(name for name in unlocked if name not in achievements_before)
        # The death penalty is its own reward term so that it survives achievement_weight = 0
        # (the judge alone scores progress, the deaths are still paid for).
        penalty = cfg.death_penalty * len(deaths)
        if segmented:
            reward = len(delta) / max(len(ACHIEVEMENTS), 1)
            trace.record_reward("achievements", reward, cfg.achievement_weight)
        else:
            reward = len(unlocked) / max(len(ACHIEVEMENTS), 1)
            trace.record_reward("achievements", reward, 1.0)
        if penalty:
            trace.record_reward("deaths", -penalty, 1.0)
        reward -= penalty
        survived = 1.0 if int((current.get("inventory") or {}).get("health", 0)) > 0 else 0.0
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(unlocked)),
                "segment_achievements": float(len(delta)),
                "total_env_reward": total_reward,
                "invalid_actions": float(invalid),
                "malformed_replies": float(malformed),
                "notes_masked_replies": float(notes_masked["turn"] + notes_masked["mid"]),
                "notes_masked_mid": float(notes_masked["mid"]),
                "notes_masked_tokens": float(notes_masked["tokens"]),
                **recall_metrics(recall),
                "survived": survived,
                "deaths": float(len(deaths)),
                **{f"deaths_{c}": float(sum(d.get("cause") == c for d in deaths)) for c in DEATH_CAUSES},
                "deaths_night": float(sum(d.get("daylight", 1.0) < 0.5 for d in deaths)),
                "respawns_near_hostile": float(sum(bool(d.get("respawn_hostile")) for d in deaths)),
                "reverted_turns": float(sum(int(d.get("lost_turns", 0)) for d in deaths)),
                "reverted_achievements": float(sum(len(d.get("lost_achievements") or []) for d in deaths)),
                "lessons": float(len(lessons)),
                "lessons_missing": float(lessons_missing),
                "lesson_chars": float(sum(l["chars"] for l in lessons) / len(lessons)) if lessons else 0.0,
                "depth": float(node.get("depth", 0)),
                "continues": float(continues),
                "final_goal": float(final_goal_reached),
                "summary_chars": float(len(summary or "")),
                "summary_retried": float(summary_retried),
                **{f"summary_retry_{r}": float(summary_retried and retry_reason == r) for r in NOTES_RETRY_REASONS},
                **({"memory_writes": float(book.writes), "scratchpad_updates": float(book.scratchpad_updates)} if book else {}),
                **(
                    {f"workspace_{k}": float(space.counts.get(k, 0)) for k in ("search", "read", "write", "append")}
                    if space
                    else {}
                ),
            }
        )
        node_id = trace_id
        record = {
            "achievements": unlocked,
            "segment_achievements": delta,
            "unlocked_at": unlocked_at,
            "recall": recall,
            "steps": steps,
            "game_step": int(getattr(env, "_step", steps) or steps),
            "done": done,
            "ended_by": end,
            "seed": seed,
            "deaths": deaths,
            "notebook": book.summary() if book is not None else None,
            "workspace": space.summary() if space is not None else None,
            # the segment's place in the search
            "node_id": node_id,
            "parent": node.get("id"),
            "depth": int(node.get("depth", 0)) + 1 if segmented else 0,
            "branch": node.get("branch"),
            "continues": continues,
            "final_goal_reached": final_goal_reached,
            "summary": summary,
            "summary_tokens": summary_tokens,
            "summary_retry_reason": retry_reason if summary_retried else None,
            "state_path": state_path,
            "state_sha256": state_sha,
            "state_bytes": state_bytes,
            "inventory": current.get("inventory") or {},
            "position": list(current.get("player_pos") or [0, 0]),
            "local_map": _format_local_map(current, cfg.view_radius) if segmented else None,
            # what the judge compares against: where the segment started, what happened, what the
            # policy could not see, and the tech tree's own next step
            "inventory_before": inventory_before,
            "achievements_before": sorted(achievements_before),
            "position_before": position_before,
            "events": events[:60],
            "cells_visited": len(visited),
            "invalid_actions": invalid,
            "malformed_replies": malformed,
            "world_nearby": _nearby(env, cfg.judge_world_radius)[1] if segmented and cfg.judge_world_radius else None,
            "next_tier": _next_tier(env) if segmented else None,
            "respawn_at": cfg.respawn_at,
            "on_death": cfg.on_death,
            "revert_to": cfg.revert_to if cfg.on_death == "revert" else None,
            "announce_death_cause": cfg.announce_death_cause,
            "death_lesson": cfg.death_lesson,
            "train_on_notes": cfg.train_on_notes,
            "notes_masked": notes_masked,
            "lessons": lessons,
            "lessons_missing": lessons_missing,
            "ended_at": time.time(),
        }
        trace.info["crafter"] = record
        if cfg.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "reward": reward,
                "achievements": len(unlocked),
                "invalid_actions": invalid,
                "malformed_replies": malformed,
                "steps": steps,
                "done": done,
                "survived": survived,
                "ended_by": end,
                "deaths": deaths,
                "notebook": record["notebook"],
                "workspace": record["workspace"],
                "seed": seed,
                "unlocked_at": unlocked_at,
            "recall": recall,
                "actions": dict(actions),
                "inventory": record["inventory"],
                "trace_tokens": trace.num_total_tokens,
                "task": "crafter",
                "node_id": node_id,
                "parent": record["parent"],
                "depth": record["depth"],
                "branch": record["branch"],
                "game_step": record["game_step"],
                "segment_achievements": delta,
                "continues": continues,
                "final_goal_reached": final_goal_reached,
                "summary_chars": len(summary or ""),
                "summary_retried": summary_retried,
                "summary_retry_reason": retry_reason if summary_retried else None,
                "summary": summary,  # the notes themselves: what the next segment starts from
                "lessons": lessons,
                "lessons_missing": lessons_missing,
                "state_path": state_path,
                "state_bytes": state_bytes,
                # the judge's inputs, so scripts/crafter/judge_replay.py can re-score the segment
                "achievements_unlocked": unlocked,
                "achievements_before": record["achievements_before"],
                "inventory_before": inventory_before,
                "position": record["position"],
                "position_before": position_before,
                "events": record["events"],
                "cells_visited": len(visited),
                "local_map": record["local_map"],
                "world_nearby": record["world_nearby"],
                "next_tier": record["next_tier"],
            }
            _append_line(cfg.episode_log, line)
        return {**record, "trace": trace, "reward": float(getattr(trace, "reward", reward))}


class CrafterTaskset(vf.Taskset[CrafterTask, CrafterConfig]):
    def load(self) -> Iterator[CrafterTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = REASONING_SYSTEM_PROMPT if cfg.enable_reasoning else SYSTEM_PROMPT
        if cfg.list_achievements:
            system = with_achievement_list(system)
        if cfg.notebook and cfg.workspace:
            raise ValueError("notebook and workspace are alternative memories: enable one")
        if cfg.workspace and not (cfg.workspace_files or cfg.workspace_ledger):
            raise ValueError("a workspace needs files, a ledger or both")
        notebook = workspace = None
        tools = None
        if cfg.notebook:
            notebook = {"slots": cfg.notebook_slots, "memory_chars": cfg.memory_chars, "scratchpad_chars": cfg.scratchpad_chars}
            system, tools = f"{_admit_tools(system, 'notebook')}\n\n{nb.instructions(**notebook)}", "notebook"
        if cfg.relaxed_reply_format and not (cfg.notebook or cfg.workspace or cfg.enable_reasoning):
            for old, new in _RELAXED_REPLY_FORMAT.items():
                system = system.replace(old, new)
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
                observation = _initial_observation(seed, cfg.max_episode_steps, cfg.view_radius, tools, cfg.enable_reasoning)
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
                        "reasoning": cfg.enable_reasoning,
                        "relaxed_reply_format": cfg.relaxed_reply_format,
                    },
                ),
                cfg.task,
            )

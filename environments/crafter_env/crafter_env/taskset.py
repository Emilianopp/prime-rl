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
from collections.abc import Iterator
from typing import Any, Literal

import crafter
from crafter import constants
from pydantic import Field

import verifiers.v1 as vf

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
            "Choose the next action. Reply only as <action>action_name</action>.",
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


class CrafterEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "inline-chat"})
    max_episode_steps: int = Field(100, ge=1)
    """Model turns (game actions) before the episode is cut off; also the world's length."""
    view_radius: int = Field(4, ge=1)
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, reward, achievements) to this file."""


def _initial_observation(seed: int, max_steps: int, view_radius: int) -> str:
    env = crafter.Env(seed=seed, length=max_steps)
    env.reset()
    return _format_observation(env=env, info=_player_info(env), step=0, max_steps=max_steps, view_radius=view_radius)


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
        opening = None
        if task.data.prompt is None:
            opening = _format_observation(
                env=env, info=_player_info(env), step=0, max_steps=max_steps, view_radius=self.config.view_radius
            )
        total_reward, invalid, steps, done = 0.0, 0, 0, False
        raw_action: str | None = None  # the last action the model wrote, as in v0's parse_answer
        current = _player_info(env)
        try:
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn(opening)
                while not segment.terminated:
                    steps += 1
                    if steps >= max_steps:
                        break  # as in v0: the capped turn's action is not played
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
                    current = _step_info(env, info, reward)
                    if done:
                        break
                    observation = _format_observation(
                        env=env,
                        info=current,
                        step=steps,
                        max_steps=max_steps,
                        view_radius=self.config.view_radius,
                        last_action=action,
                        invalid_action=invalid_action,
                    )
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
        reward = len(unlocked) / max(len(ACHIEVEMENTS), 1)
        trace.record_reward("achievements", reward, 1.0)
        survived = 1.0 if int((current.get("inventory") or {}).get("health", 0)) > 0 else 0.0
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(unlocked)),
                "total_env_reward": total_reward,
                "invalid_actions": float(invalid),
                "survived": survived,
            }
        )
        trace.info["crafter"] = {"achievements": unlocked, "steps": steps, "done": done, "seed": seed}
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
                "task": "crafter",
            }
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class CrafterTaskset(vf.Taskset[CrafterTask, CrafterConfig]):
    def load(self) -> Iterator[CrafterTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = REASONING_SYSTEM_PROMPT if cfg.enable_reasoning else SYSTEM_PROMPT
        for i in range(cfg.num_examples):
            seed = base + i
            if cfg.include_initial_obs_in_system:
                # The system prompt carries the opening observation, so this renders a
                # world per task up front (about a second each).
                observation = _initial_observation(seed, cfg.max_episode_steps, cfg.view_radius)
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
                    info={"seed": seed, "episode_idx": i, "split": cfg.split},
                ),
                cfg.task,
            )

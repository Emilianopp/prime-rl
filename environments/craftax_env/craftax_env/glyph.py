"""GlyphBench's full Craftax, without the GlyphBench package.

Ported from GlyphBench (https://github.com/VmaxAI/glyphbench, MIT, at 6cc45a4):
``envs/craftaxfull/fork.py`` (the game wrapper and its HUD),
``verifiers_integration/prompting.py`` (system prompt blocks and the per-turn
user message) and ``protocol.py`` / ``core/action.py`` (action parsing). The
game, the Unicode renderer and the action table come from the Craftax fork
GlyphBench pins (roger-creus/Craftax at ef0cd47), so observations are the
ones GlyphBench's models saw. The GlyphBench package itself pins its own
verifiers revision and Python 3.12 only, so it is not a dependency.

JAX runs on CPU: a game step is ~1.5 ms and a render ~20 ms there, and the
eval process must not take GPU memory from the inference server.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from craftax.craftax import glyphbench_api as api

ENV_ID = "glyphbench/craftaxfull-v0"
ACTION_NAMES: tuple[str, ...] = api.ACTION_NAMES
ACTION_DESCRIPTIONS: tuple[str, ...] = api.ACTION_DESCRIPTIONS
NOOP = ACTION_NAMES.index("NOOP")
MAX_RETURN = 226.0
"""Sum of Craftax's per-achievement rewards (1/3/5/8 by tier, 67 achievements):
GlyphBench's "% of max reward" is 100 * return / 226."""
FAST_FORWARD_MAX_TICKS = 10000
ENDLESS_TIMESTEPS = 1_000_000_000
"""The game's own time limit when a game has no turn limit (int32, out of reach)."""
FLOOR_NAMES = (
    "Overworld",
    "Dungeon",
    "Gnomish Mines",
    "Sewers",
    "Vaults",
    "Troll Mines",
    "Fire Realm",
    "Ice Realm",
    "Graveyard",
)

_runtime: dict[str, Any] = {}
_runtime_lock = threading.Lock()


def _load(max_timesteps: int) -> tuple[Any, Any, Any]:
    """One Craftax env per process: its step is jitted with the env as a static
    argument, so a new instance per episode would recompile (seconds) every time."""
    with _runtime_lock:
        if "env" not in _runtime:
            import jax
            from craftax.craftax import ascii_renderer
            from craftax.craftax.constants import Achievement
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset

            _runtime.update(
                jax=jax,
                env=CraftaxSymbolicEnvNoAutoReset(),
                renderer=ascii_renderer,
                achievements=tuple(Achievement),
                values=api.action_values(),
                params={},
            )
            from craftax.craftax import constants as C
            from craftax.craftax.util import game_logic_utils as U
            from craftax.craftax.world_gen.world_gen import get_new_empty_inventory

            blocked = [C.BlockType.WATER.value, C.BlockType.LAVA.value]
            _runtime.update(
                weights=np.asarray(C.ACHIEVEMENT_REWARD_MAP, dtype=float),
                not_walkable=np.asarray(C.SOLID_BLOCK_MAPPING) | np.isin(np.arange(len(C.BlockType)), blocked),
                max_vitals=U,
                empty_inventory=get_new_empty_inventory,
            )
        params = _runtime["params"]
        if max_timesteps not in params:
            from craftax.craftax.craftax_state import EnvParams

            params[max_timesteps] = EnvParams(max_timesteps=int(max_timesteps))
        return _runtime["env"], params[max_timesteps], _runtime


# --- system prompt (GlyphBench's build_system_prompt, no memory mode) ---

_MECHANICS = (
    "CRAFTAX MECHANICS\n"
    "Craftax mechanics means the upstream Craftax rules from the bundled "
    "Craftax submodule: movement and facing, DO interactions, walkability, "
    "blocking and collision, HP, food, drink, energy, mana, day/night, mobs, "
    "combat, projectiles, drops, resources, crafting, placement, plants, "
    "water, lava, inventory/equipment state, achievements, rewards, floor "
    "transitions and gates, bows/arrows, books, spells, potions, enchantments, "
    "bosses, and terminal conditions. Registered glyphbench/craftax-* tasks "
    "use the universal Craftax action menu; actions that are irrelevant or "
    "blocked by a focused TASK/SCENARIO setup remain part of the action space "
    "and are unavailable only because of that scenario state, not because this "
    "is a separate Craftax variant."
)


def _actions_block() -> str:
    lines = ["[Actions]", "Pick exactly one per turn:"]
    lines += [f"  {name} — {desc}" for name, desc in zip(ACTION_NAMES, ACTION_DESCRIPTIONS, strict=True)]
    return "\n".join(lines)


GAME_PROMPT = (
    f"You are playing {ENV_ID}, the full Craftax roguelike from "
    "the project submodule.\n\n"
    "TASK\n"
    "Survive, explore, craft equipment, clear floors, and defeat the "
    "necromancer. Rewards are the upstream Craftax rewards: newly "
    "unlocked achievements give positive reward, and changes in health "
    "also affect reward. The episode terminates on death, on defeating "
    "the boss, or at the turn limit.\n\n"
    f"{_MECHANICS}\n\n"
    "OBSERVATION\n"
    "The grid comes from the fork's stable Unicode local renderer. "
    "The view is the 9x11 visible window centered on you; the "
    "HUD carries inventory, equipment, vitals, floor, absolute position, "
    "facing, spell knowledge, ladder progress, and achievements. "
    "Directional arrows mark your current facing. To interact with a "
    "tree, ore, mob, water, chest, fountain, or edible passive creature, "
    "face that cell and use DO. Absolute position is included because "
    "Craftax is long-horizon and navigation-relevant.\n\n"
    "CORE MECHANICS\n"
    "Food, drink, energy, mana, and HP are persistent resources. Food, "
    "drink, and energy decay over time; HP can recover when food, drink, "
    "and energy are positive. Mine trees for wood, place a table, craft "
    "pickaxes and swords, mine stone and ores, place a furnace, then "
    "progress to stronger equipment. Underground floors may need "
    "torches. Kill enough monsters on a floor to open its ladder, then "
    "stand on the ladder and DESCEND. Use ASCEND on upward ladders. "
    "Books teach fireball or iceball. Potions have color-shuffled "
    "effects per run.\n\n"
    "MOVEMENT\n"
    "MOVE_LEFT, MOVE_RIGHT, MOVE_UP, and MOVE_DOWN are absolute moves. "
    "If the target cell is blocked, the move still turns you to face "
    "that direction. DO always applies to the cell you face, not "
    "necessarily the cell you occupy.\n\n" + _actions_block()
)

OBSERVATION_CONVENTIONS = (
    "OBSERVATION CONVENTIONS\n"
    "Each turn you receive a [Grid] (Unicode text) and a [Legend] mapping each "
    "glyph to its meaning for this turn. You may also receive a [HUD] "
    "block with complementary state that cannot be read from the grid "
    "(for example turn budget, HP, inventory, score, velocity, cooldowns, "
    "or facing only when the glyph itself is not directional), plus "
    "optionally a [Message] block with one-shot env feedback (last action "
    "result, etc.). Treat the "
    "[Grid] as authoritative for spatial state and the [Legend] as "
    "authoritative for glyph meaning; do not expect the HUD to repeat "
    "positions, visible enemy locations, or facing already encoded by a "
    "directional glyph. The full action list is in this system prompt "
    "and does not repeat per turn."
)

_REASONING_FORMAT = (
    "RESPONSE FORMAT\n"
    "Reason before acting. {budget} Inspect the observation, "
    "track relevant state, compare plausible moves, and choose a plan before "
    "committing to an action. Always preserve room for the closing action tag. "
    "For models without a native thinking "
    "mode, write the reasoning in plain assistant text before the action. For models with "
    "a native thinking mode, use that mode normally. "
    "Your reasoning will be stored and replayed as part of the conversation on later turns.\n"
    "\n"
    "After reasoning, end your response with exactly one final action tag on "
    "its own line. Put the exact chosen action name between the tags; for "
    "example, if MOVE_FORWARD is listed and you choose it, write:\n"
    "  <action>MOVE_FORWARD</action>\n"
    "\n"
    "The action name must be one of the names in the [Actions] list. Do not emit "
    "an <action> tag until the final line; if you discuss candidate actions "
    "during reasoning, write their names as plain text or in backticks instead "
    "of XML tags. Read the grid directly — when the player glyph is "
    "directional, it shows your orientation. The evaluator parses the final "
    "complete <action> tag as your committed move and stores the preceding "
    "reasoning in the conversation transcript.\n"
    "\n"
    "Failure modes: {truncation}If the final <action> tag is missing or "
    "contains an unknown name, the turn is also forfeited the same way."
)

_ACTION_ONLY_FORMAT = (
    "RESPONSE FORMAT\n"
    "Valid action names for this task are: {valid}.\n"
    "Reply with exactly one XML action tag and no other text:\n"
    "  <action>NOOP</action>\n"
    "\n"
    "Replace NOOP with the exact action name you choose. "
    "Do not include reasoning, explanations, markdown, code fences, extra XML, "
    "or text before or after the action tag. The evaluator parses the final "
    "complete <action> tag as your committed move.\n"
    "\n"
    "{budget}If the final <action> tag is missing "
    "or contains an unknown name, the turn is also forfeited the same way."
)

_FORFEIT = {
    "freeze": "env state unchanged, turn counter still advances, reward 0",
    "noop": (
        "the env applies its NOOP/fallback action, dynamics advance "
        "normally, and reward comes from that fallback step"
    ),
}


_TERMINATION = (
    "The episode terminates on death, on defeating "
    "the boss, or at the turn limit."
)
_CONTINUOUS = (
    "Death does not end the episode: you respawn on the Overworld near the "
    "spawn point with full health, food, drink, energy and mana, and unlocked "
    "achievements stay unlocked. There is no turn limit; the episode ends on "
    "defeating the boss or when the session's time runs out."
)


def system_prompt(reasoning: bool, max_output_tokens: int | None, forfeit_mode: str, respawn: bool = False) -> str:
    forfeit = _FORFEIT[forfeit_mode]
    if reasoning:
        if max_output_tokens is None:
            budget = (
                "No explicit output-token cap is imposed: use the model's native "
                "reasoning budget and take as much reasoning as is useful."
            )
            truncation = ""
        else:
            budget = f"You may use up to {max_output_tokens} output tokens total for reasoning plus the final action."
            truncation = (
                f"If your full response exceeds {max_output_tokens} tokens before "
                f"the closing </action> tag, the action is discarded and the turn is forfeited ({forfeit}). "
            )
        fmt = _REASONING_FORMAT.format(budget=budget, truncation=truncation)
    else:
        if max_output_tokens is None:
            budget = "No explicit output-token cap is imposed, but a valid response is only the short action tag. "
        else:
            budget = (
                f"The output budget is {max_output_tokens} tokens, but a valid "
                "response should be only the short action tag. If the full response "
                f"exceeds {max_output_tokens} tokens before the closing </action> "
                f"tag, the action is discarded and the turn is forfeited ({forfeit}). "
            )
        fmt = _ACTION_ONLY_FORMAT.format(valid=", ".join(ACTION_NAMES), budget=budget)
    game = GAME_PROMPT.replace(_TERMINATION, _CONTINUOUS) if respawn else GAME_PROMPT
    return "\n\n---\n".join([game, OBSERVATION_CONVENTIONS, fmt])


# --- action parsing (GlyphBench's parse_action_response + ActionSpec.index_of) ---

_XML_ACTION = re.compile(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_NOOP_FAMILY = ("NOOP", "WAIT", "DONE", "PASS", "SKIP")


def action_index(name: str) -> int | None:
    stripped = name.strip()
    if stripped in ACTION_NAMES:
        return ACTION_NAMES.index(stripped)
    folded = stripped.casefold()
    matches = [i for i, n in enumerate(ACTION_NAMES) if n.casefold() == folded]
    if len(matches) == 1:
        return matches[0]
    target = api.ACTION_ALIASES.get(stripped) or api.ACTION_ALIASES.get(stripped.upper())
    if target:
        return ACTION_NAMES.index(target)
    if folded.upper() in _NOOP_FAMILY:
        return NOOP
    return None


def parse_action(reply: str) -> tuple[int | None, str | None]:
    """The last complete <action> tag after the last </think>: (index, None), or
    (None, "no_action_tag" / "unknown_name")."""
    text = reply or ""
    closes = list(_THINK_CLOSE.finditer(text))
    region = text[closes[-1].end() :] if closes else text
    found = _XML_ACTION.findall(region)
    if not found or not found[-1].strip():
        return None, "no_action_tag"
    index = action_index(found[-1])
    return (index, None) if index is not None else (None, "unknown_name")


# --- the game (GlyphBench's CraftaxForkEnv) ---


@dataclass
class Step:
    reward: float
    done: bool
    unlocked: list[str]
    fast_forward_ticks: int


class CraftaxGame:
    """Full Craftax from the pinned fork, with GlyphBench's HUD, message and
    sleep/rest fast-forward. ``turn`` counts model turns (forfeits included);
    the game ends at ``max_turns`` (None: never), on death or on the boss's defeat,
    unless a death is undone with ``respawn``."""

    def __init__(self, seed: int, max_turns: int | None = 10000) -> None:
        self.max_turns = int(max_turns) if max_turns else None
        self._env, self._params, self._rt = _load(self.max_turns or ENDLESS_TIMESTEPS)
        jax = self._rt["jax"]
        self._key = jax.random.PRNGKey(int(seed))
        self._key, reset_key = jax.random.split(self._key)
        _obs, self.state = self._env.reset(reset_key, self._params)
        self.turn = 0
        self.message = ""
        self.achievements = self._achievement_names()
        self.spawn = self.position

    # state accessors
    @property
    def health(self) -> float:
        return float(np.asarray(self.state.player_health))

    @property
    def floor(self) -> int:
        return int(np.asarray(self.state.player_level))

    @property
    def timestep(self) -> int:
        """The game's own clock: model turns plus fast-forwarded sleep/rest ticks."""
        return int(np.asarray(self.state.timestep))

    @property
    def position(self) -> tuple[int, int]:
        r, c = np.asarray(self.state.player_position).tolist()
        return int(r), int(c)

    @property
    def out_of_turns(self) -> bool:
        return self.max_turns is not None and max(self.turn, self.timestep) >= self.max_turns

    @property
    def achievement_score(self) -> float:
        """Craftax's reward for the unlocked achievements alone (no health term), out of 226."""
        got = np.asarray(self.state.achievements, dtype=bool)
        return float(self._rt["weights"][got].sum())

    def respawn(self, death_inventory: str) -> tuple[int, int]:
        """Undo a death: back on the Overworld at the free cell nearest the world's spawn
        point, vitals at their maximum, sleep/rest and the vitals' counters reset, the
        inventory per ``death_inventory`` ("drop_all": everything, enchantments included;
        "drop_materials": materials and consumables, tools and armour kept; "keep").
        Achievements, XP, attributes, learned spells and the world itself stay."""
        jnp = self._rt["jax"].numpy
        st, U = self.state, self._rt["max_vitals"]
        at = self._free_cell(self.spawn)
        inventory, extra = st.inventory, {}
        if death_inventory == "drop_all":
            inventory = self._rt["empty_inventory"]()
            extra = {
                "sword_enchantment": jnp.zeros_like(st.sword_enchantment),
                "bow_enchantment": jnp.zeros_like(st.bow_enchantment),
                "armour_enchantments": jnp.zeros_like(st.armour_enchantments),
            }
        elif death_inventory == "drop_materials":
            zero = {k: jnp.zeros_like(getattr(st.inventory, k)) for k in (
                "wood", "stone", "coal", "iron", "diamond", "sapling", "ruby", "sapphire",
                "arrows", "torches", "books", "potions",
            )}
            inventory = st.inventory.replace(**zero)
        self.state = st.replace(
            inventory=inventory,
            player_level=jnp.asarray(0, dtype=jnp.int32),
            player_position=jnp.asarray(at, dtype=st.player_position.dtype),
            player_health=jnp.asarray(U.get_max_health(st), dtype=jnp.float32),
            player_food=jnp.asarray(U.get_max_food(st), dtype=jnp.int32),
            player_drink=jnp.asarray(U.get_max_drink(st), dtype=jnp.int32),
            player_energy=jnp.asarray(U.get_max_energy(st), dtype=jnp.int32),
            player_mana=jnp.asarray(U.get_max_mana(st), dtype=jnp.int32),
            player_recover=jnp.asarray(0.0, dtype=jnp.float32),
            player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
            player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
            player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
            player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
            is_sleeping=False,
            is_resting=False,
            **extra,
        )
        return at

    def _free_cell(self, target: tuple[int, int], radius: int = 16) -> tuple[int, int]:
        """The walkable, mob-free Overworld cell nearest ``target`` (ring by ring)."""
        level = np.asarray(self.state.map[0])
        mobs = np.asarray(self.state.mob_map[0])
        blocked = self._rt["not_walkable"][level] | mobs
        r0, c0 = target
        for d in range(radius + 1):
            for r in range(r0 - d, r0 + d + 1):
                for c in range(c0 - d, c0 + d + 1):
                    if max(abs(r - r0), abs(c - c0)) != d:
                        continue
                    if 0 <= r < level.shape[0] and 0 <= c < level.shape[1] and not blocked[r, c]:
                        return r, c
        return target

    def _achievement_names(self) -> set[str]:
        got = np.asarray(self.state.achievements)
        return {a.name.lower() for a in self._rt["achievements"] if bool(got[int(a.value)])}

    def _step_upstream(self, value: int) -> tuple[float, bool]:
        self._key, step_key = self._rt["jax"].random.split(self._key)
        _obs, self.state, reward, done, _info = self._env.step(step_key, self.state, value, self._params)
        return float(np.asarray(reward)), bool(np.asarray(done))

    def _resting(self) -> bool:
        return bool(np.asarray(self.state.is_sleeping)) or bool(np.asarray(self.state.is_resting))

    def step(self, action: int) -> Step:
        self.turn += 1
        values = self._rt["values"]
        before = self.achievements
        reward, done = self._step_upstream(values[action])
        ticks, capped = 0, False
        # A sleep or rest runs to its end in one turn (GlyphBench): NOOP ticks until it stops.
        while not done and self._resting():
            if ticks >= FAST_FORWARD_MAX_TICKS:
                capped = True
                break
            extra, done = self._step_upstream(values[NOOP])
            reward += extra
            ticks += 1
        self.achievements = self._achievement_names()
        unlocked = sorted(self.achievements - before)
        note = f" Fast-forwarded {ticks} sleep/rest tick{'s' if ticks != 1 else ''}." if ticks else ""
        if capped:
            note += " Fast-forward cap reached; state may still be asleep/resting."
        name, upstream = ACTION_NAMES[action], api.UPSTREAM_ACTION_NAMES[action]
        self.message = f"Last action {name} ({upstream}) yielded reward {reward:+.3f}.{note}"
        if unlocked:
            self.message += f" New achievements: {', '.join(unlocked)}."
        return Step(reward, done or self.out_of_turns, unlocked, ticks)

    def forfeit(self) -> bool:
        """GlyphBench's "freeze": the turn counts, the world does not move. True if
        that was the last turn."""
        self.turn += 1
        return self.out_of_turns

    def render(self) -> tuple[str, str, str, str]:
        """(legend, grid, hud, message) of the current state."""
        rendered = api.render_craftax_unicode_grid_local(self.state)
        return rendered.legend, rendered.grid, self._hud(np.asarray(rendered.status_data)), self.message

    def _hud(self, sd: np.ndarray) -> str:
        R = self._rt["renderer"]
        floor = int(sd[R.S_FLOOR])
        floor_name = FLOOR_NAMES[floor] if 0 <= floor < len(FLOOR_NAMES) else f"Floor {floor}"
        facing = {0: "none", 1: "left", 2: "right", 3: "up", 4: "down"}.get(int(sd[R.S_DIRECTION]), "none")
        light = float(sd[R.S_LIGHT_LEVEL])
        light_name = "Night" if light < 0.3 else "Dusk" if light < 0.6 else "Day"
        flags = [
            f"Sleeping: {'yes' if sd[R.S_IS_SLEEPING] > 0.5 else 'no'}",
            f"Resting: {'yes' if sd[R.S_IS_RESTING] > 0.5 else 'no'}",
            f"Fireball: {'learned' if sd[R.S_LEARNED_FIRE] > 0.5 else 'no'}",
            f"Iceball: {'learned' if sd[R.S_LEARNED_ICE] > 0.5 else 'no'}",
        ]
        kills, required = int(sd[R.S_MONSTERS_KILLED]), int(sd[R.S_MONSTERS_REQUIRED])
        ladder = "OPEN" if kills >= required else f"CLOSED {kills}/{required}"
        boss = "VULNERABLE" if sd[R.S_BOSS_VULNERABLE] > 0.5 else "no"
        inventory = " ".join(
            f"{label}:{int(sd[getattr(R, f'S_INV_{label.upper()}')])}"
            for label in ("Wood", "Stone", "Coal", "Iron", "Diamond", "Sapphire", "Ruby", "Sapling")
        )
        potions = " ".join(
            f"{label}:{int(sd[R.S_POTION_START + i])}"
            for i, label in enumerate(("Red", "Green", "Blue", "Pink", "Cyan", "Yellow"))
        )
        material = {0: "None", 1: "Wood", 2: "Stone", 3: "Iron", 4: "Diamond"}
        enchant = {0: "None", 1: "Fire", 2: "Ice"}
        sword_lvl = int(sd[R.S_INV_SWORD])
        sword = f"{material.get(sword_lvl, 'None')} Sword" if sword_lvl > 0 else "No Sword"
        if sword_lvl > 0 and int(sd[R.S_SWORD_ENCH]) > 0:
            sword += f"({enchant.get(int(sd[R.S_SWORD_ENCH]), 'None')})"
        bow = "Bow" if int(sd[R.S_INV_BOW]) > 0 else "No Bow"
        if bow == "Bow" and int(sd[R.S_BOW_ENCH]) > 0:
            bow += f"({enchant.get(int(sd[R.S_BOW_ENCH]), 'None')})"
        equipment = " | ".join(
            [
                f"{material.get(int(sd[R.S_INV_PICKAXE]), 'None')} Pickaxe",
                sword,
                bow,
                f"Arrows:{int(sd[R.S_INV_ARROWS])}",
                f"Torches:{int(sd[R.S_INV_TORCHES])}",
                f"Books:{int(sd[R.S_INV_BOOKS])}",
            ]
        )
        armor_parts = []
        for i, slot in enumerate(("Helm", "Chest", "Legs", "Boots")):
            level, ench = int(sd[R.S_ARMOR_START + i]), int(sd[R.S_ARMOR_ENCH_START + i])
            if level <= 0:
                continue
            part = f"{ {1: 'Iron', 2: 'Diamond'}.get(level, 'None')} {slot}"
            if ench > 0:
                part += f"({enchant.get(ench, 'None')})"
            armor_parts.append(part)
        return "\n".join(
            [
                (f"Step: {self.turn} / {self.max_turns}" if self.max_turns else f"Step: {self.turn}")
                + f"    Upstream timestep: {int(sd[R.S_TIMESTEP])}",
                f"HP: {int(sd[R.S_HEALTH])}/{int(sd[R.S_MAX_HEALTH])}    "
                f"Food: {int(sd[R.S_FOOD])}/{int(sd[R.S_MAX_FOOD])}    "
                f"Drink: {int(sd[R.S_DRINK])}/{int(sd[R.S_MAX_DRINK])}    "
                f"Energy: {int(sd[R.S_ENERGY])}/{int(sd[R.S_MAX_ENERGY])}    "
                f"Mana: {int(sd[R.S_MANA])}/{int(sd[R.S_MAX_MANA])}",
                f"XP: {int(sd[R.S_XP])}    Str: {int(sd[R.S_STR])}    Dex: {int(sd[R.S_DEX])}    Int: {int(sd[R.S_INT])}",
                f"Floor: {floor} ({floor_name})    Pos: ({int(sd[R.S_POS_ROW])},{int(sd[R.S_POS_COL])})    "
                f"Facing: {facing}    Light: {light_name}",
                "[Inventory] " + inventory,
                "[Equipment] " + equipment,
                "[Armor] " + (" | ".join(armor_parts) if armor_parts else "None"),
                "[Potions] " + potions,
                "[Status] " + " | ".join(flags) + f" | Ladder: {ladder} | Boss: {boss} | "
                f"Achievements: {len(self.achievements)}/{len(self._rt['achievements'])}",
            ]
        )


# --- the per-turn user message (GlyphBench's render_user_turn, no history frames) ---


def user_turn(game: CraftaxGame, reasoning: bool) -> str:
    legend, grid, hud, message = game.render()
    parts = []
    if legend:
        parts.append(f"[Legend]\n{legend}")
    current = [f"[Current Observation — turn {game.turn}]", f"[Grid]\n{grid.rstrip()}"]
    if hud:
        current.append(f"[HUD]\n{hud.strip()}")
    if message:
        current.append(f"[Message]\n{message.strip()}")
    parts.append("\n".join(current))
    if reasoning:
        parts.append(
            "Now reason as much as useful, then end with a final <action> tag containing "
            "exactly one action name from the system prompt's action list."
        )
    else:
        parts.append(
            "Reply now with exactly one <action>...</action> tag containing "
            "one action name from the system prompt's action list, and no "
            "other text."
        )
    return "\n\n".join(parts)

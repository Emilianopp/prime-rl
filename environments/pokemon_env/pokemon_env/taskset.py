"""pokemon-env — Pokémon FireRed as text, one episode per rollout.

The real game runs headless in mGBA. Every observation is plain text read from
the game's memory: status (location, party, bag, badges), the decoded screen text
(dialogue, menus, battle menus with their cursors), battle state, and a map of the
current area with doors labeled by where they lead. No frames are produced.

The env plays the user: the model's last ``<buttons>`` line is pressed, the game
runs until it waits for input again, and the new observation is the next user
turn, until ``max_episode_steps`` model turns or the trace's token budget.
Reward = story milestones reached / all milestones (towns, gyms, badges, …).

Needs your own FireRed US v1.0 ROM (``rom_path``), a start state made once with
``python -m pokemon_env.make_start_state``, and libmgba next to the ``mgba``
bindings (``scripts/build_libmgba.sh``); see the README.
"""

import asyncio
import json
import os
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

SYSTEM_PROMPT = """You are playing Pokémon FireRed (Game Boy Advance) through a text-only interface. You never see pixels; each turn you get a text observation read directly from the game's memory.

GOAL: make as much story progress as possible: get a starter Pokémon from Professor Oak, travel through Viridian City and Viridian Forest, beat the gyms, and become Champion. Progress is scored by story milestones (reaching new towns, routes and gyms, earning badges).

CONTROLS — reply with buttons, pressed in order:
  up / down / left / right : walk one step (one map cell); also move menu cursors
  a      : talk / interact with what you face, confirm, advance dialogue
  b      : cancel / back out of menus
  start  : open the main menu        select / l / r : rarely needed (l and r open help)
To talk to someone or read a sign, stand next to it, face it (press its direction once), then press a.

OBSERVATION:
  STATUS: location, your (x,y) position (x grows right, y grows down), facing, party, bag.
  SCREEN TEXT: dialogue and menus as on screen; ▶ marks the menu cursor, ▼ means press a to continue.
    Battle menus and the party screen show their cursor as a "▶ cursor on: ..." line.
  MESSAGES SINCE YOUR LAST ACTION: battle messages that scrolled by while your buttons played out.
  MAP: grid of the current map, one cell per step. Column numbers on top, row numbers on the left. @ is you.
    Doors are labeled with a character; POINTS OF INTEREST says where each door leads, how far it is
    from you and how to use it (stairs and exit mats: step on the tile, then press the direction shown).
    Map edge entries tell you which neighbouring map you reach by walking off that edge.
  On a name-entry keyboard, press start then a to accept the suggested name.

REPLY FORMAT: think briefly, then end with exactly one line:
<buttons>up up right a</buttons>
Use 1 to {max_buttons} buttons. The game state is shown again after all of them are pressed."""

_BUTTONS = re.compile(r"<buttons>(.*?)</buttons>", re.DOTALL | re.IGNORECASE)


def _parse_buttons(reply: str, valid: tuple[str, ...], limit: int) -> list[str]:
    found = _BUTTONS.findall(reply or "")
    if not found:
        return []
    return [t for t in re.split(r"[\s,;]+", found[-1].strip().lower()) if t in valid][:limit]


def _start_states(path: str) -> list[str]:
    p = Path(path)
    states = sorted(str(s) for s in p.glob("*.state")) if p.is_dir() else [str(p)]
    if not states or not all(Path(s).exists() for s in states):
        raise FileNotFoundError(
            f"no start state at {path!r}; make one with `python -m pokemon_env.make_start_state`"
        )
    return states


class PokemonData(vf.TaskData):
    info: dict
    """ROM, start state, RNG offset and observation options for the episode."""


class PokemonTask(vf.Task[PokemonData]):
    pass


class PokemonConfig(vf.TasksetConfig):
    rom_path: str = Field(default_factory=lambda: os.environ.get("POKEMON_FIRERED_ROM", ""))
    """Your Pokémon FireRed US v1.0 ROM (sha1 41cb23d8…); defaults to $POKEMON_FIRERED_ROM."""
    start_state: str = Field(default_factory=lambda: os.environ.get("POKEMON_FIRERED_STATE", ""))
    """A save state, or a directory of `*.state` files used round-robin; defaults to $POKEMON_FIRERED_STATE."""
    split: Literal["train", "eval"] = "train"
    num_examples: int = Field(1024, ge=1)
    rng_frames: int = Field(240, ge=1)
    """Task i idles ``i % rng_frames`` frames after loading its state (eval: offset by half), so
    tasks sharing a start state get different RNG (encounters, damage rolls)."""
    info_level: Literal["raw", "semantic", "oracle"] = "semantic"
    """raw: map glyphs only; semantic: + labeled doors, exits, points of interest; oracle: + hidden
    items and the opponent's exact HP and moves."""
    view: Literal["full", "local"] = "full"
    """Map of the whole current area, or a window of ``radius`` cells around the player."""
    radius: int = Field(7, ge=1)
    max_buttons: int = Field(6, ge=1)
    """Buttons pressed per model turn at most."""


class PokemonEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "inline-chat"})
    max_episode_steps: int = Field(1000, ge=1)
    """Model turns before the episode is cut off."""
    settle_max: int = Field(300, ge=10)
    """Idle frames to wait for the game to accept input after a button (5x while a cutscene runs)."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, reward, milestones) to this file."""


def _new_game(info: dict, settle_max: int):
    from pokemon_env.firered.emulator import FireRedGame

    game = FireRedGame(
        rom_path=info["rom_path"],
        state_path=info["start_state"],
        info_level=info["info_level"],
        view=info["view"],
        radius=info["radius"],
        settle_max=settle_max,
    )
    return game, game.reset(idle_frames=info["idle_frames"])


class PokemonEnv(vf.Env[PokemonEnvConfig]):
    async def run(self, task, agents):
        from pokemon_env.firered.emulator import BUTTONS, MILESTONES

        info = task.data.info
        max_steps = self.config.max_episode_steps
        started = time.time()
        game, observation = await asyncio.to_thread(_new_game, info, self.config.settle_max)
        steps, malformed, frames = 0, 0, 0
        pressed: Counter[str] = Counter()
        end = "terminated"  # the run stopped answering (a turn or token limit)
        try:
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn(f"Turn 1 of {max_steps}.\n\n{observation}")
                while not segment.terminated:
                    steps += 1
                    if steps >= max_steps:
                        end = "max_steps"
                        break
                    buttons = _parse_buttons(segment.last_reply, BUTTONS, info["max_buttons"])
                    loc_before, xy_before = game.state.location_name(), game.state.player_xy()
                    observation, step = await asyncio.to_thread(game.step, buttons)
                    frames += step.frames
                    pressed.update(buttons)
                    if not buttons:
                        malformed += 1
                        feedback = "Your reply had no valid <buttons>...</buttons> line; nothing was pressed."
                    elif step.location == loc_before:
                        feedback = f"Pressed {' '.join(buttons)}; moved {xy_before} → {step.coords}."
                    else:
                        feedback = f"Pressed {' '.join(buttons)}; entered {step.location} at {step.coords}."
                    if step.new_milestones:
                        feedback += f" New milestone: {', '.join(step.new_milestones)}."
                    segment = await interaction.turn(
                        f"Turn {steps + 1} of {max_steps}.\n=== LAST ACTION ===\n{feedback}\n\n{observation}"
                    )
        finally:
            party = await asyncio.to_thread(game.state.party)  # before close: memory views die with the core
            await asyncio.to_thread(game.close)
        trace = interaction.trace
        reached = dict(game.milestones)
        reward = len(reached) / len(MILESTONES)
        trace.record_reward("milestones", reward, 1.0)
        trace.record_metrics(
            {
                "milestones": float(len(reached)),
                "badges": float(sum(1 for m in reached if m.endswith("_BADGE"))),
                "malformed_replies": float(malformed),
                "emulated_frames": float(frames),
            }
        )
        outcome = {
            "milestones": reached,
            "steps": steps,
            "ended_by": end,
            "start_state": info["start_state"],
            "idle_frames": info["idle_frames"],
        }
        trace.info["pokemon"] = outcome
        if self.config.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "reward": reward,
                "malformed_replies": malformed,
                "buttons": dict(pressed),
                "emulated_frames": frames,
                "trace_tokens": trace.num_total_tokens,
                "party": party,
                "task": task.data.name,
                **outcome,
            }
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class PokemonTaskset(vf.Taskset[PokemonTask, PokemonConfig]):
    def load(self) -> Iterator[PokemonTask]:
        cfg = self.config
        if not cfg.rom_path or not Path(cfg.rom_path).exists():
            raise FileNotFoundError(
                f"FireRed ROM not found at {cfg.rom_path!r}: set taskset.rom_path or $POKEMON_FIRERED_ROM"
            )
        states = _start_states(cfg.start_state)
        offset = cfg.rng_frames // 2 if cfg.split == "eval" else 0
        system_prompt = SYSTEM_PROMPT.format(max_buttons=cfg.max_buttons)
        for i in range(cfg.num_examples):
            yield PokemonTask(
                PokemonData(
                    name=f"firered#{cfg.split}{i}",
                    prompt=None,  # prompt-less: the env opens the episode with the first observation
                    system_prompt=system_prompt,
                    info={
                        "rom_path": cfg.rom_path,
                        "start_state": states[i % len(states)],
                        "idle_frames": (i + offset) % cfg.rng_frames,
                        "info_level": cfg.info_level,
                        "view": cfg.view,
                        "radius": cfg.radius,
                        "max_buttons": cfg.max_buttons,
                        "episode_idx": i,
                        "split": cfg.split,
                    },
                ),
                cfg.task,
            )

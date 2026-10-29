"""textworld-env — pre-generated TextWorld games, one episode per rollout.

The env plays the user: the model's last ``<action>`` is stepped in the game and
the observation comes back as the next user turn, until the game ends or
``max_episode_steps`` model turns have been taken. The reward is the final game
score over the game's max score.

A port of the kv-eviction TextWorld env (verifiers v0 ``MultiTurnEnv``) with the
same prompt, action parsing and reward, so results are comparable with the
earlier concurrency grid. Dataset layout: ``metadata.json`` (``game_files``,
``max_scores``), ``dataset/`` and ``eval_dataset/`` (HF ``save_to_disk`` rows of
``question`` = initial observation, ``answer`` = game index, ``task`` = family),
and the ``games/`` they point to.
"""

import asyncio
import json
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

SYSTEM_PROMPT = """You are playing a TextWorld interactive-fiction game. Infer the objective from the opening instructions and observations, then maximize the score until the game ends.

POSSIBLE OBJECTIVES:
- Find and take a coin or a requested treasure.
- Explore rooms, unlock or open passages, and manipulate household objects as instructed.
- Read a cookbook, gather and prepare ingredients, prepare the meal, and eat it.

GENERAL COMMANDS (use exact object names and valid game syntax):
- look; inventory; examine [object]
- go north/south/east/west
- open/close [object]; unlock/lock [object] with [key]
- take [object]; take [object] from [container]; drop [object]
- put [object] on [supporter]; insert [object] into [container]
- eat [object]

COOKING COMMANDS WHEN A RECIPE IS PRESENT:
- examine cookbook
- slice/dice/chop [ingredient] with knife
- cook [ingredient] with oven/stove/BBQ
- prepare meal; eat meal

IMPORTANT RULES:
- Follow the stated objective. If it is hidden or incomplete, explore and use score changes to infer progress.
- Use exact names from observations. Examine or inventory objects when uncertain.
- Track visited rooms, object locations, inventory, locked doors, and completed steps.
- Do not repeat an invalid action without changing your approach.
- For cooking, read the recipe first, match preparation and appliance exactly, and do not cook an ingredient twice.

Respond with your reasoning, then exactly one game command in this format:
<action>your command here</action>"""

_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)

# textworld.start() goes through tatsu's module-level parser, which is not thread
# safe: serialize starts and give each one a fresh parser.
_START_LOCK = threading.Lock()


def _start_game(game_file: str):
    import textworld

    infos = textworld.EnvInfos(score=True, max_score=True, won=True, description=True, inventory=True)
    with _START_LOCK:
        try:
            import textworld.logic as twl
            from textworld.logic.parser import GameLogicParser

            twl._PARSER = GameLogicParser()
        except (ImportError, AttributeError):
            pass
        game = textworld.start(game_file, infos)
        game.reset()
    return game


def _step(game, action: str) -> tuple[str, int | None, bool, bool]:
    state, score, done = game.step(action)
    return state.feedback, score, done, bool(getattr(state, "won", False))


def _resolve_game_file(dataset: Path, raw: str) -> str:
    """Relative to the dataset dir, absolute if it still exists, else rebased under games/."""
    path = Path(raw)
    if not path.is_absolute() and (dataset / path).exists():
        return str(dataset / path)
    if path.is_absolute() and path.exists():
        return str(path)
    if (dataset / "games" / path.name).exists():
        return str(dataset / "games" / path.name)
    raise FileNotFoundError(f"could not resolve game file {raw!r} under {dataset}")


class TextWorldData(vf.TaskData):
    info: dict
    """The game file and its max score."""


class TextWorldTask(vf.Task[TextWorldData]):
    pass


class TextWorldConfig(vf.TasksetConfig):
    dataset_path: str
    """Directory with metadata.json, dataset/, eval_dataset/ and games/."""
    split: Literal["train", "eval"] = "train"
    num_examples: int | None = None
    """First N rows of the split (all by default)."""
    families: list[str] | None = None
    """Only games of these families, e.g. ["tw-coin_collector"] (all by default)."""
    include_initial_obs_in_system: bool = True
    """Put the opening observation in the system prompt (the user turn is then "What do you do?")."""


class TextWorldEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    max_episode_steps: int = Field(200, ge=1)
    """Model turns (game actions) before the episode is cut off."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, reward, steps) to this file."""


class TextWorldEnv(vf.Env[TextWorldEnvConfig]):
    async def run(self, task, agents):
        info = task.data.info
        started = time.time()
        game = await asyncio.to_thread(_start_game, info["game_file"])
        score, won, steps, done = 0, False, 0, False
        action = "look"  # the previous action stands in for a reply without one
        try:
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn()
                while not segment.terminated:
                    steps += 1
                    if steps >= self.config.max_episode_steps:
                        break  # as in v0: the capped turn's action is not played
                    found = _ACTION.findall(segment.last_reply)
                    if found:
                        action = found[0]
                    try:
                        observation, new_score, done, won = await asyncio.to_thread(_step, game, action)
                        score = score if new_score is None else new_score
                    except Exception:
                        observation = "Something went wrong. Try a different action."
                    if done:
                        break
                    segment = await interaction.turn(observation)
        finally:
            await asyncio.to_thread(game.close)
        trace = interaction.trace
        reward = float(score) / max(float(info["max_score"]), 1.0)
        trace.record_reward("game_score", reward, 1.0)
        outcome = {"score": score, "max_score": info["max_score"], "won": won, "steps": steps, "done": done}
        trace.info["textworld"] = outcome
        if self.config.episode_log:
            line = {"start": started, "end": time.time(), "reward": reward, "task": info["task"], "game": info["game_idx"], **outcome}
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class TextWorldTaskset(vf.Taskset[TextWorldTask, TextWorldConfig]):
    def load(self) -> Iterator[TextWorldTask]:
        from datasets import load_from_disk

        dataset = Path(self.config.dataset_path)
        meta = json.loads((dataset / "metadata.json").read_text())
        rows = load_from_disk(str(dataset / ("dataset" if self.config.split == "train" else "eval_dataset")))
        if self.config.families is not None:
            rows = rows.filter(lambda row: row["task"].split(":")[0] in self.config.families)
        if self.config.num_examples is not None:
            rows = rows.select(range(min(self.config.num_examples, len(rows))))
        for i, row in enumerate(rows):
            game_idx = int(row["answer"])
            observation = row["question"]
            if self.config.include_initial_obs_in_system:
                system_prompt = f"{SYSTEM_PROMPT}\n\n---\nINITIAL OBSERVATION:\n{observation}"
                prompt = "What do you do?"
            else:
                system_prompt, prompt = SYSTEM_PROMPT, observation
            yield TextWorldTask(
                TextWorldData(
                    name=f"{row['task']}#{self.config.split}{i}",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    info={
                        "game_file": _resolve_game_file(dataset, meta["game_files"][game_idx]),
                        "max_score": meta["max_scores"][game_idx],
                        "game_idx": game_idx,
                        "task": row["task"],
                    },
                ),
                self.config.task,
            )

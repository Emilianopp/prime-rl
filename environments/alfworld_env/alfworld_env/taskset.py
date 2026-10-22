"""alfworld-env — ALFRED household tasks (ALFWorld's TextWorld builds), one game per rollout.

A port of the kv-eviction ALFWorld env (verifiers v0 ``MultiTurnEnv``) with the
same prompt, action parsing, admissible-commands regime and reward: 1 if the
task is completed, else 0 (ALFRED has no partial scores). The game's opening
observation, which states the task, goes in the system prompt, as in v0: under
KV-Streams only the system prompt is protected from eviction, and an agent that
loses its task cannot finish it.

The tw-pddl engine is pure Python and not thread safe, so every engine call in a
process goes through one worker thread (a step takes milliseconds).

Dataset layout (``alfworld-download``): ``<dataset_path>/json_2.1.1/<split>/*/*/game.tw-pddl``.
"""

import asyncio
import json
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

SYSTEM_PROMPT = """You are an agent in a household environment (ALFRED). \
You interact with the world through text commands to accomplish a task.

Common commands: go to <receptacle>, open <receptacle>, close <receptacle>, \
take <object> from <receptacle>, put <object> in/on <receptacle>, \
use <object>, heat <object> with <receptacle>, cool <object> with <receptacle>, \
clean <object> with <receptacle>, examine <object>, inventory, look.

Pay attention to the task given in the initial observation and work toward \
completing it. Track what you have already tried; do not repeat an invalid \
action without changing your approach.

Respond with your reasoning, then exactly one game command in this format:
<action>your command here</action>"""

_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_ENGINE = ThreadPoolExecutor(max_workers=1)


def _start_game(game_file: str, admissible: bool):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    infos = textworld.EnvInfos(won=True, admissible_commands=admissible)
    game = textworld.start(game_file, infos, wrappers=[AlfredDemangler()])
    return game, game.reset()


def _observation(state, admissible: bool) -> str:
    commands = (getattr(state, "admissible_commands", None) or []) if admissible else []
    if not commands:
        return state.feedback
    return state.feedback + "\n\nAdmissible commands:\n" + "\n".join(f"- {c}" for c in commands)


async def _engine(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_ENGINE, fn, *args)


class AlfWorldData(vf.TaskData):
    info: dict
    """The game file, its task type and split."""


class AlfWorldTask(vf.Task[AlfWorldData]):
    pass


class AlfWorldConfig(vf.TasksetConfig):
    dataset_path: str
    """The ALFWorld data directory (with json_2.1.1/)."""
    split: Literal["train", "valid_seen", "valid_unseen"] = "valid_seen"
    num_examples: int | None = None
    """First N games of the split, in path order (all by default)."""


class AlfWorldEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    max_episode_steps: int = Field(150, ge=1)
    """Model turns (game actions) before the episode is cut off."""
    admissible_commands: bool = True
    """List the admissible commands after every observation (base-model success goes
    from ~1-3% to ~35-45% with them in v0)."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, reward, steps) to this file."""


class AlfWorldEnv(vf.Env[AlfWorldEnvConfig]):
    async def run(self, task, agents):
        info = task.data.info
        admissible = self.config.admissible_commands
        started = time.time()
        game, state = await _engine(_start_game, info["game_file"], admissible)
        task = task.with_data(system_prompt=f"{SYSTEM_PROMPT}\n\n---\nINITIAL OBSERVATION:\n{_observation(state, admissible)}")
        won, steps, done, malformed = False, 0, False, 0
        try:
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn()
                while not segment.terminated:
                    steps += 1
                    if steps >= self.config.max_episode_steps:
                        break  # as in v0: the capped turn's action is not played
                    found = _ACTION.findall(segment.last_reply)
                    action = found[0] if found else "look"  # v0 plays "look" for a reply without one
                    malformed += not found
                    try:
                        state, _, done = await _engine(game.step, action)
                        won = bool(getattr(state, "won", False))
                        observation = _observation(state, admissible and not done)
                    except Exception:
                        observation = "Something went wrong. Try a different action."
                    if done:
                        break
                    segment = await interaction.turn(observation)
        finally:
            await _engine(game.close)
        trace = interaction.trace
        reward = 1.0 if won else 0.0
        trace.record_reward("task_success", reward, 1.0)
        outcome = {"won": won, "steps": steps, "done": done, "malformed_replies": malformed}
        trace.info["alfworld"] = outcome
        if self.config.episode_log:
            line = {"start": started, "end": time.time(), "reward": reward, "game": info["game"], "task_type": info["task_type"], **outcome}
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class AlfWorldTaskset(vf.Taskset[AlfWorldTask, AlfWorldConfig]):
    def load(self) -> Iterator[AlfWorldTask]:
        root = Path(self.config.dataset_path) / "json_2.1.1" / self.config.split
        games = sorted(root.glob("*/*/game.tw-pddl"))
        if not games:
            raise ValueError(f"no games under {root}")
        for game_file in games[: self.config.num_examples]:
            game = f"{game_file.parent.parent.name}/{game_file.parent.name}"
            task_type = game_file.parent.parent.name.split("-")[0]
            yield AlfWorldTask(
                AlfWorldData(
                    name=f"{task_type}#{self.config.split}/{game}",
                    prompt="What do you do?",
                    system_prompt=SYSTEM_PROMPT,  # the env appends the game's opening observation
                    info={"game_file": str(game_file), "game": game, "task_type": task_type, "split": self.config.split},
                ),
                self.config.task,
            )

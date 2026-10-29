"""craftax-env — GlyphBench's full Craftax (glyphbench/craftaxfull-v0) as a v1 taskset.

The game, the Unicode glyph observation (legend, 9x11 local grid, HUD, message), the
system prompt and the action parsing are GlyphBench's (see ``craftax_env.glyph``), so
returns compare with the paper's Craftax numbers. The harness is ours: GlyphBench shows
the model only the system prompt and the current observation each turn; here the env
plays the user of one multi-turn conversation, so KV-Streams, segments and the
renderer's thinking modes apply as they do to crafter-env.

Reward: the episode's Craftax return / 226 (GlyphBench's "% of max reward" / 100).
Episodes end on death, on the boss's defeat, at ``max_episode_steps`` model turns,
on ``max_episode_seconds`` or at a turn/token limit of the player.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

from craftax_env import glyph

logger = logging.getLogger(__name__)


class CraftaxData(vf.TaskData):
    info: dict
    """The episode's world seed and prompt settings."""


class CraftaxTask(vf.Task[CraftaxData]):
    pass


class CraftaxConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    num_examples: int = Field(5000, ge=1)
    seed: int = 42
    """Episode i of the train split uses world seed ``seed + i`` (eval: ``seed + 1_000_000 + i``)."""
    enable_reasoning: bool = True
    """GlyphBench's reasoning response format (reason, then a final <action> tag); off:
    its action-only format."""
    max_output_tokens: int | None = Field(None, ge=1)
    """The per-reply token budget the system prompt states (match the sampling cap;
    None: "no explicit cap")."""
    forfeit_mode: Literal["freeze", "noop"] = "freeze"
    """What a reply without a valid action does, as the system prompt states (match the env's)."""


class CraftaxEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "inline-chat"})
    max_episode_steps: int = Field(10000, ge=1)
    """Model turns before the episode is cut off (GlyphBench's craftaxfull default); also
    the game's own time limit and the HUD's "Step: t / max"."""
    max_episode_seconds: float | None = Field(None, gt=0)
    """Wall-clock budget per episode: the episode ends (``ended_by`` = ``time_limit``) at
    the first turn after this many seconds, the capped turn's action not played."""
    forfeit_mode: Literal["freeze", "noop"] = "freeze"
    """A reply without a valid action: "freeze" (GlyphBench's default) counts the turn
    without moving the world; "noop" plays NOOP."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, return, achievements) to this file."""


def _new_game(seed: int, max_turns: int) -> glyph.CraftaxGame:
    return glyph.CraftaxGame(seed, max_turns)


class CraftaxEnv(vf.Env[CraftaxEnvConfig]):
    async def run(self, task, agents):
        seed = int(task.data.info["seed"])
        reasoning = bool(task.data.info.get("reasoning", True))
        started = time.time()
        deadline = started + self.config.max_episode_seconds if self.config.max_episode_seconds else None
        game = await asyncio.to_thread(_new_game, seed, self.config.max_episode_steps)
        opening = await asyncio.to_thread(glyph.user_turn, game, reasoning)
        total, done, end = 0.0, False, "terminated"  # terminated: the run stopped answering
        forfeits: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        unlocked_at: dict[str, int] = {}
        max_floor = 0
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn(opening)
            while not segment.terminated:
                if deadline is not None and time.time() >= deadline:
                    end = "time_limit"
                    break
                index, failure = glyph.parse_action(segment.last_reply)
                if index is None:
                    forfeits[failure] += 1
                if index is None and self.config.forfeit_mode == "freeze":
                    done = game.forfeit()
                else:
                    index = glyph.NOOP if index is None else index
                    actions[glyph.ACTION_NAMES[index]] += 1
                    try:
                        step = await asyncio.to_thread(game.step, index)
                    except Exception as exc:  # a broken step must not kill the rollout
                        logger.warning("Craftax step error: %s", exc)
                        step = glyph.Step(0.0, False, [], 0)
                    total += step.reward
                    done = step.done
                    for name in step.unlocked:
                        unlocked_at.setdefault(name, game.turn)
                    max_floor = max(max_floor, game.floor)
                if done:
                    if game.health <= 0:
                        end = "died"
                    elif max(game.turn, game.timestep) >= game.max_turns:
                        end = "max_episode_steps"  # the game's clock also counts fast-forwarded sleep
                    else:
                        end = "won"
                    break
                observation = await asyncio.to_thread(glyph.user_turn, game, reasoning)
                segment = await interaction.turn(observation)
        trace = interaction.trace
        trace.record_reward("craftax_return", total / glyph.MAX_RETURN, 1.0)
        achievements = sorted(game.achievements)
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(achievements)),
                "total_env_reward": total,
                "forfeits": float(sum(forfeits.values())),
                "survived": 1.0 if game.health > 0 else 0.0,
                "max_floor": float(max_floor),
                "turns": float(game.turn),
            }
        )
        trace.info["craftax"] = {
            "achievements": achievements,
            "unlocked_at": unlocked_at,
            "turns": game.turn,
            "ended_by": end,
            "seed": seed,
            "max_floor": max_floor,
        }
        if self.config.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "return": total,
                "reward": total / glyph.MAX_RETURN,
                "achievements": achievements,
                "unlocked_at": unlocked_at,
                "turns": game.turn,
                "forfeits": dict(forfeits),
                "actions": dict(actions),
                "ended_by": end,
                "survived": game.health > 0,
                "max_floor": max_floor,
                "seed": seed,
                "trace_tokens": trace.num_total_tokens,
                "task": "craftax",
            }
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class CraftaxTaskset(vf.Taskset[CraftaxTask, CraftaxConfig]):
    def load(self) -> Iterator[CraftaxTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = glyph.system_prompt(cfg.enable_reasoning, cfg.max_output_tokens, cfg.forfeit_mode)
        for i in range(cfg.num_examples):
            # Prompt-less: the env opens the episode with the first observation of the
            # world it builds, so loading stays instant.
            yield CraftaxTask(
                CraftaxData(
                    name=f"craftax#{cfg.split}{i}",
                    prompt=None,
                    system_prompt=system,
                    info={"seed": base + i, "episode_idx": i, "split": cfg.split, "reasoning": cfg.enable_reasoning},
                ),
                cfg.task,
            )

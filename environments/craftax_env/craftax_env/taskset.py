"""craftax-env — GlyphBench's full Craftax (glyphbench/craftaxfull-v0) as a v1 taskset.

The game, the Unicode glyph observation (legend, 9x11 local grid, HUD, message), the
system prompt and the action parsing are GlyphBench's (see ``craftax_env.glyph``), so
returns compare with the paper's Craftax numbers. The harness is ours: GlyphBench shows
the model only the system prompt and the current observation each turn; here the env
plays the user of one multi-turn conversation, so KV-Streams, segments and the
renderer's thinking modes apply as they do to crafter-env.

Reward: the episode's Craftax return / 226 (GlyphBench's "% of max reward" / 100), or
with ``reward = "achievements"`` the unlocked achievements' Craftax reward / 226 (no
health term, which a respawning game would pay again on every death). Episodes end on
death (unless ``on_death = "respawn"``: continuous play, as crafter-env's), on the
boss's defeat, at ``max_episode_steps`` model turns (0: none), on
``max_episode_seconds`` or at a turn/token limit of the player.
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
    respawn: bool = False
    """State GlyphBench's TASK block for continuous play: a death respawns the player and
    there is no turn limit (match the env's ``on_death`` and ``max_episode_steps = 0``)."""


class CraftaxEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "inline-chat"})
    max_episode_steps: int = Field(10000, ge=0)
    """Model turns before the episode is cut off (GlyphBench's craftaxfull default); also
    the game's own time limit and the HUD's "Step: t / max". 0: no limit (continuous
    play, cut off only by `max_episode_seconds`, a death without respawn, the boss's
    defeat or a turn/token limit of the player)."""
    max_episode_seconds: float | None = Field(None, gt=0)
    """Wall-clock budget per episode: the episode ends (``ended_by`` = ``time_limit``) at
    the first turn after this many seconds, the capped turn's action not played."""
    forfeit_mode: Literal["freeze", "noop"] = "freeze"
    """A reply without a valid action: "freeze" (GlyphBench's default) counts the turn
    without moving the world; "noop" plays NOOP."""
    on_death: Literal["end", "respawn"] = "end"
    """`end`: death ends the episode, as in Craftax and GlyphBench. `respawn`: the player
    comes back on the Overworld at the free cell nearest the world's spawn point with
    full vitals, in the same world and trace, achievements, XP and attributes kept."""
    death_inventory: Literal["drop_all", "drop_materials", "keep"] = "drop_all"
    """What a death takes: every item (enchantments too), the materials and consumables
    but not the tools and armour, or nothing."""
    death_penalty: float = Field(0.0, ge=0.0)
    """Subtracted from the episode reward per death (the reward is out of 1: 1/226 is one
    tier-1 achievement)."""
    reward: Literal["return", "achievements"] = "return"
    """`return`: Craftax return / 226, GlyphBench's metric. `achievements`: the unlocked
    achievements' reward / 226, without the health term (use with respawn)."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, return, achievements) to this file."""


def _new_game(seed: int, max_turns: int | None) -> glyph.CraftaxGame:
    return glyph.CraftaxGame(seed, max_turns)


class CraftaxEnv(vf.Env[CraftaxEnvConfig]):
    async def run(self, task, agents):
        seed = int(task.data.info["seed"])
        reasoning = bool(task.data.info.get("reasoning", True))
        started = time.time()
        deadline = started + self.config.max_episode_seconds if self.config.max_episode_seconds else None
        game = await asyncio.to_thread(_new_game, seed, self.config.max_episode_steps or None)
        opening = await asyncio.to_thread(glyph.user_turn, game, reasoning)
        total, done, end = 0.0, False, "terminated"  # terminated: the run stopped answering
        forfeits: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        unlocked_at: dict[str, int] = {}
        deaths: list[dict] = []  # one entry per respawned death: turn, floor, position, where it came back
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
                died = game.health <= 0
                if done and died and not game.out_of_turns and self.config.on_death == "respawn":
                    where = {"turn": game.turn, "floor": game.floor, "pos": list(game.position)}
                    at = await asyncio.to_thread(game.respawn, self.config.death_inventory)
                    deaths.append({**where, "respawn": list(at)})
                    lost = {
                        "drop_all": "lost your inventory",
                        "drop_materials": "lost your materials (tools and armour kept)",
                        "keep": "kept your inventory",
                    }
                    game.message = (
                        f"{game.message} You died (death {len(deaths)}, on floor {where['floor']} at {tuple(where['pos'])}). "
                        f"You respawned on the Overworld at {at} with full health, food, drink, energy and "
                        f"mana, and {lost[self.config.death_inventory]}. Unlocked achievements stay unlocked."
                    )
                    done = False
                if done:
                    if died:
                        end = "died"
                    elif game.out_of_turns:
                        end = "max_episode_steps"  # the game's clock also counts fast-forwarded sleep
                    else:
                        end = "won"
                    break
                observation = await asyncio.to_thread(glyph.user_turn, game, reasoning)
                segment = await interaction.turn(observation)
        trace = interaction.trace
        score = total if self.config.reward == "return" else game.achievement_score
        reward = score / glyph.MAX_RETURN - self.config.death_penalty * len(deaths)
        trace.record_reward(f"craftax_{self.config.reward}", reward, 1.0)
        achievements = sorted(game.achievements)
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(achievements)),
                "total_env_reward": total,
                "achievement_score": game.achievement_score,
                "deaths": float(len(deaths)),
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
            "deaths": deaths,
        }
        if self.config.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "return": total,
                "achievement_score": game.achievement_score,
                "reward": reward,
                "deaths": deaths,
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
        system = glyph.system_prompt(cfg.enable_reasoning, cfg.max_output_tokens, cfg.forfeit_mode, cfg.respawn)
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

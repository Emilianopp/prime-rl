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
boss's defeat, at ``max_episode_steps`` game steps (0: none; the game's clock, so every
played action and each fast-forwarded sleep step counts, not model replies), on
``max_episode_seconds`` or at a turn/token limit of the player.

With ``state_dir`` a game outlives its job: each job of a chain plays the same seeds
on from where the last one stopped (``scripts/craftax/continuous_chain.sh``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.memory import Memory, tokens
from verifiers.v1.segments import segment_new_tokens

from craftax_env import glyph, qv

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
    max_actions_per_turn: int = Field(1, ge=1)
    """Actions the model may chain in one reply (1: GlyphBench's single final action tag).
    Above 1 the system prompt explains it and the env plays the reply's final run of
    <action> tags in order; turns (replies) and actions (game steps) are counted apart."""
    respawn: bool = False
    """State GlyphBench's TASK block for continuous play: a death respawns the player and
    there is no turn limit (match the env's ``on_death`` and ``max_episode_steps = 0``)."""
    death_drops: bool = False
    """With ``respawn``: say that a death drops the inventory where it happened, to be picked
    up within ``grave_ttl`` steps (match the env's ``death_inventory = "drop_grave"``)."""
    grave_ttl: int = Field(300, ge=1)
    """The steps the prompt says a drop lasts (match the env's ``grave_ttl``)."""
    grave_limit: int = Field(3, ge=1)
    """The drops the prompt says are kept (match the env's ``grave_limit``)."""
    episodes: list[int] | None = None
    """Play only these episode indices of the split (None: the first ``num_examples``)."""
    variants: int = Field(1, ge=1)
    """Copies of each episode, told apart by ``info["variant"]`` (above 1 only). With the
    env's ``state_dir`` each copy resumes its own save, ``{seed}-{variant}.pkl``: the same
    moment of a game played on several ways, e.g. from different handoff summaries."""


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
    death_inventory: Literal["drop_all", "drop_grave", "drop_materials", "keep"] = "drop_all"
    """What a death takes: every item (enchantments too); every item, but left where the
    player died (shown as ⊕) to pick up by walking onto it within ``grave_ttl`` steps;
    the materials and consumables but not the tools and armour; or nothing."""
    grave_ttl: int = Field(300, ge=1)
    """``drop_grave``: game steps (forfeited turns included) a drop stays before it is gone.
    300 is one Craftax day; most deaths are 8-25 cells from the respawn point."""
    grave_limit: int = Field(3, ge=1)
    """``drop_grave``: drops kept at once; a newer death removes the oldest."""
    death_penalty: float = Field(0.0, ge=0.0)
    """Subtracted from the episode reward per death (the reward is out of 1: 1/226 is one
    tier-1 achievement)."""
    reward: Literal["return", "achievements"] = "return"
    """`return`: Craftax return / 226, GlyphBench's metric. `achievements`: the unlocked
    achievements' reward / 226, without the health term (use with respawn)."""
    action_feedback: bool = False
    """Say in the [Message] block what each action did (moved or blocked by what, what DO
    collected or what tool the block needs, what a craft or placement changed, vitals that
    moved) and when a reply was forfeited. Off: GlyphBench's message (reward only)."""
    ahead_in_hud: bool = False
    """Name the faced cell and its absolute position in the HUD ("Ahead: (24,17) tree"), so
    the player can note where things are without working positions out from the grid."""
    grid_coords: bool = False
    """Label the grid's rows and columns with absolute positions, so the player can read the
    position of anything on its screen instead of working it out from its own Pos."""
    visible_list: bool = False
    """Add a [Visible] line listing what is on screen by kind with absolute positions,
    nearest first ("tree (27,14) (25,13); stone (26,12)"), so remembering a place is a copy."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, return, achievements) to this file."""
    state_dir: str | None = None
    """(Environment variables expanded, e.g. ``$CHAIN/worlds``.) Save each game to ``{state_dir}/{seed}.pkl`` (a taskset's variants: ``{seed}-{variant}.pkl``)
    every ``save_every`` turns and when the episode ends, and resume it from there: a
    chain of jobs plays the same worlds
    on. At ``max_episode_seconds`` the model first writes a handoff summary (the
    player's segments' summary request), which opens the next job's first segment as a
    segment switch would; a game cut off without one resumes with its last segment's
    inherited summary. Counters (turns, deaths, achievements' turns, ...) carry over."""
    save_every: int = Field(50, ge=1)
    """Model turns between saves of a game with ``state_dir``."""
    state_log: str | None = None
    """A folder (environment variables expanded, e.g. ``$OUT/qv_states``) where the true
    state is written at every segment boundary: ``{trace}.start.json`` when a segment
    opens and ``{trace}.end.json`` (with the next segment's id) when it ends - what the
    Q/V judge (``craftax_env.qv.score_segment``) grades."""
    max_segments: int = Field(0, ge=0)
    """Segments this run plays (0: no limit). At the next segment's first turn the run
    stops (``ended_by`` = ``segment_limit``) with that reply not played, so the game stays
    as the summary turn found it, and a ``state_dir`` save holds it there with the new
    summary."""


def _new_game(seed: int, max_turns: int | None) -> glyph.CraftaxGame:
    return glyph.CraftaxGame(seed, max_turns)


def _load_state(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _save_state(path: str, state: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)  # atomic: a job killed mid-save keeps the previous save


def _inherited_summary(trace, header: str | None) -> str | None:
    """The handoff summary the current segment opened with (its system prompt's tail)."""
    if header is None or not trace.nodes or trace.nodes[0].message.role != "system":
        return None
    content = trace.nodes[0].message.content
    return content.split(header, 1)[1].strip() if isinstance(content, str) and header in content else None


class CraftaxEnv(vf.Env[CraftaxEnvConfig]):
    async def run(self, task, agents):
        seed = int(task.data.info["seed"])
        reasoning = bool(task.data.info.get("reasoning", True))
        max_actions = int(task.data.info.get("max_actions", 1))
        feedback = self.config.action_feedback
        started = time.time()
        deadline = started + self.config.max_episode_seconds if self.config.max_episode_seconds else None
        game = await asyncio.to_thread(_new_game, seed, self.config.max_episode_steps or None)
        game.show_ahead = self.config.ahead_in_hud
        game.grid_coords = self.config.grid_coords
        game.visible_list = self.config.visible_list
        game.grave_ttl, game.grave_limit = self.config.grave_ttl, self.config.grave_limit
        total, done, end = 0.0, False, "terminated"  # terminated: the run stopped answering
        forfeits: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        turns = 0  # model replies; each plays up to max_actions actions (game steps)
        per_turn: Counter[int] = Counter()  # actions played per reply
        unlocked_at: dict[str, int] = {}
        deaths: list[dict] = []  # one entry per respawned death: turn, floor, position, where it came back
        max_floor = 0
        segments = self.config.player.segments
        header = segments.summary_header if segments is not None else None
        key = f"{seed}-{task.data.info['variant']}" if "variant" in task.data.info else f"{seed}"
        # A population sampler's node (info["tree"]): one segment from the node's saved world
        # ("load"; none for a fresh seed), saved to a file of this branch's own in "dir", so
        # the node's branches never share a save. The segment ends at the segments' summary
        # threshold with the handoff summary on this trace: one trace per episode.
        tree = task.data.info.get("tree")
        if tree is not None:
            if segments is None:
                raise ValueError("craftax tree nodes need player.segments (the segment's token budget)")
            tree_dir = os.path.expandvars(tree["dir"])
            os.makedirs(tree_dir, exist_ok=True)
            path = os.path.join(tree_dir, f"{uuid.uuid4().hex}.pkl")
            saved = await asyncio.to_thread(_load_state, tree["load"]) if tree.get("load") else None
            if tree.get("load") and saved is None:
                raise FileNotFoundError(f"tree node save {tree['load']} is gone")
        else:
            state_dir = os.path.expandvars(self.config.state_dir) if self.config.state_dir else None
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, f"{key}.pkl") if state_dir else None
            saved = await asyncio.to_thread(_load_state, path) if path else None
        block, played_before, summary = 0, 0.0, None
        if saved is not None:
            await asyncio.to_thread(game.restore, saved["game"])
            total, turns, max_floor = saved["total"], saved["turns"], saved["max_floor"]
            forfeits, actions, per_turn = Counter(saved["forfeits"]), Counter(saved["actions"]), Counter(saved["per_turn"])
            unlocked_at, deaths = saved["unlocked_at"], saved["deaths"]
            block, played_before = saved["block"] + 1, saved["played_seconds"]
            summary = saved["summary"] or saved["last_summary"]  # an empty final summary hands over nothing
            if segments is None:
                summary = None
        # The player's own memory (player.memory): saved and resumed with the game.
        memory = Memory((saved or {}).get("memory")) if self.config.player.memory is not None else None

        def save(summary_now: str | None, last_summary: str | None) -> dict:
            return {
                "version": 1, "seed": seed, "block": block, "game": game.dump(),
                "total": total, "turns": turns, "max_floor": max_floor,
                "forfeits": dict(forfeits), "actions": dict(actions), "per_turn": dict(per_turn),
                "unlocked_at": unlocked_at, "deaths": deaths,
                "played_seconds": played_before + time.time() - started,
                "summary": summary_now, "last_summary": last_summary,
                "memory": dict(memory.entries) if memory is not None else None,
            }

        opening = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions)
        kwargs = {"summary": summary or ""} if saved is not None and segments is not None else {}
        if memory is not None:
            kwargs["memory"] = memory
        async with agents.player.interaction(task, **kwargs) as interaction:
            boundaries = self.config.state_log

            def trace_id() -> str | None:
                return getattr(interaction.trace, "id", None)

            async def boundary(end_of: str | None, start_of: str | None) -> None:
                if boundaries and (end_of or start_of):
                    await asyncio.to_thread(qv.write_boundary, boundaries, game, len(deaths), end_of, start_of)

            await boundary(None, trace_id())
            segment = await interaction.turn(opening)
            while not segment.terminated:
                if deadline is not None and time.time() >= deadline:
                    end = "time_limit"
                    break
                if self.config.max_segments and getattr(interaction, "segments", 1) > self.config.max_segments:
                    end = "segment_limit"
                    break
                if path and turns and turns % self.config.save_every == 0:
                    state = save(None, _inherited_summary(interaction.trace, header))
                    await asyncio.to_thread(_save_state, path, state)
                turns += 1
                indices, failure, dropped = glyph.parse_actions(segment.last_reply, max_actions)
                if not indices:
                    forfeits[failure] += 1
                if not indices and self.config.forfeit_mode == "freeze":
                    done = game.forfeit()
                    per_turn[0] += 1
                    if feedback:
                        why = "no <action> tag" if failure == "no_action_tag" else "an unknown action name"
                        game.message = f"Your last reply had {why}, so the turn was forfeited: the world did not move."
                else:
                    forfeited = None if indices else failure
                    indices = indices or [glyph.NOOP]
                    lines, reward_sum, unlocked, played, pickups = [], 0.0, [], 0, []
                    for index in indices:
                        played += 1
                        actions[glyph.ACTION_NAMES[index]] += 1
                        before = game.snapshot() if feedback else None
                        try:
                            step = await asyncio.to_thread(game.step, index)
                        except Exception as exc:  # a broken step must not kill the rollout
                            logger.warning("Craftax step error: %s", exc)
                            step = glyph.Step(0.0, False, [], 0)
                        total += step.reward
                        reward_sum += step.reward
                        unlocked += step.unlocked
                        done = step.done
                        for name in step.unlocked:
                            unlocked_at.setdefault(name, game.turn)
                        max_floor = max(max_floor, game.floor)
                        if feedback:
                            note = f" Fast-forwarded {step.fast_forward_ticks} sleep/rest ticks." if step.fast_forward_ticks else ""
                            note += f" {step.note}" if step.note else ""
                            lines.append(game.feedback(index, before, game.snapshot()) + note)
                        elif step.note:
                            pickups.append(step.note)
                        if done:
                            break
                    per_turn[played] += 1
                    if len(indices) > 1 or feedback or dropped:
                        head = (f"Played {played} of {len(indices)} actions" if played < len(indices) else
                                f"Played {played} action{'s' if played != 1 else ''}") + f", reward {reward_sum:+.3f}."
                        body = " ".join(f"{k + 1}. {line}" for k, line in enumerate(lines)) if feedback else (
                            "Actions: " + ", ".join(glyph.ACTION_NAMES[i] for i in indices[:played]) + ".")
                        game.message = " ".join(filter(None, [
                            (f"Your last reply had {'no <action> tag' if forfeited == 'no_action_tag' else 'an unknown action name'}, "
                             "so the turn was forfeited: NOOP was played and the world moved on.") if forfeited else "",
                            head, body,
                            f"New achievements: {', '.join(unlocked)}." if unlocked else "",
                            " ".join(pickups),
                            f"{dropped} more action tag{'s were' if dropped != 1 else ' was'} not played "
                            f"(at most {max_actions} per reply, and a run stops at an unknown name)." if dropped else "",
                        ]))
                died = game.health <= 0
                if done and died and not game.out_of_turns and self.config.on_death == "respawn":
                    where = {"turn": game.turn, "floor": game.floor, "pos": list(game.position)}
                    at = await asyncio.to_thread(game.respawn, self.config.death_inventory)
                    deaths.append({**where, "respawn": list(at), **({"dropped": game.dropped is not None} if self.config.death_inventory == "drop_grave" else {})})
                    g = game.dropped
                    lost = {
                        "drop_all": "lost your inventory",
                        "drop_grave": (f"dropped your inventory ({glyph._describe(g['items'])}) at ({g['pos'][0]},{g['pos'][1]}) on floor {g['floor']}: "
                                       f"walk onto that cell by step {g['at'] + game.grave_ttl} to pick it up" if g else "had nothing to drop"),
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
                observation = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions)
                if tree is not None and segment_new_tokens(interaction.trace) >= segments.new_tokens - segments.summary_tokens:
                    end = "segment_end"  # the node's segment is done: its handoff summary closes it below
                    break
                before = trace_id()
                segment = await interaction.turn(observation)
                if trace_id() != before:  # a segment switch: the game is as the summary turn found it
                    await boundary(before, trace_id())
            await boundary(trace_id(), None)
            if path:
                handoff = None
                if end in ("time_limit", "segment_end") and hasattr(interaction, "summarize"):
                    # The game pauses here: the model writes the summary the next job opens with,
                    # over the observation it will be shown again (the cut-off reply's actions
                    # were not played).
                    pending = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions)
                    handoff = await interaction.summarize(pending)
                state = save(handoff, _inherited_summary(interaction.trace, header))
                await asyncio.to_thread(_save_state, path, state)
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
                "turns": float(turns),
                "actions": float(sum(actions.values())),
                "game_steps": float(game.turn),
            }
        )
        trace.info["craftax"] = {
            "achievements": achievements,
            "unlocked_at": unlocked_at,
            "turns": turns,
            "actions": sum(actions.values()),
            "game_steps": game.turn,
            "ended_by": end,
            "seed": seed,
            "max_floor": max_floor,
            "deaths": deaths,
            "points": game.achievement_score,  # Craftax achievement points so far (1/3/5/8 by tier)
            **({"state_path": path, "continues": end == "segment_end"} if tree is not None else {}),
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
                "turns": turns,  # model replies
                "game_steps": game.turn,  # actions played + forfeited turns; unlocked_at and deaths count these
                "actions_played": sum(actions.values()),
                "actions_per_turn": {str(k): v for k, v in sorted(per_turn.items())},
                "forfeits": dict(forfeits),
                "actions": dict(actions),
                "ended_by": end,
                "survived": game.health > 0,
                "max_floor": max_floor,
                **({"graves": dict(game.grave_stats)} if self.config.death_inventory == "drop_grave" else {}),
                "seed": seed,
                "block": block,  # jobs this game has run in before this one (state_dir)
                "played_seconds": played_before + time.time() - started,
                "trace_tokens": trace.num_total_tokens,
                **({"memory": {"tokens": {k: tokens(v) for k, v in memory.entries.items()}, "ops": dict(memory.stats)}} if memory is not None else {}),
                "task": "craftax",
            }
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class CraftaxTaskset(vf.Taskset[CraftaxTask, CraftaxConfig]):
    def load(self) -> Iterator[CraftaxTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = glyph.system_prompt(cfg.enable_reasoning, cfg.max_output_tokens, cfg.forfeit_mode, cfg.respawn, cfg.max_actions_per_turn,
                                     (cfg.grave_ttl, cfg.grave_limit) if cfg.respawn and cfg.death_drops else None)
        for i in cfg.episodes if cfg.episodes is not None else range(cfg.num_examples):
            for v in range(cfg.variants):
                # Prompt-less: the env opens the episode with the first observation of the
                # world it builds, so loading stays instant.
                yield CraftaxTask(
                    CraftaxData(
                        name=f"craftax#{cfg.split}{i}" + (f"v{v}" if cfg.variants > 1 else ""),
                        prompt=None,
                        system_prompt=system,
                        info={
                            "seed": base + i, "episode_idx": i, "split": cfg.split, "reasoning": cfg.enable_reasoning,
                            "max_actions": cfg.max_actions_per_turn,
                            **({"variant": v} if cfg.variants > 1 else {}),
                        },
                    ),
                    cfg.task,
                )

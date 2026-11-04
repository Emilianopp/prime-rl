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
import gc
import json
import logging
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

import verifiers.v1 as vf

from craftax_env import glyph
from craftax_env.forfeits import ForfeitTracker, classify, reply_parts
from craftax_env.segments import (
    CONTINUATION_HEADER,
    NOTES_RETRY_REASONS,
    Timeline,
    TurnsLost,
    count_findings,
    NOTES_OPENER,
    SUMMARY_PROMPT,
    SUMMARY_RETRY,
    PLAN_PROMPT,
    PLAN_RETRY,
    PLAN_SECTIONS,
    SECTIONS,
    event_log,
    continuation_prompt,
    extract_notes,
    floor_name,
    game_record,
    notes_format,
    notes_of_summary,
    standing_stations,
    notes_problem,
    parse_notes,
    summary_text,
    with_notes_protocol,
)
from craftax_env import world
from craftax_env.adherence import AdherenceTracker, parse_plan
from craftax_env.navigation import ExplorationTracker, exploration_aid, observe_game
from craftax_env.plan_check import check_plan
from craftax_env.judge import (
    CraftaxAdherenceJudge,
    CraftaxGroupAdherenceJudge,
    CraftaxGroupNotesJudge,
    CraftaxNotesJudge,
    apply_adherence,
    apply_verdict,
    code_adherence_verdict,
    group_adherence_verdicts,
    group_verdicts,
    judge_adherence,
    judge_adherence_group,
    judge_notes,
    judge_notes_group,
    skip_adherence,
    skip_verdict,
)
# Shared with crafter-env (Crafter jobs are retired; these move to a shared module later).
from crafter_env.capi import CrafterJudgeConfig
from crafter_env.judge import GroupRendezvous, scale_cost, usage_cost
from crafter_env.state_store import StateStore, open_store, retry_io
from crafter_env.taskset import (_read_beam, _turn_sampling, _write_beam, mask_notes_from_loss, notes_node_indexes,
                                 request_term_routing, select_children)

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
    action_example: bool = False
    """Show the exact action-tag format with a real Craftax action (<action>MOVE_LEFT</action>) in
    the system prompt's RESPONSE FORMAT and in every turn's closing request, and name the format in
    the forfeit feedback. Off: GlyphBench's text (its example is MOVE_FORWARD, not a Craftax action)."""
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
    episode_deadline: float | None = None
    """Unix time at which every episode ends (``ended_by`` = ``deadline``), checked like
    `max_episode_seconds`: the job's wall clock, so episodes started late (N > concurrency)
    are logged instead of killed with the job. They are cut short, so summaries keep them apart."""
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
    respawn_safe: bool = False
    """Opt-in, as in crafter-env: respawn at the free cell nearest the spawn point with no
    hostile mob or mob projectile within `respawn_safe_radius` tiles (else the nearest free
    cell). Off: the nearest free cell, as upstream."""
    respawn_safe_radius: int = Field(2, ge=1)
    """With respawn_safe: how far (in tiles, a square around the cell) no hostile may be."""
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
    episode_log: str | None = None
    """Append one JSON line per finished episode (wall-clock start/end, return, achievements) to this file."""
    # Segments (docs/craftax-segments-judge.md): the game continues across episodes. A
    # segment restores the game its parent saved, plays until the cut, writes notes in the
    # same session and saves the game; the next one starts from the game's record of the
    # state plus those notes. Reward: the segment's Craftax achievement reward (1/3/5/8,
    # no health term) as the `progress` term. Set max_episode_steps = 0 (no game clock).
    segment_turns: int | None = Field(None, ge=1)
    """Model turns per segment (100 in the design)."""
    segment_tokens: int | None = Field(None, ge=1)
    """Logical positions per segment (`trace.num_total_tokens`, evicted tokens included):
    the safety cut that leaves room for the notes turn under the model's limit."""
    summary_max_tokens: int | None = Field(32768, ge=1)
    """Output cap for the notes turn only (cut to what the context still holds)."""
    summary_prompt: str | None = None
    """Overrides the default notes request (segments.SUMMARY_PROMPT)."""
    action_from_reasoning: bool = False
    """Play the action tag(s) a reply's reasoning ends with when its content holds no tag (the
    model closed its thinking on the final tag). False: such a turn is forfeited (GlyphBench reads
    the content only). Counted as `actions_from_reasoning`."""
    exploration_aid: bool = False
    """Show GlyphBench's [Exploration Aid] (navigation.py: visits, recent route, frontiers, last-seen
    features on this floor, computed from the observations) after every observation, as its Pro harness
    does. False (GlyphBench's Basic condition): it appears only in the code_facts facts at a segment's end."""
    code_facts: bool = False
    """Split the notes between code and the model: the game writes the KNOWN PLACES (world.known_places,
    from every cell seen on the floor) and the EVENT LOG (the segment's timeline) into the notes request
    and into the next segment's start, and the model writes only FINDINGS / PLAN / SURVIVAL
    (segments.PLAN_PROMPT); its MAP claims are no longer graded. False: the model writes all four sections."""
    findings_weight: float = Field(0.3, ge=0.0, le=1.0)
    """code_facts, notes_judge = "verdict": the notes reward is (1 - w) x plan + w x findings, where plan = the
    code check (plan_check: feasible, first subgoal on the critical path) raised by the judge's verdict on a
    departure from that path, and findings = the judge's lesson score (dropped when the judge fails)."""
    notes_judge: Literal["verdict", "rank"] = "rank"
    """code_facts: how the LLM grades the notes. `rank` (2026-11-04): one comparative call per parent ranks
    the siblings' notes (judge.CF_RANK_PROMPT) and marks false findings; notes = (1 - judge_weight) x
    plan + judge_weight x rank score - false_finding_penalty x (false findings / findings), where plan =
    mean(feasible, on_track) by code (plan_check). A branch with no siblings in one call gets the code
    half alone. `verdict`: the 2026-11-03 per-finding verdict (truth / lesson / applied, waste, deviation)
    with plan = mean(feasible, on_path raised by deviation); see docs/craftax-reward-code-facts.md."""
    judge_weight: float = Field(0.5, ge=0.0, le=1.0)
    """notes_judge = "rank": the rank score's weight in the notes reward."""
    false_finding_penalty: float = Field(0.3, ge=0.0, le=1.0)
    """notes_judge = "rank": taken off the notes reward per false finding (as a fraction of the findings)."""
    state_weight: float = Field(1.0, ge=0.0)
    """Segments: weight of the `state` reward, the achievement points of what the player holds at the cut
    (world.state_value: pickaxe and sword tier, a table or furnace standing on a seen cell). Under the
    drop-everything respawn it tells a lineage that kept its tools from one that died (progress alone
    cannot, and in the 2026-11-04 judged beam held tools predicted the children's progress where the
    judge did not). 0: off. In RL (`advantage_mode = "per_term"`) give it a std floor like progress."""
    beam_cumulative: bool = True
    """The in-env beam ranks branches by the lineage's total achievement score (progress so far) plus the
    segment's other terms, so a lineage ahead on achievements is never dropped for a lower notes or
    adherence score. False: by the segment's reward alone (the 2026-11-03 runs)."""
    summary_retry: bool = True
    """Ask once more (segments.SUMMARY_RETRY) when the notes reply holds no <notes> of 80+ characters."""
    train_on_notes: bool = False
    """Segments: train on the notes turn (and its retry). False: those replies, and any earlier reply
    that holds <notes>, keep their tokens as context but leave the loss (crafter_env's
    `mask_notes_from_loss`); the notes are still graded and still reach the next segment."""
    record_stations: bool = True
    """List the tables and furnaces the player placed in the game record."""
    state_dir: str | None = None
    """Where saved games live (one pickle per segment; unset: in memory, one process only)."""
    tree_depth: int = Field(0, ge=0)
    """> 0: the episode is a whole frozen beam inside the env (eval arm A0): `branches`
    segments per node, the `keep` best continue, down to this depth. 0: one segment per
    episode (the RL pipeline's tree sampler drives the search)."""
    branches: int = Field(4, ge=1)
    keep: int = Field(2, ge=1)
    keep_per_parent: int | None = Field(None, ge=1)
    """Cap on the kept children of one parent (see crafter_env.taskset.select_children)."""
    root_branches: int | None = Field(None, ge=1)
    """Branches of the root (default branches x keep)."""
    # The notes judge (Milestone 2): an LLM grades the notes' findings and plan, code checks
    # their MAP lines; the `notes` reward is 0.5 x (0.6 map_check + 0.4 findings) + 0.5 x plan.
    judge: CrafterJudgeConfig | None = None
    """The judge's client (backend "openai" incl. Gemini's endpoint, or "capi"); None: no notes reward."""
    notes_weight: float = Field(0.5, ge=0.0)
    """Weight of the `notes` reward next to `progress` (1.0)."""
    judge_mode: Literal["single", "group"] = "single"
    """`single`: each segment's notes are graded alone as it ends. `group`: the branches of one
    node in one comparative call once the last ends (RL: they meet in `<state_dir>/judge_groups`)."""
    judge_concurrency: int = Field(8, ge=1)
    """Concurrent judge calls per env worker."""
    judge_group_size: int = Field(4, ge=2)
    """Branches per node in group mode in the RL pipeline (the orchestrator's group_size)."""
    judge_group_timeout_s: float = Field(1800.0, gt=0)
    """How long a finished branch waits for its siblings before it is graded alone."""
    judge_price: tuple[float, float, float] | None = None
    """USD per 1M tokens (input, output, cached input) for the cost metrics; unset: crafter_env.judge.PRICES."""
    judge_world_radius: int = Field(24, ge=0)
    """The judge's nearest-features list covers this many tiles around the player (privileged)."""
    map_claim_tolerance: int = Field(1, ge=0)
    """A MAP line is correct when a seen cell of its feature is within this many tiles (Chebyshev)."""
    # Adherence (Milestone 3, arm C): an LLM grades, on a code-built digest (adherence.py), how
    # a segment worked on the PLAN its parent wrote. Reward `adherence` = the parent's plan
    # score (the gate) x the adherence score, on the child's trace only.
    adherence_weight: float = Field(0.0, ge=0.0)
    """Weight of the `adherence` reward (0: off; arm C: 0.25). `judge_mode` applies as for the notes when
    `adherence_mode` is "judge". The digest is logged whenever a segment has a parent PLAN."""
    adherence_mode: Literal["judge", "code"] = "code"
    """`code` (2026-11-04): the statuses come from the digest alone (judge.code_adherence_verdict: done by
    code, attempted = came within 2 tiles of the subgoal's place, else ignored; subgoals code cannot check
    are not scored), no LLM call and no gate. `judge`: the LLM status list gated by the parent's plan score."""

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
        if self.adherence_weight > 0 and self.adherence_mode == "judge" and self.judge is None:
            raise ValueError("adherence_weight > 0 with adherence_mode = 'judge' needs a judge")
        return self


def _new_game(seed: int, max_turns: int | None) -> glyph.CraftaxGame:
    return glyph.CraftaxGame(seed, max_turns)


_PLACE_STATIONS = {glyph.ACTION_NAMES.index("PLACE_TABLE"): ("table", glyph.TABLE_BLOCK),
                   glyph.ACTION_NAMES.index("PLACE_FURNACE"): ("furnace", glyph.FURNACE_BLOCK)}
_SLEEP = glyph.ACTION_NAMES.index("SLEEP")
_DO = glyph.ACTION_NAMES.index("DO")
_LOGGED_PREFIXES = ("MAKE_", "PLACE_", "ENCHANT_", "DRINK_", "READ_", "LEVEL_UP_", "CAST_")


def _logs_feedback(index: int) -> bool:
    """Actions whose effect (or failure, and why) goes on the judge's timeline."""
    name = glyph.ACTION_NAMES[index]
    return name.startswith(_LOGGED_PREFIXES) or name in ("DO", "DESCEND", "ASCEND", "SHOOT_ARROW")


def _notable_do(text: str) -> bool:
    """A DO worth a timeline line: a collection blocked for a tool, a chest, a fountain."""
    return "needs" in text or "chest" in text or "fountain" in text


def _node_of(task: vf.Task) -> dict[str, Any]:
    node = dict(task.data.info.get("node") or {})
    node.setdefault("id", f"{task.data.info.get('seed')}-root")
    node.setdefault("depth", 0)
    node.setdefault("parent", None)
    node.setdefault("state_path", None)
    node.setdefault("summary", None)
    return node


def child_task(task: vf.Task, node: dict[str, Any]) -> vf.Task:
    """The task of a segment that continues `node` (the in-env beam mints it; the tree
    sampler builds the same prompt): the root task with the record and notes after the
    base system prompt and the node in the info."""
    base = task.data.info.get("base_system_prompt") or task.data.system_prompt or ""
    info = {**task.data.info, "base_system_prompt": base, "node": node}
    if int(node.get("depth", 0)) == 0:
        return task.with_data(system_prompt=base, info=info)
    name = f"{task.data.info.get('root_name') or task.data.name}/d{node.get('depth', 0)}/{node.get('id')}"
    return task.with_data(system_prompt=continuation_prompt(base, node.get("summary")), prompt=None, info=info, name=name)


class CraftaxEnv(vf.Env[CraftaxEnvConfig]):
    # --- resources ---------------------------------------------------------------
    def _store(self) -> StateStore:
        store = self.__dict__.get("_state_store")
        if store is None:
            store = self.__dict__["_state_store"] = open_store(self.config.state_dir)
        return store

    def segmented(self) -> bool:
        return self.config.segment_turns is not None or self.config.segment_tokens is not None

    def _judge_gate(self) -> asyncio.Semaphore:
        gate = self.__dict__.get("_judge_semaphore")
        if gate is None:
            gate = self.__dict__["_judge_semaphore"] = asyncio.Semaphore(max(1, self.config.judge_concurrency))
        return gate

    def _rendezvous(self) -> GroupRendezvous:
        rv = self.__dict__.get("_judge_rendezvous")
        if rv is None:
            if not self.config.state_dir:
                raise ValueError("judge_mode = 'group' in the RL pipeline needs state_dir (the branches meet on disk)")
            rv = self.__dict__["_judge_rendezvous"] = GroupRendezvous(
                os.path.join(self.config.state_dir, "judge_groups"), self.config.judge_group_size, self.config.judge_group_timeout_s)
        return rv

    def adhering(self) -> bool:
        cfg = self.config
        return cfg.adherence_weight > 0 and (cfg.adherence_mode == "code" or cfg.judge is not None)

    def _adherence_rendezvous(self) -> GroupRendezvous:
        rv = self.__dict__.get("_adherence_rendezvous_obj")
        if rv is None:
            if not self.config.state_dir:
                raise ValueError("judge_mode = 'group' in the RL pipeline needs state_dir (the branches meet on disk)")
            rv = self.__dict__["_adherence_rendezvous_obj"] = GroupRendezvous(
                os.path.join(self.config.state_dir, "judge_groups_adherence"), self.config.judge_group_size,
                self.config.judge_group_timeout_s)
        return rv

    def _plan_score_path(self, node_id: str | None) -> str | None:
        if not self.config.state_dir or not node_id:
            return None
        return os.path.join(self.config.state_dir, "plan_scores", f"{node_id}.json")

    @staticmethod
    def _plan_score(trace: Any) -> float | None:
        """The notes judge's plan score of a graded segment (0 when its notes were skipped)."""
        j = (trace.info.get("craftax") or {}).get("judge") or {}
        if j.get("skipped"):
            return 0.0
        return float(j["plan"]) if j.get("plan") is not None else None

    def _save_plan_score(self, trace: Any) -> None:
        """The RL pipeline: the children read their gate from `<state_dir>/plan_scores/<node>.json`
        (the tree sampler hands them only the summary)."""
        info = trace.info.get("craftax") or {}
        path, plan = self._plan_score_path(info.get("node_id")), self._plan_score(trace)
        if path is None or plan is None or not info.get("continues"):
            return

        def write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"plan": plan, "failed": bool((info.get("judge") or {}).get("failed")), "at": time.time()}, f)
            os.replace(tmp, path)

        retry_io(write, what=f"writing {path}")

    def _parent_plan_score(self, node: dict[str, Any]) -> float | None:
        if node.get("plan_score") is not None:  # the in-env beam carries it on the node
            return float(node["plan_score"])
        path = self._plan_score_path(node.get("id"))
        try:
            with open(path) as f:
                return float(json.load(f)["plan"])
        except (TypeError, OSError, ValueError, KeyError):
            return None

    async def _grade_notes(self, trace: Any, *, rendezvous: bool) -> None:
        cfg = self.config
        info = trace.info.get("craftax") or {}
        if not info.get("notes"):
            skip_verdict(trace, weight=cfg.notes_weight, reason="no notes", episode_log=cfg.episode_log)
        elif rendezvous and cfg.judge_mode == "group" and info.get("parent"):
            await judge_notes_group(self._rendezvous(), CraftaxNotesJudge(cfg.judge), CraftaxGroupNotesJudge(cfg.judge), trace,
                                    weight=cfg.notes_weight, gate=self._judge_gate(), episode_log=cfg.episode_log, price=cfg.judge_price)
        else:
            await judge_notes(CraftaxNotesJudge(cfg.judge), trace, weight=cfg.notes_weight, gate=self._judge_gate(),
                              episode_log=cfg.episode_log, price=cfg.judge_price)

    async def _grade_adherence(self, trace: Any, *, rendezvous: bool) -> None:
        """A segment with a parent PLAN (depth 2+): its adherence; none for the root's children."""
        cfg = self.config
        info = trace.info.get("craftax") or {}
        digest = info.get("adherence_digest")
        if not self.adhering() or digest is None or "adherence_judge" in info:
            return
        if not digest.get("subgoals"):
            skip_adherence(trace, weight=cfg.adherence_weight, reason="no parent plan", episode_log=cfg.episode_log)
        elif cfg.adherence_mode == "code":
            apply_adherence(trace, code_adherence_verdict(info), weight=cfg.adherence_weight, timing={"call_s": 0.0},
                            episode_log=cfg.episode_log, mode="code", gated=False)
        elif rendezvous and cfg.judge_mode == "group" and info.get("parent"):
            await judge_adherence_group(self._adherence_rendezvous(), CraftaxAdherenceJudge(cfg.judge),
                                        CraftaxGroupAdherenceJudge(cfg.judge), trace, weight=cfg.adherence_weight,
                                        gate=self._judge_gate(), episode_log=cfg.episode_log, price=cfg.judge_price)
        else:
            await judge_adherence(CraftaxAdherenceJudge(cfg.judge), trace, weight=cfg.adherence_weight, gate=self._judge_gate(),
                                  episode_log=cfg.episode_log, price=cfg.judge_price)

    def _judged(self, result: dict[str, Any]) -> None:
        trace = result["trace"]
        result["reward"], result["plan_score"] = float(trace.reward), self._plan_score(trace)
        result["beam_score"] = self._beam_score(result)

    def _beam_score(self, result: dict[str, Any]) -> float:
        """What the in-env beam ranks by: the segment's reward (every term) plus, with
        `beam_cumulative`, the achievement score the lineage had before this segment."""
        before = float(result.get("achievement_score_before") or 0.0) if self.config.beam_cumulative else 0.0
        return float(result["trace"].reward) + before

    async def finalize(self, task, episode) -> None:
        """The RL pipeline's judges: grade each segment the run left ungraded (notes, and
        adherence when on) and leave its plan score for its children. The in-env beam grades
        its branches as they close, so this is a no-op there."""
        cfg = self.config
        if cfg.judge is None and not self.adhering():
            return
        for trace in episode.traces:
            info = trace.info.get("craftax") or {}
            if not info or not getattr(trace, "ok", True) or "node_id" not in info:
                continue
            jobs = [self._grade_adherence(trace, rendezvous=True)]
            if cfg.judge is not None and "judge" not in info:
                jobs.append(self._grade_notes(trace, rendezvous=True))
            await asyncio.gather(*jobs)
            self._save_plan_score(trace)

    async def _judge_branch(self, result: dict[str, Any]) -> None:
        """The beam's single mode: grade one finished branch (notes, and adherence when on)
        and refresh its reward and plan score."""
        if (self.config.judge is None and not self.adhering()) or result.get("trace") is None:
            return
        trace = result["trace"]
        jobs = [self._grade_adherence(trace, rendezvous=False)]
        if self.config.judge is not None:
            jobs.append(self._grade_notes(trace, rendezvous=False))
        await asyncio.gather(*jobs)
        self._judged(result)

    async def _judge_group(self, results: list[dict[str, Any]]) -> None:
        """The beam's group mode: the branches of one node in one comparative call per
        judge; a branch a call does not cover is graded alone."""
        if self.config.judge is None and not self.adhering():
            return
        done = [r for r in results if r.get("trace") is not None]
        await asyncio.gather(self._notes_group(done), self._adherence_group(done))
        for r in done:
            self._judged(r)

    async def _notes_group(self, done: list[dict[str, Any]]) -> None:
        cfg = self.config
        if cfg.judge is None:
            return
        with_notes = [r for r in done if (r["trace"].info.get("craftax") or {}).get("notes")]
        for r in done:
            if r not in with_notes:
                skip_verdict(r["trace"], weight=cfg.notes_weight, reason="no notes", episode_log=cfg.episode_log)
        if len(with_notes) <= 1:
            for r in with_notes:
                await self._grade_notes(r["trace"], rendezvous=False)
            return
        infos = [{k: v for k, v in r["trace"].info["craftax"].items() if k not in ("judge", "adherence_judge")} for r in with_notes]
        started, costs = time.time(), []
        verdicts, call_s, error = await group_verdicts(CraftaxGroupNotesJudge(cfg.judge), infos, trace=with_notes[0]["trace"],
                                                       gate=self._judge_gate(), price=cfg.judge_price, costs=costs)
        share = scale_cost(costs[0], 1.0 / len(with_notes)) if costs else usage_cost(None, cfg.judge.model)
        for i, r in enumerate(with_notes):
            tr = r["trace"]
            if verdicts is None:
                await self._grade_notes(tr, rendezvous=False)
                continue
            timing = {"group_wait_s": started - float(tr.info["craftax"].get("ended_at") or started) + call_s,
                      "call_s": call_s, "group_size": float(len(with_notes)), "group_fallback": 0.0}
            apply_verdict(tr, verdicts[i], weight=cfg.notes_weight, timing=timing, episode_log=cfg.episode_log, mode="group", cost=share)

    async def _adherence_group(self, done: list[dict[str, Any]]) -> None:
        cfg = self.config
        if not self.adhering():
            return
        graded = [r for r in done if (r["trace"].info.get("craftax") or {}).get("adherence_digest") is not None]
        with_plan = [r for r in graded if r["trace"].info["craftax"]["adherence_digest"].get("subgoals")]
        if cfg.adherence_mode == "code" or len(with_plan) <= 1 or len(with_plan) < len(graded):  # siblings share the plan
            for r in graded:
                await self._grade_adherence(r["trace"], rendezvous=False)
            return
        infos = [{k: v for k, v in r["trace"].info["craftax"].items() if k not in ("judge", "adherence_judge")} for r in with_plan]
        costs: list = []
        verdicts, call_s, error = await group_adherence_verdicts(CraftaxGroupAdherenceJudge(cfg.judge), infos,
                                                                 trace=with_plan[0]["trace"], gate=self._judge_gate(),
                                                                 price=cfg.judge_price, costs=costs)
        share = scale_cost(costs[0], 1.0 / len(with_plan)) if costs else usage_cost(None, cfg.judge.model)
        for i, r in enumerate(with_plan):
            if verdicts is None:
                await self._grade_adherence(r["trace"], rendezvous=False)
                continue
            timing = {"call_s": call_s, "group_size": float(len(with_plan)), "group_fallback": 0.0}
            apply_adherence(r["trace"], verdicts[i], weight=cfg.adherence_weight, timing=timing, episode_log=cfg.episode_log,
                            mode="group", cost=share)

    # --- one episode -------------------------------------------------------------
    async def run(self, task, agents):
        if self.config.tree_depth > 0:
            await self._run_tree(task, agents)
            return
        await self.play_segment(task, agents.player)

    # --- the frozen beam inside one run (arm A0 of docs/craftax-segments-judge.md) ---
    async def _run_tree(self, task, agents) -> None:
        """`branches` segments per node, the `keep` best by reward continue, down to
        `tree_depth`. Each branch's reward is its progress plus, with a judge, the notes
        reward (graded as the branch ends, or per node in group mode). Resumes from
        `<state_dir>/beam/<seed>.json`."""
        cfg = self.config
        store = self._store()
        seed = int(task.data.info["seed"])
        beam = [{"id": f"{seed}-root", "parent": None, "depth": 0, "state_path": None, "summary": None}]
        root_fanout = cfg.root_branches or cfg.branches * cfg.keep
        beam_file = os.path.join(cfg.state_dir, "beam", f"{seed}.json") if cfg.state_dir else None
        start_depth, depth = 1, 0
        saved = _read_beam(beam_file)
        if saved:
            if saved.get("done") and not (saved["done"] == "tree_depth" and int(saved.get("depth") or 0) < cfg.tree_depth):
                logger.info("seed %d: the beam already ended (%s) at depth %s", seed, saved["done"], saved.get("depth"))
                return
            nodes = [n for n in saved.get("beam") or [] if n.get("state_path") and os.path.exists(n["state_path"])]
            if nodes:
                beam, start_depth = nodes, int(saved["depth"]) + 1
                logger.info("seed %d: resuming the beam from depth %d with %d nodes", seed, saved["depth"], len(nodes))
        if start_depth > cfg.tree_depth:  # already past this run's cap (a chunked run): leave the checkpoint as it is
            logger.info("seed %d: the beam is at depth %d, at or past tree_depth %d", seed, start_depth - 1, cfg.tree_depth)
            return
        found, last_level_s = None, None
        for depth in range(start_depth, cfg.tree_depth + 1):
            level_started = time.time()
            # A level the job's deadline would cut is not started: the checkpoint stays at the
            # last full level, and the next job (STATE_DIR, RESUME_FROM) goes on from there.
            if cfg.episode_deadline is not None and last_level_s is not None and level_started + 1.1 * last_level_s > cfg.episode_deadline:
                logger.info("seed %d: depth %d would not finish before the deadline (the last level took %.1f min); "
                            "the beam stays at depth %d for a resume", seed, depth, last_level_s / 60, depth - 1)
                return
            fanout = root_fanout if depth == 1 else cfg.branches
            jobs = [(node, b) for node in beam for b in range(fanout)]

            async def node_branches(node: dict[str, Any]) -> list[Any]:
                outs = await asyncio.gather(*(self.play_segment(child_task(task, {**node, "branch": b}), agents.player)
                                              for b in range(fanout)), return_exceptions=True)
                await self._judge_group([o for o in outs if not isinstance(o, BaseException)])
                return list(outs)

            async def branch(node: dict[str, Any], b: int) -> dict[str, Any]:
                out = await self.play_segment(child_task(task, {**node, "branch": b}), agents.player)
                await self._judge_branch(out)
                return out

            if cfg.judge is not None and cfg.judge_mode == "group":
                per_node = await asyncio.gather(*(node_branches(node) for node in beam))
                outcomes = [o for outs in per_node for o in outs]
            else:
                outcomes = await asyncio.gather(*(branch(node, b) for node, b in jobs), return_exceptions=True)
            if any(not isinstance(o, BaseException) and o.get("ended_by") == "deadline" for o in outcomes):
                # The deadline cut this level: its branches are partial, so none of them is a child.
                # Drop them and keep the parents' states and the checkpoint of the last full level.
                for o in outcomes:
                    if not isinstance(o, BaseException) and o.get("state_path"):
                        store.delete(o["state_path"])
                logger.info("seed %d: the deadline cut depth %d; dropped its branches, the beam stays at depth %d "
                            "for a resume", seed, depth, depth - 1)
                return
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
            for r in results:
                r.setdefault("beam_score", self._beam_score(r))
            kept = select_children(results, cfg.keep, cfg.keep_per_parent)
            kept_paths = {r.get("state_path") for r in kept}
            for r in results:
                if r.get("state_path") and r["state_path"] not in kept_paths:
                    store.delete(r["state_path"])
            beam = [{"id": r["node_id"], "parent": r["parent"], "depth": depth, "state_path": r["state_path"],
                     "state_sha256": r.get("state_sha256"), "summary": r.get("summary"), "plan_score": r.get("plan_score")}
                    for r in kept]
            logger.info("seed %d depth %d: %d branches, kept %d (best beam score %.3f, achievements %s), %.1f min",
                        seed, depth, len(results), len(kept), max((float(r.get("beam_score", 0.0)) for r in kept), default=float("nan")),
                        max((len(r.get("achievements") or []) for r in kept), default=0), (time.time() - level_started) / 60)
            _write_beam(beam_file, {"seed": seed, "depth": depth, "beam": beam, "done": None, "at": time.time()})
            last_level_s = time.time() - level_started
            gc.collect()
            gc.freeze()  # finished levels stay in the episode; keep the cyclic GC off them
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

    # --- one segment (or a whole game when segments are off) ------------------------
    async def play_segment(self, task, agent) -> dict[str, Any]:
        cfg = self.config
        segmented = self.segmented()
        node = _node_of(task)
        seed = int(task.data.info["seed"])
        reasoning = bool(task.data.info.get("reasoning", True))
        max_actions = int(task.data.info.get("max_actions", 1))
        example = bool(task.data.info.get("action_example", False))
        feedback = cfg.action_feedback
        started = time.time()
        deadline = started + cfg.max_episode_seconds if cfg.max_episode_seconds else None
        store = self._store() if segmented else None
        max_turns = cfg.max_episode_steps or None
        if node.get("state_path"):
            saved = await asyncio.to_thread(store.get, node["state_path"], node.get("state_sha256"))
            game = await asyncio.to_thread(glyph.CraftaxGame.restore, saved["game"], max_turns)
            # A station destroyed in an earlier segment (DO facing it, an arrow) is dropped here.
            explored, stations = saved["explored"], standing_stations(game, list(saved.get("stations") or []))
            nav = saved.get("nav")
        else:
            game = await asyncio.to_thread(_new_game, seed, max_turns)
            explored, stations, nav = None, [], None
        if segmented:
            task = task.with_data(system_prompt=with_notes_protocol(task.data.system_prompt))
            if explored is None:
                explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
            floor, mask = game.visible()
            explored[floor] |= mask
        if segmented and (cfg.code_facts or cfg.exploration_aid):  # the route and visits carry across segments
            nav = nav or ExplorationTracker()
            await asyncio.to_thread(observe_game, nav, game)
        else:
            nav = None

        def aid() -> str | None:
            return exploration_aid(nav, game.floor) if nav is not None and cfg.exploration_aid else None

        first_turn = game.turn
        record_before = game_record(game, stations=stations, record_stations=cfg.record_stations, depth=int(node.get("depth", 0)),
                                    first_turn=first_turn, deaths=[], death_inventory=cfg.death_inventory, previous=False) if segmented else None
        score_before = game.achievement_score
        achievements_before = sorted(game.achievements)
        # The parent's PLAN, tracked for the adherence judge (segments below the root's children).
        adherence = parent_plan_score = None
        if segmented and int(node.get("depth", 0)) > 0:
            names = [a.name.lower() for a in game._rt["achievements"]]
            # The parent wrote its PLAN in the view of this segment's first observation.
            subgoals = parse_plan(parse_notes(notes_of_summary(node.get("summary")))["PLAN"], names, game.floor,
                                  view=(game.floor, *map(int, game.position)))
            adherence = AdherenceTracker(subgoals, game, achievements_before, explored, cfg.map_claim_tolerance)
            parent_plan_score = self._parent_plan_score(node)
        opening = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions, True, example, aid())
        total, done, end = 0.0, False, "terminated"  # terminated: the run stopped answering
        forfeits: Counter[str] = Counter()
        from_reasoning = 0  # action_from_reasoning: turns played from the tag the reasoning ended with
        actions: Counter[str] = Counter()
        turns = 0  # model replies; each plays up to max_actions actions (game steps)
        per_turn: Counter[int] = Counter()  # actions played per reply
        unlocked_at: dict[str, int] = {}
        deaths: list[dict] = []  # one entry per respawned death: turn, floor, position, where it came back
        timeline = Timeline()  # the segment's events for the judge: unlocks, floors, stations, crafts, sleeps, deaths
        max_floor = game.floor
        tracker = ForfeitTracker()  # forfeit kinds, streaks and collapse onset
        notes, retry_reason, notes_retried, continues = None, None, False, False
        facts: str | None = None  # code_facts: the known places + event log the notes turn showed
        wasted = TurnsLost() if segmented and cfg.code_facts else None  # code_facts: turns that changed nothing
        sampled_before_notes = None  # sampled nodes before the notes request (train_on_notes = False)
        observation = opening
        async with agent.interaction(task) as interaction:
            segment = await interaction.turn(opening)
            while not segment.terminated:
                if deadline is not None and time.time() >= deadline:
                    end = "time_limit"
                    break
                if cfg.episode_deadline is not None and time.time() >= cfg.episode_deadline:
                    end = "deadline"
                    break
                turns += 1
                indices, failure, dropped = glyph.parse_actions(segment.last_reply, max_actions)
                if not indices and cfg.action_from_reasoning:
                    tail = glyph.trailing_actions(reply_parts(segment, interaction.trace)[1])
                    if tail:
                        rescued = glyph.parse_actions(tail, max_actions)
                        if rescued[0]:
                            indices, failure, dropped = rescued
                            from_reasoning += 1
                if adherence is not None:
                    adherence.reply(*reply_parts(segment, interaction.trace)[:2])
                if not indices:
                    forfeits[failure] += 1
                    tracker.update(classify(*reply_parts(segment, interaction.trace), failure))
                else:
                    tracker.update(None)
                if not indices and wasted is not None:
                    wasted.forfeit("no <action> tag" if failure == "no_action_tag" else "an unknown action name")
                if not indices and cfg.forfeit_mode == "freeze":
                    done = game.forfeit()
                    per_turn[0] += 1
                    if feedback:
                        why = "no <action> tag" if failure == "no_action_tag" else "an unknown action name"
                        game.message = f"Your last reply had {why}, so the turn was forfeited: the world did not move."
                        if example:
                            game.message += f" End your reply with one line such as <action>{glyph.EXAMPLE_ACTION}</action>."
                else:
                    counted = bool(indices)  # a forfeit's NOOP is already counted as a forfeit
                    indices = indices or [glyph.NOOP]
                    lines, reward_sum, unlocked, played = [], 0.0, [], 0
                    for index in indices:
                        played += 1
                        actions[glyph.ACTION_NAMES[index]] += 1
                        logged = segmented and _logs_feedback(index)
                        before = game.snapshot() if feedback or logged or wasted is not None else None
                        floor_before = game.floor
                        ahead = game.block_ahead() if segmented and index in _PLACE_STATIONS else None
                        # The cell a DO hits, to tell the agent's own DO from an arrow when a station goes.
                        do_cell = game.block_ahead()[0] if segmented and index == _DO and stations else None
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
                        if wasted is not None and counted:
                            after = game.snapshot()
                            wasted.observe(glyph.ACTION_NAMES[index], game.feedback(index, before, after), before, after)
                        if segmented:
                            for name in step.unlocked:
                                timeline.add(game.turn, f"unlocked {name}")
                            if game.floor != floor_before:
                                timeline.add(game.turn, f"moved to floor {game.floor} ({floor_name(game.floor)}), now at {glyph.place(game.floor, game.position)}")
                            elif logged:
                                text = game.feedback(index, before, game.snapshot())
                                if glyph.ACTION_NAMES[index] != "DO" or _notable_do(text):
                                    timeline.add(game.turn, text.rstrip("."))
                            if ahead is not None:
                                (cell, block_before), (kind, block) = ahead, _PLACE_STATIONS[index]
                                now = int(np.asarray(game.state.map[floor_before])[cell]) if block_before >= 0 else -1
                                if now == block and block_before != block:
                                    stations.append({"kind": kind, "floor": floor_before, "pos": [int(cell[0]), int(cell[1])], "turn": game.turn})
                                    timeline.add(game.turn, f"placed a {kind} at {glyph.place(floor_before, cell)}")
                            if stations and any(s["floor"] == floor_before for s in stations):
                                standing = {id(s) for s in standing_stations(game, stations, floor=floor_before)}
                                for s in [s for s in stations if s["floor"] == floor_before and id(s) not in standing]:
                                    stations.remove(s)
                                    how = "your DO facing it" if do_cell is not None and tuple(s["pos"]) == tuple(do_cell) else "an arrow or another projectile"
                                    timeline.add(game.turn, f"the {s['kind']} at {glyph.place(s['floor'], s['pos'])} was destroyed by {how} (it is path now)")
                            if index == _SLEEP and step.fast_forward_ticks:
                                timeline.add(game.turn, f"slept {step.fast_forward_ticks} ticks at {glyph.place(game.floor, game.position)}")
                            floor, mask = game.visible()
                            explored[floor] |= mask
                        if feedback:
                            note = f" Fast-forwarded {step.fast_forward_ticks} sleep/rest ticks." if step.fast_forward_ticks else ""
                            lines.append(game.feedback(index, before, game.snapshot()) + note)
                        if done:
                            break
                    per_turn[played] += 1
                    if len(indices) > 1 or feedback or dropped:
                        head = (f"Played {played} of {len(indices)} actions" if played < len(indices) else
                                f"Played {played} action{'s' if played != 1 else ''}") + f", reward {reward_sum:+.3f}."
                        body = " ".join(f"{k + 1}. {line}" for k, line in enumerate(lines)) if feedback else (
                            "Actions: " + ", ".join(glyph.ACTION_NAMES[i] for i in indices[:played]) + ".")
                        game.message = " ".join(filter(None, [
                            head, body,
                            f"New achievements: {', '.join(unlocked)}." if unlocked else "",
                            f"{dropped} more action tag{'s were' if dropped != 1 else ' was'} not played "
                            f"(at most {max_actions} per reply, and a run stops at an unknown name)." if dropped else "",
                        ]))
                died = game.health <= 0
                if done and died and not game.out_of_turns and cfg.on_death == "respawn":
                    where = {"turn": game.turn, "floor": game.floor, "pos": list(game.position)}
                    safe_radius = cfg.respawn_safe_radius if cfg.respawn_safe else None
                    at = await asyncio.to_thread(game.respawn, cfg.death_inventory, safe_radius)
                    # a hostile within 8 tiles of the respawn cell, whatever the strategy (as crafter-env logs it)
                    deaths.append({**where, "respawn": list(at), "respawn_hostile": game.hostile_near(at, 8)})
                    if segmented:
                        timeline.add(where["turn"], f"died at {glyph.place(where['floor'], where['pos'])}; respawned at {glyph.place(0, at)}")
                    if adherence is not None:
                        adherence.died(where["turn"], glyph.place(where["floor"], where["pos"]))
                    lost = {
                        "drop_all": "lost your inventory",
                        "drop_materials": "lost your materials (tools and armour kept)",
                        "keep": "kept your inventory",
                    }
                    game.message = (
                        f"{game.message} You died (death {len(deaths)}) at {glyph.place(where['floor'], where['pos'])}. "
                        f"You respawned on the Overworld at {glyph.place(0, at)} with full health, food, drink, energy and "
                        f"mana, and {lost[cfg.death_inventory]}. Unlocked achievements stay unlocked."
                    )
                    done = False
                if adherence is not None:
                    adherence.observe(game, unlocked_at)
                if done:
                    if died:
                        end = "died"
                    elif game.out_of_turns:
                        end = "max_episode_steps"  # the game's clock also counts fast-forwarded sleep
                    else:
                        end = "won"
                    break
                if nav is not None:
                    await asyncio.to_thread(observe_game, nav, game)
                observation = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions, True, example, aid())
                # The cut comes before the next action is sampled: the last observation goes
                # to the model with the notes request instead.
                if segmented and cfg.segment_turns is not None and turns >= cfg.segment_turns:
                    end = "segment_turns"
                    break
                if segmented and cfg.segment_tokens is not None and interaction.trace.num_total_tokens >= cfg.segment_tokens:
                    end = "segment_tokens"
                    break
                segment = await interaction.turn(observation)
            continues = segmented and end in {"segment_turns", "segment_tokens"}
            if continues:
                shown = await asyncio.to_thread(glyph.user_turn, game, reasoning, max_actions, False)
                # The observation first, the request last: what the model reads last is what it answers.
                # NOTES_OPENER first (the system prompt's NOTES_PROTOCOL names it), the observation,
                # then the request last: what the model reads last is what it answers.
                # The HUD shows only a count: name the unlocked ones, so the PLAN does not repeat them.
                unlocked = ", ".join(sorted(game.achievements)) or "none"
                if cfg.code_facts:  # the game's facts sit between the observation and the request
                    facts = await asyncio.to_thread(
                        lambda: (f"{world.known_places(game, explored)}\n\n{exploration_aid(nav, game.floor)}\n\n"
                                 f"{wasted.text(turns)}\n\n{event_log(timeline.lines())}"))
                    message = (f"{NOTES_OPENER}\n\nThe state at the end of the segment:\n\n{shown}\n\n{facts}\n\n"
                               f"[Achievements already unlocked]\n{unlocked}\n\n{cfg.summary_prompt or PLAN_PROMPT}")
                else:
                    message = (f"{NOTES_OPENER}\n\nThe state at the end of the segment:\n\n{shown}\n\n"
                               f"[Achievements already unlocked]\n{unlocked}\n\n{cfg.summary_prompt or SUMMARY_PROMPT}")

                def notes_cap() -> int | None:
                    # summary_max_tokens, cut to what the context still holds: SGLang refuses a
                    # request whose logical length plus max_new_tokens passes the model's limit.
                    limit = cfg.player.max_total_tokens
                    if limit is None:
                        return cfg.summary_max_tokens
                    room = limit - interaction.trace.num_total_tokens - 1024  # the request message itself
                    return max(1, min(cfg.summary_max_tokens or room, room))

                sampled_before_notes = sum(1 for n in (getattr(interaction.trace, "nodes", None) or []) if getattr(n, "sampled", False))
                async with _turn_sampling(interaction, notes_cap()):
                    reply = await interaction.turn(message)
                if not reply.terminated:
                    content, thinking, _ = reply_parts(reply, interaction.trace)
                    notes = extract_notes(content, thinking)
                    retry_reason = notes_problem(content, thinking, notes)
                    cap = notes_cap()
                    if cfg.summary_retry and retry_reason is not None and (cap is None or cap >= 256):
                        notes_retried = True
                        async with _turn_sampling(interaction, cap):
                            reply = await interaction.turn(PLAN_RETRY if cfg.code_facts else SUMMARY_RETRY)
                        if not reply.terminated:
                            content, thinking, _ = reply_parts(reply, interaction.trace)
                            notes = extract_notes(content, thinking) or notes
                if reply.terminated:
                    end = "terminated_at_summary"
        trace = interaction.trace
        notes_masked = {"turn": 0, "mid": 0, "tokens": 0}
        if segmented and not cfg.train_on_notes:
            notes_masked = mask_notes_from_loss(trace, sampled_before_notes if sampled_before_notes is not None else 10**9)
        elif segmented:  # per-term routing: the notes term trains the notes turn, the others the action turns
            request_term_routing(trace, {"notes": notes_node_indexes(trace, sampled_before_notes if sampled_before_notes is not None else 10**9)})
        trace_id = getattr(trace, "id", None) or uuid.uuid4().hex
        depth = int(node.get("depth", 0)) + 1 if segmented else 0
        record = summary = state_path = state_sha = None
        state_bytes = 0
        if continues:
            record = game_record(game, stations=stations, record_stations=cfg.record_stations, depth=depth,
                                 first_turn=first_turn, deaths=deaths, death_inventory=cfg.death_inventory)
            summary = summary_text(record, notes, facts)
            saved = {"game": game.save(), "explored": explored, "stations": stations, "seed": seed, "depth": depth, "nav": nav}
            state_path, state_sha, state_bytes = await asyncio.to_thread(store.put, trace_id, saved)
        sections = parse_notes(notes or "")
        events = timeline.lines()
        judge_inputs: dict[str, Any] = {}
        if segmented:
            view = (game.floor, *map(int, game.position))  # where the notes turn's [Grid] was centred
            claims = world.check_map_claims(game, sections["MAP"], explored, tolerance=cfg.map_claim_tolerance, view=view)
            judge_inputs = {
                "record_before": record_before,
                "map_ascii": world.ascii_map(game, explored),
                "nearest": world.nearest(game, explored, cfg.judge_world_radius),
                "next_step": world.next_step(game, stations),
                "map_claims": "(MAP written by the game: not graded)" if cfg.code_facts else claims.text(),
                "notes_view": world.view_note(view),
                "map_check": claims.score,
                "map_claim_counts": {"correct": claims.correct, "wrong": claims.wrong, "invented": claims.invented,
                                     "unchecked": claims.unchecked},
            }
            if cfg.code_facts:
                check = check_plan(game, explored, sections["PLAN"], view=view)
                judge_inputs.update({"plan_feasible": check.feasible, "plan_on_path": check.on_path,
                                     "plan_on_track": check.on_track, "plan_target": check.target,
                                     "critical_path": check.critical, "plan_check": check.text(),
                                     "notes_judge": cfg.notes_judge, "judge_weight": cfg.judge_weight,
                                     "false_finding_penalty": cfg.false_finding_penalty,
                                     "findings_weight": cfg.findings_weight, "findings_count": count_findings(sections["FINDINGS"]),
                                     "turns_lost": wasted.as_dict() if wasted is not None else {},
                                     "turns_lost_total": wasted.total if wasted is not None else 0})
        achievements = sorted(game.achievements)
        delta = game.achievement_score - score_before  # Craftax's 1/3/5/8 reward of this segment's unlocks
        score = total if cfg.reward == "return" else game.achievement_score
        state_points, state_parts = (world.state_value(game, explored) if segmented else (0.0, {}))
        if segmented:
            reward = delta
            trace.record_reward("progress", reward, 1.0)
            if cfg.state_weight > 0:
                trace.record_reward("state", state_points, cfg.state_weight)
        else:
            reward = score / glyph.MAX_RETURN - cfg.death_penalty * len(deaths)
            trace.record_reward(f"craftax_{cfg.reward}", reward, 1.0)
        trace.record_metrics(
            {
                "achievements_unlocked": float(len(achievements)),
                "total_env_reward": total,
                "achievement_score": game.achievement_score,
                "deaths": float(len(deaths)),
                "forfeits": float(sum(forfeits.values())),
                "actions_from_reasoning": float(from_reasoning),
                "survived": 1.0 if game.health > 0 else 0.0,
                "max_floor": float(max_floor),
                "turns": float(turns),
                "actions": float(sum(actions.values())),
                "game_steps": float(game.turn),
                **tracker.metrics(),
                **({
                    "depth": float(depth),
                    "continues": float(continues),
                    "achievement_reward_delta": float(delta),
                    "state_value": float(state_points),
                    "segment_achievements": float(len(set(achievements) - set(achievements_before))),
                    "notes_chars": float(len(notes or "")),
                    "notes_masked_replies": float(notes_masked["turn"] + notes_masked["mid"]),
                    "notes_masked_mid": float(notes_masked["mid"]),
                    "notes_masked_tokens": float(notes_masked["tokens"]),
                    "notes_retried": float(notes_retried),
                    **{f"notes_retry_{r}": float(notes_retried and retry_reason == r) for r in NOTES_RETRY_REASONS},
                    "notes_format": notes_format(sections, PLAN_SECTIONS if cfg.code_facts else SECTIONS),
                    "explored_cells": float(explored.sum()),
                    "map_check": float(judge_inputs.get("map_check", 0.0)),
                    "map_claims_correct": float(judge_inputs["map_claim_counts"]["correct"]),
                    "map_claims_invented": float(judge_inputs["map_claim_counts"]["invented"]),
                } if segmented else {}),
            }
        )
        info = {
            "achievements": achievements,
            "unlocked_at": unlocked_at,
            "turns": turns,
            "actions": sum(actions.values()),
            "game_steps": game.turn,
            "ended_by": end,
            "seed": seed,
            "max_floor": max_floor,
            "deaths": deaths,
        }
        if segmented:
            info.update({
                # the segment's place in the search (what the tree sampler reads)
                "node_id": trace_id, "parent": node.get("id"), "depth": depth, "branch": node.get("branch"),
                "continues": continues, "final_goal_reached": end == "won",
                "state_path": state_path, "state_sha256": state_sha, "state_bytes": state_bytes,
                "summary": summary, "record": record, "notes": notes, "notes_sections": sections,
                "notes_retry_reason": retry_reason if notes_retried else None,
                "notes_masked": notes_masked, "train_on_notes": cfg.train_on_notes,
                "code_facts": cfg.code_facts, "facts": facts,
                # what the judges compare against (docs/craftax-segments-judge.md)
                "first_turn": first_turn, "achievements_before": achievements_before,
                "achievement_reward_delta": delta, "achievement_score_before": float(score_before),
                "state_value": float(state_points), "state_parts": state_parts,
                "events": events[:80], "stations": stations,
                "floor": game.floor, "position": list(game.position), "actions_hist": dict(actions),
                "forfeit_kinds": dict(tracker.kinds), "ended_at": time.time(), **judge_inputs,
                "adherence_digest": adherence.digest() if adherence is not None else None,
                "parent_plan_score": parent_plan_score,
            })
        trace.info["craftax"] = info
        if cfg.episode_log:
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
                "actions_from_reasoning": from_reasoning,
                **tracker.log(),
                "actions": dict(actions),
                "ended_by": end,
                "survived": game.health > 0,
                "max_floor": max_floor,
                "seed": seed,
                "trace_tokens": trace.num_total_tokens,
                "task": "craftax",
            }
            if segmented:
                line.update({k: info[k] for k in (
                    "node_id", "parent", "depth", "branch", "continues", "final_goal_reached", "state_path", "first_turn",
                    "achievements_before", "achievement_reward_delta", "events", "stations", "floor", "position",
                    "notes", "record", "notes_retry_reason", "record_before", "map_ascii", "nearest", "next_step",
                    "map_claims", "map_check", "map_claim_counts", "actions_hist", "forfeit_kinds", "adherence_digest",
                    "parent_plan_score", "code_facts", "facts", "plan_feasible", "plan_on_path", "plan_on_track",
                    "plan_target", "critical_path", "plan_check", "findings_count", "turns_lost", "turns_lost_total",
                    "achievement_score_before", "state_value", "state_parts") if k in info}
                    | {"state_bytes": state_bytes, "explored_cells": int(explored.sum())})
            with open(cfg.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")
        out = {**info, "trace": trace, "reward": float(getattr(trace, "reward", reward))}
        if segmented:
            out["beam_score"] = self._beam_score(out)
        return out


class CraftaxTaskset(vf.Taskset[CraftaxTask, CraftaxConfig]):
    def load(self) -> Iterator[CraftaxTask]:
        cfg = self.config
        base = cfg.seed + (1_000_000 if cfg.split == "eval" else 0)
        system = glyph.system_prompt(cfg.enable_reasoning, cfg.max_output_tokens, cfg.forfeit_mode, cfg.respawn, cfg.max_actions_per_turn,
                                     action_example=cfg.action_example)
        for i in range(cfg.num_examples):
            # Prompt-less: the env opens the episode with the first observation of the
            # world it builds, so loading stays instant.
            yield CraftaxTask(
                CraftaxData(
                    name=f"craftax#{cfg.split}{i}",
                    prompt=None,
                    system_prompt=system,
                    info={
                        "seed": base + i, "episode_idx": i, "split": cfg.split, "reasoning": cfg.enable_reasoning,
                        "max_actions": cfg.max_actions_per_turn,
                        "action_example": cfg.action_example,
                        # what the tree sampler reads (samplers/tree.py): where the segment's info is and
                        # how a child's prompt introduces the summary, so it matches continuation_prompt
                        "segments": {"info_key": "craftax", "continuation_header": CONTINUATION_HEADER,
                                     "empty_summary": "(no record or notes were written)"},
                    },
                ),
                cfg.task,
            )

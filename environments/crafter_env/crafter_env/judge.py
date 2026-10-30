"""The segment judge: one absolute score per branch, from an OpenAI-compatible
endpoint or GitHub Copilot CAPI (selected by ``judge.backend``).

The judge sees where the segment started (inventory and achievements the lineage
had), what happened in it (unlocks, deaths, tables, sleep, exploration), the
ground truth at the cut (vitals, inventory, achievements, the local map), facts the
policy could not see (the nearest resources within `judge_world_radius` tiles) and
Crafter's own next tech-tree step, next to the notes the policy wrote. It returns
three sub-scores in [0, 1]: `progress` (what this segment moved along the tech tree
and toward survival), `plan` (whether the notes' plan is the right, feasible next
step) and `consistency` (whether the notes match the ground truth). The env mixes
them with `judge_mix` into the `judge` reward.

The judge scores branches one at a time, in ``finalize`` (the RL pipeline) or right
after a branch closes (the eval driver's in-env beam); the comparison between the
branches of one node is GRPO's group baseline, so an absolute score already yields a
relative advantage.

A failure after the SDK's retries falls back to the segment's achievement delta and
is recorded as ``judge_failed`` on the trace, so a segment is never replayed because
the judge was down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, TypeVar

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from crafter_env.capi import CrafterJudgeConfig, capi_complete

logger = logging.getLogger(__name__)

# --- what the judge costs ---------------------------------------------------------
# USD per 1M tokens (input, output incl. thinking, cached input), from
# https://ai.google.dev/gemini-api/docs/pricing on 2026-10-25. gemini-3.8-flash doubles
# on 2027-01-01. Override per run with env.judge_price.
PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
}


def _patch_usage() -> None:
    """Gemini's OpenAI-compatible endpoint leaves thinking tokens out of
    `completion_tokens` (they appear only in `total_tokens`) but bills them as output.
    Fold them into the completion, as reasoning tokens are everywhere else, so every
    `Usage` (and so the trace's `extra_usage`) counts what is billed."""
    from verifiers.v1.types import Usage

    if getattr(Usage.from_openai, "_crafter_thinking", False):
        return
    original = Usage.from_openai.__func__

    def from_openai(cls, usage):
        out = original(cls, usage)
        if out is None or usage is None or getattr(usage, "completion_tokens_details", None) is not None:
            return out
        total = getattr(usage, "total_tokens", None) or 0
        hidden = total - (usage.prompt_tokens or 0) - (usage.completion_tokens or 0)
        if hidden > 0:
            out = out.model_copy(update={"completion_tokens": out.completion_tokens + hidden, "reasoning_tokens": hidden})
        return out

    from_openai._crafter_thinking = True
    Usage.from_openai = classmethod(from_openai)


_patch_usage()


def usage_cost(
    usage: Any, model: str, price: tuple[float, float, float] | None = None, *, backend: str = "openai"
) -> dict[str, float]:
    """Billed tokens and USD of one call (zeros when the provider returned no usage, e.g. a 4xx)."""
    if usage is None:
        return {"input_tokens": 0.0, "output_tokens": 0.0, "thinking_tokens": 0.0, "cost_usd": 0.0}
    p_in, p_out, p_cached = price or (
        PRICES.get(model.split("/")[-1], (float("nan"),) * 3) if backend == "openai" else (float("nan"),) * 3
    )
    cached = float(usage.cached_input_tokens or 0)
    inp, out = float(usage.prompt_tokens), float(usage.completion_tokens)
    return {"input_tokens": inp + cached, "output_tokens": out, "thinking_tokens": float(usage.reasoning_tokens or 0),
            "cost_usd": (inp * p_in + cached * p_cached + out * p_out) / 1e6}


def scale_cost(cost: dict[str, float], share: float) -> dict[str, float]:
    return {k: v * share for k, v in cost.items()}

CRAFTER_GUIDE = """\
Crafter is a 2D survival and crafting game on a 64x64 map. The player has health, \
food, drink and energy (each 0 to 9); food comes from cows and plants, drink from \
water, energy from sleeping; a vital at 0 drains health. Zombies spawn on grass at \
night, skeletons live in caves and shoot arrows, lava kills on contact. The 22 \
achievements: collect_wood, place_table, eat_cow, collect_sapling, collect_drink, \
make_wood_pickaxe, make_wood_sword, place_plant, defeat_zombie, collect_stone, \
place_stone, eat_plant, defeat_skeleton, make_stone_pickaxe, make_stone_sword, \
wake_up, place_furnace, collect_coal, collect_iron, collect_diamond, \
make_iron_pickaxe, make_iron_sword.

The tech tree to the diamond, in order: collect wood from trees (`do` facing a \
tree); place a table (2 wood); make a wood pickaxe at the table (1 wood); collect \
stone with the wood pickaxe; make a stone pickaxe at the table (1 wood, 1 stone); \
place a furnace (4 stone); collect coal (wood pickaxe) and iron (stone pickaxe); \
make an iron pickaxe at a table with a furnace nearby (1 wood, 1 coal, 1 iron); \
collect the diamond with the iron pickaxe. Coal, iron and diamonds \
sit in mountain and cave areas (stone, paths, darkness), which also hold skeletons \
and lava. Water and cows are on grassland.

The game continues after a death in these runs: the player respawns with full \
vitals at its last sleeping place, its base or the spawn, keeping tools and \
achievements but losing materials. Placed tables and furnaces stay in the world."""

JUDGE_PROMPT = """\
You are grading one segment of a Crafter game played by a language-model agent. The \
agent plays in segments of a few hundred turns; at the end of a segment it writes \
notes, and the next segment starts from the saved world with only those notes as \
memory. Your scores train the agent, so grade what actually happened and whether \
the notes set up the next segment well, not the prose.

{guide}

The final goal of the whole game is to unlock {goal}.

== Where this segment started (segment {depth} of the game, game step {game_step_before}) ==
achievements already unlocked: {achievements_before}
inventory at the start: {inventory_before}
position at the start: {position_before}

== What happened in this segment ({steps} turns) ==
{events}
distinct cells visited: {cells_visited}; invalid actions: {invalid_actions}; replies without an action: {malformed_replies}

== Ground truth at the end of the segment (trust this over the notes) ==
{state}

== Facts the agent could not see (nearest resources within {world_radius} tiles, offsets from the player) ==
{world_nearby}

== The tech tree's own next step, from the game's recipes ==
{next_tier}

== The agent's notes and plan for the next segment ==
{summary}

Give three scores from 0 to 1 and a one-line rationale.
- progress: what this segment moved along the tech tree and toward the goal, given \
where it started. 0.0 to 0.2: nothing new, or wandering and dying; 0.3 to 0.5: a \
new resource, a table, food and drink secured, or useful exploration toward a \
needed resource; 0.6 to 0.8: a new tier (first pickaxe, stone tools, furnace, coal \
and iron); 0.9 to 1.0: the iron pickaxe is made and a diamond is located, or the \
diamond is collected. A segment that repeats an already-unlocked tier earns little.
- plan: whether the notes name the right next step (compare with the tech tree's \
next step and the nearby resources) with concrete, feasible actions and locations, \
and handle survival (night, food, drink). Generic advice, or a plan that \
contradicts the inventory, the achievements or the known surroundings, scores low.
- consistency: whether the notes' inventory, achievements, position and known \
locations match the ground truth. Invented tools, resources or achievements score \
low; omissions cost less than fabrications."""


class JudgeVerdict(BaseModel):
    progress: float = Field(ge=0.0, le=1.0)
    plan: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


VerdictT = TypeVar("VerdictT")


class _CrafterJudge(vf.Judge[VerdictT, CrafterJudgeConfig]):
    async def complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
        if getattr(self.config, "backend", "openai") == "capi":
            return await capi_complete(
                self.config, self.reward_name, messages, trace=trace, schema=schema, parse=parse, **sampling
            )
        return await super().complete(messages, trace=trace, schema=schema, parse=parse, **sampling)


class CrafterJudge(_CrafterJudge[JudgeVerdict]):
    prompt = JUDGE_PROMPT
    schema = JudgeVerdict

    def build_messages(self, **fields: Any) -> str:
        return super().build_messages(guide=CRAFTER_GUIDE, **fields)


def judge_state_text(info: dict[str, Any]) -> str:
    """The ground truth at the cut, from the trace's ``crafter`` info."""
    inventory = info.get("inventory") or {}
    vitals = {k: inventory.get(k, 0) for k in ("health", "food", "drink", "energy")}
    items = {k: v for k, v in inventory.items() if k not in vitals and v}
    lines = [
        f"game step: {info.get('game_step')}",
        f"vitals: {json.dumps(vitals)}",
        f"inventory: {json.dumps(items) if items else 'empty'}",
        f"achievements unlocked so far ({len(info.get('achievements') or [])}/22): {', '.join(info.get('achievements') or []) or 'none'}",
        f"unlocked for the first time this segment: {', '.join(info.get('segment_achievements') or []) or 'none'}",
        f"deaths this segment: {len(info.get('deaths') or [])}",
        f"position: {info.get('position')}",
    ]
    if info.get("local_map"):
        lines += ["local map at the cut (north is up; @ player, T tree, s stone, c coal, i iron, d diamond, B table, F furnace, ~ water, ^ lava, Z zombie, S skeleton, C cow, P plant):", info["local_map"]]
    if info.get("final_goal_reached"):
        lines.append("THE FINAL GOAL WAS REACHED THIS SEGMENT.")
    return "\n".join(lines)


def judge_fields(info: dict[str, Any], goal: str, world_radius: int | None = None) -> dict[str, Any]:
    """The prompt's fields from the trace's ``crafter`` info (also what the replay script feeds)."""
    before = info.get("inventory_before") or {}
    steps = int(info.get("steps") or 0)
    return {
        "goal": goal,
        "depth": info.get("depth", 1),
        "game_step_before": int(info.get("game_step") or 0) - steps,
        "achievements_before": ", ".join(info.get("achievements_before") or []) or "none",
        "inventory_before": json.dumps(before) if before else "empty",
        "position_before": info.get("position_before"),
        "steps": steps,
        "events": "\n".join(info.get("events") or []) or "(no unlock, death, table, furnace or sleep)",
        "cells_visited": info.get("cells_visited", "?"),
        "invalid_actions": info.get("invalid_actions", 0),
        "malformed_replies": info.get("malformed_replies", 0),
        "state": judge_state_text(info),
        "world_radius": world_radius if world_radius is not None else "?",
        "world_nearby": info.get("world_nearby") or "unknown",
        "next_tier": info.get("next_tier") or "unknown",
        "summary": info.get("summary") or "(the agent wrote no notes)",
    }


def mix_score(verdict: JudgeVerdict, mix: tuple[float, float, float]) -> float:
    total = sum(mix) or 1.0
    score = (mix[0] * verdict.progress + mix[1] * verdict.plan + mix[2] * verdict.consistency) / total
    return float(min(1.0, max(0.0, score)))


# Crafter's tech ladder toward the diamond, by the level each achievement reaches (the
# survival achievements - drink, sapling, plant, cow, sleep, zombie, skeleton - are level 0).
TECH_LADDER = {
    "collect_wood": 1, "place_table": 2, "make_wood_pickaxe": 3, "make_wood_sword": 3,
    "collect_stone": 4, "place_stone": 4, "collect_coal": 4,
    "make_stone_pickaxe": 5, "make_stone_sword": 5, "place_furnace": 5,
    "collect_iron": 6, "make_iron_pickaxe": 7, "make_iron_sword": 7, "collect_diamond": 8,
}
TECH_LEVELS = max(TECH_LADDER.values())


def tech_tier(achievements) -> float:
    """The deepest ladder level the lineage has unlocked so far, as a fraction of the ladder."""
    return max((TECH_LADDER.get(a, 0) for a in achievements or ()), default=0) / TECH_LEVELS


def with_tier(verdict: dict[str, Any], info: dict[str, Any], tier_weight: float) -> dict[str, Any]:
    """Blend the lineage's cumulative tier into a verdict's score: (1 - w) x the judge's
    score + w x tier. The judge's own score stays as `llm_score`. Off at weight 0.
    The in-env beam ranks branches by this score, so a plateaued advanced lineage no
    longer loses to a beginner's first table (job 11420870 dropped its 14-achievement
    lineage at depth 7 for 6-achievement branches that had just placed a table)."""
    if tier_weight <= 0:
        return verdict
    tier = tech_tier(info.get("achievements") or [])
    llm = float(verdict["score"])
    return {**verdict, "llm_score": llm, "tier": tier, "score": float(min(1.0, max(0.0, (1 - tier_weight) * llm + tier_weight * tier)))}


async def judge_segment(
    judge: CrafterJudge,
    trace: Any,
    *,
    goal: str,
    weight: float,
    mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
    world_radius: int | None = None,
    gate: asyncio.Semaphore | None = None,
    episode_log: str | None = None,
    price: tuple[float, float, float] | None = None,
    tier_weight: float = 0.0,
) -> float:
    """Score one finished segment alone and record it on its trace as the ``judge``
    reward, with `judge_wait_s` (segment end to score: queue + call) and
    `judge_call_s`. Falls back to the segment's achievement delta on a failure.
    `tier_weight` > 0 blends the lineage's cumulative tech tier into the score (`with_tier`)."""
    info = trace.info.get("crafter") or {}
    delta = info.get("segment_achievements") or []
    fallback = len(delta) / 22.0
    queued = t0 = time.time()
    verdict: dict[str, Any]
    cost = usage_cost(None, judge.config.model)
    try:
        async with gate or _null():
            t0 = time.time()
            response = await judge.evaluate(trace=trace, **judge_fields(info, goal, world_radius))
        call_s = time.time() - t0
        cost = usage_cost(
            getattr(response, "usage", None), judge.config.model, price, backend=getattr(judge.config, "backend", "openai")
        )
        parsed = response.parsed
        if parsed is None:
            raise RuntimeError("judge returned no verdict")
        verdict = {"score": mix_score(parsed, mix), "progress": parsed.progress, "plan": parsed.plan,
                   "consistency": parsed.consistency, "rationale": parsed.rationale, "failed": False}
    except Exception as exc:  # the segment is ten minutes of sampling: never fail the episode on the judge
        call_s = time.time() - queued
        logger.warning("judge failed (%s); falling back to the achievement delta %.3f", exc, fallback)
        verdict = {"score": fallback, "rationale": f"judge failed: {exc}", "failed": True}
    verdict = with_tier(verdict, info, tier_weight)
    ended = float(info.get("ended_at") or queued)
    timing = {"wait_s": time.time() - ended, "queue_s": max(0.0, t0 - queued), "call_s": call_s}
    return apply_verdict(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="single", cost=cost)


async def judge_segment_group(
    rendezvous: GroupRendezvous,
    judge: CrafterJudge,
    group_judge: CrafterGroupJudge,
    trace: Any,
    *,
    goal: str,
    weight: float,
    mix: tuple[float, float, float],
    world_radius: int | None,
    gate: asyncio.Semaphore | None,
    episode_log: str | None,
    price: tuple[float, float, float] | None = None,
    tier_weight: float = 0.0,
) -> float:
    """Group mode in the RL pipeline: meet the node's other branches, take this
    branch's verdict from the one comparative call, or fall back to judging alone."""
    info = trace.info.get("crafter") or {}
    key, member = str(info.get("parent")), str(info.get("node_id"))

    async def call(infos):
        costs: list = []
        out, call_s, err = await group_verdicts(group_judge, infos, goal=goal, mix=mix, world_radius=world_radius, trace=trace,
                                                gate=gate, price=price, costs=costs, tier_weight=tier_weight)
        return out, call_s, err, (costs[0] if costs else None)

    verdict, timing = await rendezvous.meet(key, member, {k: v for k, v in info.items() if k != "judge"}, call)
    cost = timing.pop("_cost", None) or usage_cost(None, group_judge.config.model)
    ended = float(info.get("ended_at") or time.time())
    if verdict is None:  # a sibling never arrived, or the group call failed: judge this branch alone
        score = await judge_segment(judge, trace, goal=goal, weight=weight, mix=mix, world_radius=world_radius, gate=gate, episode_log=None, price=price, tier_weight=tier_weight)
        single = trace.info["crafter"]["judge"]
        cost = {k: cost[k] + float(single.get(k, 0.0)) for k in cost}
        verdict = {k: single[k] for k in ("score", "progress", "plan", "consistency", "rationale", "failed", "llm_score", "tier") if k in single}
        timing = {**timing, "call_s": timing.get("call_s", 0.0) + single.get("call_s", 0.0), "group_fallback": 1.0}
    else:
        timing = {**timing, "group_fallback": 0.0}
    timing["wait_s"] = time.time() - ended
    return apply_verdict(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="group", cost=cost)


class _null:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


# --- group mode: the branches of one node in one comparative call -------------------

GROUP_PROMPT = """\
You are grading {n} branches of one Crafter game played by a language-model agent. \
All branches started from the same saved world and the same notes, and each played \
one segment of a few hundred turns; at the end of a segment the agent writes notes, \
and the next segment starts from the saved world with only those notes as memory. \
Your scores train the agent by comparing the branches with each other, so grade \
what actually happened in each branch and whether its notes set up the next segment \
well, not the prose. Use the whole 0 to 1 range when the branches differ, and give \
equal scores only when they are equally good.

{guide}

The final goal of the whole game is to unlock {goal}.

== Where every branch started (segment {depth} of the game, game step {game_step_before}) ==
achievements already unlocked: {achievements_before}
inventory at the start: {inventory_before}
position at the start: {position_before}

{branches}

For every branch give three scores from 0 to 1 and a one-line rationale, with the \
same rubric as for a single segment:
- progress: what the branch moved along the tech tree and toward the goal, given the \
shared start. 0.0 to 0.2 nothing new or wandering and dying; 0.3 to 0.5 a new resource, \
a table, food and drink secured, or useful exploration toward a needed resource; 0.6 to \
0.8 a new tier (first pickaxe, stone tools, furnace, coal and iron); 0.9 to 1.0 the iron \
pickaxe and a located diamond, or the diamond.
- plan: whether its notes name the right next step (compare with its tech tree's next \
step and its nearby resources) with concrete, feasible actions and locations, and \
handle survival.
- consistency: whether its notes match its own ground truth; fabrications cost more \
than omissions.
Return one verdict per branch, with the branch number as given."""

BRANCH_BLOCK = """\
== Branch {branch} ==
-- what happened ({steps} turns) --
{events}
distinct cells visited: {cells_visited}; invalid actions: {invalid_actions}; replies without an action: {malformed_replies}
-- ground truth at the end (trust this over the notes) --
{state}
-- facts the agent could not see (nearest resources within {world_radius} tiles) --
{world_nearby}
-- the tech tree's own next step --
{next_tier}
-- the agent's notes and plan --
{summary}"""


class BranchVerdict(JudgeVerdict):
    branch: int


class GroupVerdict(BaseModel):
    verdicts: list[BranchVerdict]
    comparison: str = ""


class CrafterGroupJudge(_CrafterJudge[GroupVerdict]):
    prompt = GROUP_PROMPT
    schema = GroupVerdict

    def build_messages(self, *, infos: list[dict[str, Any]], goal: str, world_radius: int | None = None, **_: Any) -> str:
        shared = judge_fields(infos[0], goal, world_radius)
        blocks = []
        for i, info in enumerate(infos):
            f = judge_fields(info, goal, world_radius)
            blocks.append(BRANCH_BLOCK.format(branch=i, **{k: f[k] for k in (
                "steps", "events", "cells_visited", "invalid_actions", "malformed_replies", "state",
                "world_radius", "world_nearby", "next_tier", "summary")}))
        return vf.Judge.build_messages(
            self, guide=CRAFTER_GUIDE, n=len(infos), goal=goal, depth=shared["depth"],
            game_step_before=shared["game_step_before"], achievements_before=shared["achievements_before"],
            inventory_before=shared["inventory_before"], position_before=shared["position_before"],
            branches="\n\n".join(blocks),
        )


async def group_verdicts(
    judge: CrafterGroupJudge,
    infos: list[dict[str, Any]],
    *,
    goal: str,
    mix: tuple[float, float, float],
    world_radius: int | None = None,
    trace: Any = None,
    gate: asyncio.Semaphore | None = None,
    price: tuple[float, float, float] | None = None,
    costs: list | None = None,
    tier_weight: float = 0.0,
) -> tuple[list[dict[str, Any]] | None, float, str | None]:
    """One comparative call over `infos` (the branches of one node). Returns one
    verdict dict per branch in order (None on failure), the call's latency and the error."""
    t0 = time.time()
    try:
        async with gate or _null():
            messages = judge.build_messages(infos=infos, goal=goal, world_radius=world_radius)
            response = await judge.complete(messages, trace=trace, schema=judge.schema, parse=judge.parse)
        if costs is not None:
            costs.append(usage_cost(
                getattr(response, "usage", None), judge.config.model, price, backend=getattr(judge.config, "backend", "openai")
            ))
        parsed: GroupVerdict | None = response.parsed
        by_branch = {v.branch: v for v in (parsed.verdicts if parsed else [])}
        if sorted(by_branch) != list(range(len(infos))):
            raise RuntimeError(f"group verdict covers branches {sorted(by_branch)}, expected 0..{len(infos) - 1}")
        out = [with_tier({"score": mix_score(by_branch[i], mix), "progress": by_branch[i].progress, "plan": by_branch[i].plan,
                          "consistency": by_branch[i].consistency, "rationale": by_branch[i].rationale, "failed": False,
                          "comparison": parsed.comparison}, infos[i], tier_weight) for i in range(len(infos))]
        return out, time.time() - t0, None
    except Exception as exc:
        logger.warning("group judge failed (%s)", exc)
        return None, time.time() - t0, str(exc)


def apply_verdict(
    trace: Any,
    verdict: dict[str, Any],
    *,
    weight: float,
    timing: dict[str, float],
    episode_log: str | None = None,
    mode: str = "single",
    cost: dict[str, float] | None = None,
) -> float:
    """Record a verdict (from either mode) and its waiting times on the trace and in the log."""
    info = trace.info.get("crafter") or {}
    score = float(verdict["score"])
    trace.record_reward("judge", score, weight)
    trace.record_metrics({
        "judge_score": score, "judge_failed": float(verdict["failed"]),
        **({k: float(verdict[k]) for k in ("progress", "plan", "consistency")} if not verdict["failed"] else {}),
        **({"judge_tier": float(verdict["tier"]), "judge_llm_score": float(verdict["llm_score"])} if "tier" in verdict else {}),
        **{f"judge_{k}": float(v) for k, v in timing.items()},
        **{f"judge_{k}": float(v) for k, v in (cost or {}).items()},
    })
    info["judge"] = {**verdict, "mode": mode, **timing, **(cost or {})}
    trace.info["crafter"] = info
    if episode_log:
        line = {"task": "crafter-judge", "node_id": info.get("node_id"), "depth": info.get("depth"), "seed": info.get("seed"),
                "judge_score": score, "judge_failed": verdict["failed"], "rationale": verdict.get("rationale", ""), "at": time.time(),
                "mode": mode, **timing, **(cost or {}), **{k: verdict[k] for k in ("progress", "plan", "consistency", "tier", "llm_score") if k in verdict}}
        from crafter_env.state_store import retry_io

        def write():
            with open(episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")

        retry_io(write, what=f"appending the judge line to {episode_log}")
    return score


class GroupRendezvous:
    """Where the branches of one node meet across env workers (a directory per node).

    Each branch writes its judge inputs as `<dir>/<node>/<member>.json`. The branch
    that completes the set takes the leader lock (an O_EXCL file), makes the one
    comparative call and writes `result.json`; the others poll for it. A branch that
    waits longer than `timeout_s` (a sibling failed or was cancelled) gives up and is
    judged alone. Directories older than two hours are swept by later leaders."""

    def __init__(self, root: str, size: int, timeout_s: float, poll_s: float = 2.0) -> None:
        self.root, self.size, self.timeout_s, self.poll_s = root, size, timeout_s, poll_s
        os.makedirs(root, exist_ok=True)

    @staticmethod
    def _write(path: str, payload: dict[str, Any]) -> None:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, default=str)
        os.replace(tmp, path)

    def _sweep(self) -> None:
        cutoff = time.time() - 7200
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    for f in os.listdir(path):
                        os.remove(os.path.join(path, f))
                    os.rmdir(path)
            except OSError:
                pass

    async def meet(self, key: str, member: str, info: dict[str, Any], call) -> tuple[dict[str, Any] | None, dict[str, float]]:
        """Returns this member's verdict (None: judge alone) and the timing:
        `group_wait_s` (waiting for the siblings and the call), `call_s`, `group_size`, `group_incomplete`."""
        d = os.path.join(self.root, key)
        os.makedirs(d, exist_ok=True)
        self._write(os.path.join(d, f"m_{member}.json"), info)
        arrived = time.time()
        while True:
            result = os.path.join(d, "result.json")
            if os.path.exists(result):
                with open(result) as f:
                    res = json.load(f)
                timing = {"group_wait_s": time.time() - arrived, "call_s": float(res.get("call_s", 0.0)),
                          "group_size": float(len(res.get("members", []))), "group_incomplete": 0.0}
                if member in res.get("members", []) and res.get("cost"):
                    timing["_cost"] = scale_cost(res["cost"], 1.0 / len(res["members"]))
                if res.get("failed") or member not in res.get("verdicts", {}):
                    return None, timing
                return res["verdicts"][member], timing
            members = sorted(f[2:-5] for f in os.listdir(d) if f.startswith("m_") and f.endswith(".json"))
            if len(members) >= self.size:
                try:
                    fd = os.open(os.path.join(d, "leader"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    pass
                else:
                    infos = []
                    for m in members:
                        with open(os.path.join(d, f"m_{m}.json")) as f:
                            infos.append(json.load(f))
                    waited = time.time() - arrived
                    verdicts, call_s, error, cost = await call(infos)
                    res = {"members": members, "call_s": call_s, "failed": verdicts is None, "error": error, "cost": cost,
                           "verdicts": dict(zip(members, verdicts)) if verdicts else {}}
                    self._write(result, res)
                    self._sweep()
                    timing = {"group_wait_s": waited + call_s, "call_s": call_s, "group_size": float(len(members)), "group_incomplete": 0.0}
                    if cost:
                        timing["_cost"] = scale_cost(cost, 1.0 / len(members))
                    return (res["verdicts"].get(member) if verdicts else None), timing
            if time.time() - arrived > self.timeout_s:
                return None, {"group_wait_s": time.time() - arrived, "call_s": 0.0, "group_size": float(len(members)), "group_incomplete": 1.0}
            await asyncio.sleep(self.poll_s)

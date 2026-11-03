"""The judges for Craftax segments (docs/craftax-segments-judge.md): the notes judge
(Milestone 2) and, at the end of this module, the adherence judge (Milestone 3).

The judge grades the notes a segment wrote for the next one; the segment's progress is
scored by code (``progress`` reward). Code also checks the notes' MAP lines against the
cells the agent saw (``map_check``); the judge scores the rest:

- ``findings`` (0-1): each finding is a specific fact the timeline supports;
- ``plan`` (0-1): the first subgoal is the right next step, every subgoal is feasible from
  the record, its places are real places the agent saw, the survival rules fit.

``grounding = 0.6 map_check + 0.4 findings``; ``notes = 0.5 grounding + 0.5 plan``, recorded
as the ``notes`` reward with ``notes_weight``. Single mode grades one segment alone; group
mode grades the branches of one node in one comparative call (they share the start). On a
judge failure the LLM parts fall back to 0.5 each (neutral) and ``judge_failed`` is set, so
a segment is never replayed because the judge was down.

The client, the CAPI backend, the cost accounting and the group rendezvous are crafter_env's.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

import verifiers.v1 as vf

from crafter_env.judge import GroupRendezvous, _CrafterJudge, _null, scale_cost, usage_cost
from crafter_env.state_store import retry_io

logger = logging.getLogger(__name__)

CRAFTAX_GUIDE = """\
Craftax is a 2D survival and crafting roguelike on 9 floors, each a 48x48 map: 0 Overworld, \
1 Dungeon, 2 Gnomish Mines, 3 Sewers, 4 Vaults, 5 Troll Mines, 6 Fire Realm, 7 Ice Realm, \
8 Graveyard, where the necromancer is the final goal. The player sees a 9x11 window and a HUD. \
Every place the game writes below is F<floor> (row,col): the HUD's Floor and Pos; row grows \
downward, col to the right. The agent's notes and plans write places differently: (row,col) in \
the 9x11 view it saw when it wrote them, itself at (4,5), or a direction from itself; code converts \
each view place to a map place and shows both.
Vitals: HP, food (cows, plants, bats, snails), drink (water, fountains), energy (sleep), mana \
(spells). The tech tree: wood (DO on a tree) -> table (2 wood) -> wood pickaxe and sword \
(1 wood, at a table) -> stone -> stone pickaxe and sword (1 wood 1 stone, at a table) -> furnace \
(1 stone, placed anywhere: no table needed) -> coal (wood pickaxe), iron (stone pickaxe) -> iron pickaxe and sword (1 wood \
1 stone 1 iron 1 coal, table and furnace) -> diamond (iron pickaxe) -> diamond tools (3 or 2 \
diamonds). Torches (1 wood 1 coal, at a table) light dark floors. The Overworld's down-ladder is open from \
the start; every deeper floor's opens after 8 kills there. Chests hold loot, books teach \
fireball or iceball, potion colours have effects shuffled per game.
In these runs a death respawns the player on the Overworld near the spawn with full vitals, \
losing the whole inventory; achievements, XP and placed tables and furnaces stay. A table or \
furnace is destroyed (it becomes path, nothing comes back) by DO facing it or by an arrow; \
the events say when one was, and the record lists only the stations that still stand."""

# Under code_facts the notes take their places from KNOWN PLACES: map places, nothing to convert.
CF_GUIDE = CRAFTAX_GUIDE.replace(
    "The agent's notes and plans write places differently: (row,col) in the 9x11 view it saw when it wrote "
    "them, itself at (4,5), or a direction from itself; code converts each view place to a map place and shows both.",
    "The agent's notes take their places from KNOWN PLACES, in the same F<floor> (row,col) form.")

NOTES_JUDGE_PROMPT = """\
You are grading the notes a language-model agent wrote at the end of one segment of a \
Craftax game. The agent plays in segments of about 100 turns. The next segment starts from \
the saved world with a fresh context that holds only the game's record of its state and \
these notes, so the notes must carry what the record cannot: where things are, what was \
learned, and the plan. Your scores train the agent; grade what the notes would do for the \
next segment, not the prose.

{guide}

== The game record at the start and at the end of this segment ==
{record_before}
---
{record_after}

== What happened in this segment ({steps} turns) ==
{events}
actions: {actions}; forfeited turns: {forfeits}

== The current floor (upper case: the agent saw this cell; lower case: never seen) ==
legend: {legend}
{map_ascii}
nearest features: {nearest}

== The tech tree's next step, from the game's recipes ==
{next_step}

== The agent's notes ({notes_view}) ==
{notes}

== Code check of the MAP lines against the cells the agent saw (view places converted) ==
{map_claims}

Give two scores from 0 to 1 and a one-line rationale.
- findings: whether each finding is a specific fact supported by the events above (a failed \
craft and its reason, a potion's effect, a cause of damage or death, an action that did \
nothing). Generic advice, or a finding the events contradict, scores low; no findings at all \
scores 0.3 when nothing notable happened, 0 otherwise.
- plan: whether the first subgoal is the right next step (compare with the tech tree's next \
step and the record), whether every subgoal is feasible from the record's inventory and \
equipment, whether its places are real places the agent saw (the upper-case part of the \
map), and whether the survival rules fit the vitals and the light. Judge a search direction \
only by the explored part of the map: do not reward a plan for heading toward resources the \
agent never saw. Generic plans ("explore", "gather resources") score low."""

GROUP_PROMPT = """\
You are grading the notes of {n} branches of one Craftax game played by a language-model \
agent. All branches started from the same saved world and the same notes and each played \
one segment of about 100 turns; at the end of a segment the agent writes notes, and the next \
segment starts from the saved world with only the game's record of its state and those \
notes. Your scores train the agent by comparing the branches with each other, so use the \
whole 0 to 1 range when the branches differ, and give equal scores only when they are \
equally good. Grade what the notes would do for the next segment, not the prose.

{guide}

== Where every branch started ==
{record_before}

{branches}

For every branch give two scores from 0 to 1 and a one-line rationale, with the same rubric \
as for a single segment:
- findings: each finding is a specific fact supported by that branch's events; generic \
advice or a contradicted finding scores low; none at all scores 0.3 when nothing notable \
happened, 0 otherwise.
- plan: the first subgoal is the right next step (its tech tree's next step and its record), \
every subgoal is feasible from its inventory and equipment, its places are real places that \
branch saw (upper case on its map), and its survival rules fit its vitals and the light. Do \
not reward a plan for heading toward resources the branch never saw.
Return one verdict per branch, with the branch number as given."""

BRANCH_BLOCK = """\
== Branch {branch} ==
-- the game record at the end --
{record_after}
-- what happened ({steps} turns) --
{events}
actions: {actions}; forfeited turns: {forfeits}
-- its floor (upper case: seen; lower case: never seen) --
{map_ascii}
nearest features: {nearest}
-- the tech tree's next step --
{next_step}
-- its notes ({notes_view}) --
{notes}
-- code check of its MAP lines --
{map_claims}"""


# --- code_facts: the judge grades only what code cannot ------------------------------------
# The game wrote the places, the turns lost and the event log; code checks the plan's feasibility and
# whether its first subgoal is on the critical path (plan_check). The judge checks each finding (true?
# a lesson? its fix carried out by the plan?), whether finding 1 explains the largest TURNS LOST line
# (`waste`), and `deviation` = whether a plan that LEAVES the critical path has a reason that holds.

CF_TASK = """\
You are grading the notes a language-model agent wrote at the end of one segment of a Craftax \
game, for the next segment. The game itself writes the places the agent has seen (KNOWN PLACES), \
the turns that changed nothing (TURNS LOST) and the segment's events (EVENT LOG) into the next \
segment, and code has already checked the plan's feasibility and whether its first subgoal is on \
the tech tree's critical path. You check only what code cannot: whether each finding is true and \
useful, whether the first finding explains the largest loss, and whether a plan that leaves the \
critical path has a reason. The next segment reads these findings as advice from its past self, so \
a false finding does harm: check every cause against the rules above, the events and the record, \
and do not accept a rule of the game that the guide does not state. Your verdicts train the agent; \
grade what the notes would do for the next segment, not the prose."""

CF_RUBRIC = """\
- rationale: one or two lines, written first.
- findings: one entry per numbered FINDING, in order (an empty list when the FINDINGS say "none"):
  - n: its number.
  - truth: "true" when the rules, the events and the record support its cause; "false" when they \
contradict it (a rule that does not exist, a cause the events rule out, an event or place that did \
not happen); "unsupported" when you cannot tell.
  - lesson: true when it names a cause and a fix specific to what happened in this segment; false \
for a rule of the game restated, a copied event, or generic advice ("explore more").
  - applied: true when the PLAN (in a subgoal or in how one is done, e.g. the way to go) or a \
SURVIVAL rule carries out its fix; false when nothing in them does.
- waste (0 to 1, or null): null when TURNS LOST counts fewer than 10 lost turns. Otherwise: does \
finding 1 explain the first (largest) TURNS LOST line? 1: it names that loss, a cause the events \
support and a fix that would stop it. 0.5: it names the loss, but its cause or fix is vague or \
partly wrong. 0: it is about something else, or its cause is wrong.
- deviation (0 to 1): ONLY when the code check says the first subgoal is NOT on the critical path: \
is there a reason that holds for doing something else first (a vital at 3 or below, night with \
enemies near, a needed resource never seen so a search comes first, a station destroyed)? 1: the \
departure is clearly right and the plan says why. 0.5: defensible but unexplained. 0: no reason \
holds (a side achievement, a repeat, wandering). When the code check says the first subgoal IS \
on the critical path, give 1."""

CF_SINGLE_PROMPT = CF_TASK + """

{guide}

== The game record at the start and at the end of this segment ==
{record_before}
---
{record_after}

== What happened in this segment ({steps} turns) ==
{events}
actions played: {actions}; forfeited turns: {forfeits}

== What the game wrote for the next segment ==
{facts}

== The current floor (upper case: the agent saw this cell; lower case: never seen) ==
legend: {legend}
{map_ascii}

== The agent's notes ==
{notes}

== Code check of the plan ==
{plan_check}

Give your verdict:
""" + CF_RUBRIC

CF_GROUP_PROMPT = CF_TASK + """ The branches below all started from the same saved world and \
the same notes; your scores compare them, so use the whole range when they differ.

{guide}

== Where every branch started ==
{record_before}

{branches}

For every branch give a verdict:
""" + CF_RUBRIC + """
Return one verdict per branch, with the branch number as given."""

CF_BRANCH_BLOCK = """\
== Branch {branch} ==
-- the game record at the end --
{record_after}
-- what happened ({steps} turns) --
{events}
actions played: {actions}; forfeited turns: {forfeits}
-- what the game wrote for the next segment --
{facts}
-- its notes --
{notes}
-- code check of its plan --
{plan_check}"""


class NotesVerdict(BaseModel):
    findings: float = Field(ge=0.0, le=1.0)
    plan: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class BranchNotesVerdict(NotesVerdict):
    branch: int


class GroupNotesVerdict(BaseModel):
    verdicts: list[BranchNotesVerdict]
    comparison: str = ""


class FindingVerdict(BaseModel):
    n: int
    truth: Literal["true", "false", "unsupported"]
    lesson: bool
    applied: bool


class CFNotesVerdict(BaseModel):
    """`code_facts`: a verdict per finding, the largest loss explained, a justified departure."""
    rationale: str = ""
    findings: list[FindingVerdict]
    waste: float | None = Field(default=None, ge=0.0, le=1.0)
    deviation: float = Field(ge=0.0, le=1.0)


class CFBranchNotesVerdict(CFNotesVerdict):
    branch: int


class CFGroupNotesVerdict(BaseModel):
    verdicts: list[CFBranchNotesVerdict]
    comparison: str = ""


class CraftaxNotesJudge(_CrafterJudge[NotesVerdict]):
    prompt = NOTES_JUDGE_PROMPT
    schema = NotesVerdict

    def build_messages(self, **fields: Any) -> str:
        if fields.get("code_facts"):
            self.prompt, self.schema = CF_SINGLE_PROMPT, CFNotesVerdict
            return super().build_messages(guide=CF_GUIDE, **fields)
        return super().build_messages(guide=CRAFTAX_GUIDE, **fields)


class CraftaxGroupNotesJudge(_CrafterJudge[GroupNotesVerdict]):
    prompt = GROUP_PROMPT
    schema = GroupNotesVerdict

    def build_messages(self, *, infos: list[dict[str, Any]], **_: Any) -> str:
        blocks = []
        code_facts = bool(infos and infos[0].get("code_facts"))
        for i, info in enumerate(infos):
            f = judge_fields(info)
            if code_facts:
                blocks.append(CF_BRANCH_BLOCK.format(branch=i, **{k: f[k] for k in (
                    "record_after", "steps", "events", "actions", "forfeits", "facts", "notes", "plan_check")}))
            else:
                blocks.append(BRANCH_BLOCK.format(branch=i, **{k: f[k] for k in (
                    "record_after", "steps", "events", "actions", "forfeits", "map_ascii", "nearest",
                    "next_step", "notes", "notes_view", "map_claims")}))
        if code_facts:
            self.prompt, self.schema = CF_GROUP_PROMPT, CFGroupNotesVerdict
        shared = judge_fields(infos[0])
        return vf.Judge.build_messages(self, guide=CF_GUIDE if code_facts else CRAFTAX_GUIDE, n=len(infos),
                                       record_before=shared["record_before"],
                                       branches="\n\n".join(blocks))


def judge_fields(info: dict[str, Any]) -> dict[str, Any]:
    """The prompt's fields from a segment's ``trace.info["craftax"]`` (also what a replay feeds)."""
    from craftax_env.world import LEGEND

    acts = info.get("actions_hist") or {}
    top = ", ".join(f"{k} {v}" for k, v in sorted(acts.items(), key=lambda kv: -kv[1])[:12]) or "none"
    kinds = info.get("forfeit_kinds") or {}
    return {
        "record_before": info.get("record_before") or "(not recorded)",
        "record_after": info.get("record") or "(the segment wrote no record)",
        "steps": info.get("turns", "?"),
        "events": "\n".join(info.get("events") or []) or "(no unlock, station, floor change, sleep, craft or death)",
        "actions": top,
        "forfeits": f"{sum(kinds.values())} ({', '.join(f'{k} {v}' for k, v in kinds.items())})" if kinds else "0",
        "legend": LEGEND,
        "map_ascii": info.get("map_ascii") or "(unknown)",
        "nearest": info.get("nearest") or "unknown",
        "next_step": info.get("next_step") or "unknown",
        "notes": info.get("notes") or "(the agent wrote no notes)",
        "map_claims": info.get("map_claims") or "(no MAP lines)",
        "notes_view": info.get("notes_view") or "places in the notes are as written",
        "facts": info.get("facts") or "(none)",
        "plan_check": info.get("plan_check") or "(no plan check)",
        "code_facts": bool(info.get("code_facts")),
    }


def notes_score(map_check: float, findings: float, plan: float, map_weight: float = 0.6) -> dict[str, float]:
    """`map_weight` 0 under `code_facts` (the game writes the places: nothing to check)."""
    grounding = map_weight * map_check + (1.0 - map_weight) * findings
    return {"score": float(min(1.0, max(0.0, 0.5 * grounding + 0.5 * plan))), "grounding": grounding}


FINDINGS_WEIGHT = 0.3

# A finding's value: a false one is worse than none (the next segment acts on it), a true restated
# rule or copied event is worth little, a true lesson whose fix no plan subgoal or survival rule
# carries out is worth part of one that is carried out.
CLAIM_FALSE, CLAIM_GENERIC, CLAIM_UNAPPLIED = -1.0, 0.2, 0.6


def claim_value(c: Any) -> float:
    if c.truth == "false":
        return CLAIM_FALSE
    if c.truth == "unsupported":
        return 0.0
    if not c.lesson:
        return CLAIM_GENERIC
    return 1.0 if c.applied else CLAIM_UNAPPLIED


def findings_score(claims: list[Any], n_findings: int, lost: int, died: bool) -> float:
    """The mean claim value over the findings (one the judge skipped counts 0), clipped to [0, 1].
    No findings ("none"): 0.5 when fewer than LOST_MIN turns were lost and nobody died, else 0.2."""
    from craftax_env.segments import LOST_MIN

    n = max(n_findings, len(claims))
    if n == 0:
        return 0.5 if lost < LOST_MIN and not died else 0.2
    values = [claim_value(c) for c in claims[:n]] + [0.0] * (n - len(claims))
    return float(min(1.0, max(0.0, sum(values) / n)))
"""code_facts: notes = (1 - w) plan + w findings, plan = code (feasible, on_path) + the judge's deviation verdict."""


def code_facts_verdict(info: dict[str, Any], findings: float | None, deviation: float | None, rationale: str,
                       failed: bool, waste: float | None = None, **extra: Any) -> dict[str, Any]:
    """The notes reward under `code_facts`: (1 - w) x plan + w x the judge's part, the mean of
    `findings` and `waste` (when TURNS LOST asked for one). A failed judge drops its parts (no
    neutral 0.5): the plan keeps its code score."""
    from craftax_env.plan_check import plan_code_score

    feasible, on_path = info.get("plan_feasible"), float(info.get("plan_on_path") or 0.0)
    if failed:
        findings = deviation = waste = None
    plan, path = plan_code_score(feasible, on_path, deviation)
    w = float(info.get("findings_weight", FINDINGS_WEIGHT))
    llm = [float(x) for x in (findings, waste) if x is not None]
    score = plan if not llm else (1.0 - w) * plan + w * sum(llm) / len(llm)
    out = {"score": float(min(1.0, max(0.0, score))), "grounding": float(findings or 0.0), "map_check": 0.0,
           "findings": float(findings) if findings is not None else 0.0, "plan": plan, "path": path,
           "plan_on_path": on_path, "rationale": rationale, "failed": failed, **extra}
    if feasible is not None:
        out["plan_feasible"] = float(feasible)
    if deviation is not None:
        out["deviation"] = float(deviation)
    if waste is not None:
        out["waste"] = float(waste)
    return out


def cf_verdict(info: dict[str, Any], parsed: CFNotesVerdict, **extra: Any) -> dict[str, Any]:
    """A parsed `code_facts` verdict -> the notes reward, with the per-finding verdicts kept."""
    from craftax_env.segments import LOST_MIN

    lost = int(info.get("turns_lost_total") or 0)
    findings = findings_score(parsed.findings, int(info.get("findings_count") or 0), lost, bool(info.get("deaths")))
    waste = parsed.waste if lost >= LOST_MIN else None
    if lost >= LOST_MIN and waste is None:
        waste = 0.0  # the judge owed a waste score: the first finding did not explain the loss
    claims = [{"n": c.n, "truth": c.truth, "lesson": c.lesson, "applied": c.applied} for c in parsed.findings]
    return code_facts_verdict(info, findings, parsed.deviation, parsed.rationale, False, waste=waste,
                              claims=claims, **extra)


def _verdict(info: dict[str, Any], findings: float, plan: float, rationale: str, failed: bool, **extra: Any) -> dict[str, Any]:
    if info.get("code_facts"):
        return code_facts_verdict(info, findings, plan, rationale, failed, **extra)
    mc = float(info.get("map_check") or 0.0)
    s = notes_score(mc, findings, plan, 0.0 if info.get("code_facts") else 0.6)
    return {"score": s["score"], "grounding": s["grounding"], "map_check": mc, "findings": findings, "plan": plan,
            "rationale": rationale, "failed": failed, **extra}


def fallback_verdict(info: dict[str, Any], error: str) -> dict[str, Any]:
    """The judge failed: the code part stays; the LLM parts are neutral (0.5), or dropped under `code_facts`."""
    if info.get("code_facts"):
        return code_facts_verdict(info, None, None, f"judge failed: {error}", True)
    return _verdict(info, 0.5, 0.5, f"judge failed: {error}", True)


async def judge_notes(
    judge: CraftaxNotesJudge,
    trace: Any,
    *,
    weight: float,
    gate: asyncio.Semaphore | None = None,
    episode_log: str | None = None,
    price: tuple[float, float, float] | None = None,
) -> float:
    """Grade one segment's notes alone and record the ``notes`` reward on its trace."""
    info = trace.info.get("craftax") or {}
    queued = t0 = time.time()
    cost = usage_cost(None, judge.config.model)
    try:
        async with gate or _null():
            t0 = time.time()
            response = await judge.evaluate(trace=trace, **judge_fields(info))
        cost = usage_cost(getattr(response, "usage", None), judge.config.model, price, backend=getattr(judge.config, "backend", "openai"))
        parsed = response.parsed
        if parsed is None:
            raise RuntimeError("judge returned no verdict")
        if info.get("code_facts"):
            verdict = cf_verdict(info, parsed)
        else:
            verdict = _verdict(info, parsed.findings, parsed.plan, parsed.rationale, False)
    except Exception as exc:  # a segment is minutes of sampling: never fail it on the judge
        logger.warning("notes judge failed (%s); neutral LLM parts", exc)
        verdict = fallback_verdict(info, str(exc))
    timing = {"queue_s": max(0.0, t0 - queued), "call_s": time.time() - t0,
              "wait_s": time.time() - float(info.get("ended_at") or queued)}
    return apply_verdict(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="single", cost=cost)


async def group_verdicts(
    judge: CraftaxGroupNotesJudge,
    infos: list[dict[str, Any]],
    *,
    trace: Any = None,
    gate: asyncio.Semaphore | None = None,
    price: tuple[float, float, float] | None = None,
    costs: list | None = None,
) -> tuple[list[dict[str, Any]] | None, float, str | None]:
    """One comparative call over the branches of one node: a verdict per branch in order
    (None on failure), the call's latency and the error."""
    t0 = time.time()
    try:
        async with gate or _null():
            messages = judge.build_messages(infos=infos)
            response = await judge.complete(messages, trace=trace, schema=judge.schema, parse=judge.parse)
        if costs is not None:
            costs.append(usage_cost(getattr(response, "usage", None), judge.config.model, price,
                                    backend=getattr(judge.config, "backend", "openai")))
        parsed: GroupNotesVerdict | None = response.parsed
        by_branch = {v.branch: v for v in (parsed.verdicts if parsed else [])}
        if sorted(by_branch) != list(range(len(infos))):
            raise RuntimeError(f"group verdict covers branches {sorted(by_branch)}, expected 0..{len(infos) - 1}")
        out = [cf_verdict(infos[i], by_branch[i], comparison=parsed.comparison) if infos[i].get("code_facts") else
               _verdict(infos[i], by_branch[i].findings, by_branch[i].plan, by_branch[i].rationale, False,
                        comparison=parsed.comparison) for i in range(len(infos))]
        return out, time.time() - t0, None
    except Exception as exc:
        logger.warning("group notes judge failed (%s)", exc)
        return None, time.time() - t0, str(exc)


async def judge_notes_group(
    rendezvous: GroupRendezvous,
    judge: CraftaxNotesJudge,
    group_judge: CraftaxGroupNotesJudge,
    trace: Any,
    *,
    weight: float,
    gate: asyncio.Semaphore | None,
    episode_log: str | None,
    price: tuple[float, float, float] | None = None,
) -> float:
    """Group mode in the RL pipeline: meet the node's other branches on disk, take this
    branch's verdict from the one comparative call, or fall back to judging alone."""
    info = trace.info.get("craftax") or {}
    key, member = str(info.get("parent")), str(info.get("node_id"))

    async def call(infos):
        costs: list = []
        out, call_s, err = await group_verdicts(group_judge, infos, trace=trace, gate=gate, price=price, costs=costs)
        return out, call_s, err, (costs[0] if costs else None)

    verdict, timing = await rendezvous.meet(key, member, {k: v for k, v in info.items() if k != "judge"}, call)
    cost = timing.pop("_cost", None) or usage_cost(None, group_judge.config.model)
    if verdict is None:  # a sibling never arrived, or the group call failed: judge this branch alone
        await judge_notes(judge, trace, weight=weight, gate=gate, episode_log=None, price=price)
        single = trace.info["craftax"]["judge"]
        cost = {k: cost[k] + float(single.get(k, 0.0)) for k in cost}
        verdict = {k: single[k] for k in ("score", "grounding", "map_check", "findings", "plan", "path", "plan_on_path",
                                          "plan_feasible", "deviation", "waste", "claims", "rationale", "failed")
                   if k in single}
        timing = {**timing, "call_s": timing.get("call_s", 0.0) + single.get("call_s", 0.0), "group_fallback": 1.0}
    else:
        timing = {**timing, "group_fallback": 0.0}
    timing["wait_s"] = time.time() - float(info.get("ended_at") or time.time())
    return apply_verdict(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="group", cost=cost)


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
    """Record a verdict (either mode) as the ``notes`` reward, its parts as metrics, and a
    ``craftax-judge`` line in the episode log."""
    info = trace.info.get("craftax") or {}
    score = float(verdict["score"])
    trace.record_reward("notes", score, weight)
    parts = ("grounding", "map_check", "findings", "plan", "path", "plan_on_path", "plan_feasible", "deviation", "waste")
    trace.record_metrics({
        "notes_score": score, "judge_failed": float(verdict["failed"]),
        **{f"notes_{k}": float(verdict[k]) for k in parts if k in verdict},
        **{f"judge_{k}": float(v) for k, v in timing.items()},
        **{f"judge_{k}": float(v) for k, v in (cost or {}).items()},
    })
    info["judge"] = {**verdict, "mode": mode, **timing, **(cost or {})}
    trace.info["craftax"] = info
    if episode_log:
        line = {"task": "craftax-judge", "node_id": info.get("node_id"), "parent": info.get("parent"), "depth": info.get("depth"),
                "seed": info.get("seed"), "notes_score": score, "judge_failed": verdict["failed"],
                "rationale": verdict.get("rationale", ""), "at": time.time(), "mode": mode, **timing, **(cost or {}),
                **{k: verdict[k] for k in parts if k in verdict}, **({"claims": verdict["claims"]} if "claims" in verdict else {})}

        def write():
            with open(episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")

        retry_io(write, what=f"appending the judge line to {episode_log}")
    return score


def skip_verdict(trace: Any, *, weight: float, reason: str, episode_log: str | None = None) -> None:
    """A segment with no notes (empty even after the retry, or it ended: death without
    respawn, the boss, a deadline): a notes reward of 0 without a judge call, so the group's
    rewards stay comparable; logged as a `craftax-judge` line with `skipped`."""
    trace.record_reward("notes", 0.0, weight)
    trace.record_metrics({"notes_score": 0.0, "judge_skipped": 1.0})
    info = trace.info.get("craftax") or {}
    info["judge"] = {"score": 0.0, "skipped": reason, "failed": False, "mode": "skipped"}
    trace.info["craftax"] = info
    if episode_log:
        line = {"task": "craftax-judge", "node_id": info.get("node_id"), "parent": info.get("parent"), "depth": info.get("depth"),
                "seed": info.get("seed"), "notes_score": 0.0, "judge_failed": False, "skipped": reason, "mode": "skipped",
                "rationale": f"not graded: {reason}", "at": time.time()}

        def write():
            with open(episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")

        retry_io(write, what=f"appending the judge line to {episode_log}")


# --- adherence (Milestone 3) -------------------------------------------------------------
#
# Graded at the end of segment k+1 against the PLAN segment k wrote, on a code-built digest
# (adherence.py), not the segment's text. Reward on trace k+1 only:
# `adherence` = plan_{k} (the parent's plan score, the gate) x the adherence score, with
# `adherence_weight` (0.25 in arm C). The gate is one number per sibling group, so it sets
# how much following counts next to progress and leaves the ranking inside the group alone.

ADHERENCE_JUDGE_PROMPT = """\
You are grading how well one segment of a Craftax game followed the plan it was given. The \
agent wrote this plan at the end of the previous segment; this segment started from that \
world with the plan in its context. Following the plan means working toward its subgoals \
in order. Leaving a subgoal is fine when the agent's reasoning gives a reason that holds: \
danger, a vital running low, a better resource found on the way, or a subgoal that turned \
out impossible. Your scores train the agent; grade what it did, not what it wrote.

{guide}

== The plan (its (row,col) places are cells of the view it was written in; the map place code \
read from a subgoal is in brackets) ==
{parent_plan}

== The game record at the start and at the end of this segment ==
{record_before}
---
{record_after}

== Code status of each subgoal (code matches a subgoal to an achievement, item counts and a \
place when it can; "no code check" means you decide from the events and positions) ==
{subgoal_status}

== Position every 5 turns, with the distance to the active subgoal's place (the active \
subgoal is the first one code does not see done) ==
{position_track}

== What happened ({steps} turns) ==
{events}
actions: {actions}; forfeited turns: {forfeits}

== The agent's reasoning where it moved away from the active subgoal, finished subgoals out \
of order, or died (the last part of the reasoning of that turn) ==
{decision_excerpts}

For each subgoal, by its number, give one status:
- done: achieved in this segment, or already true and the agent moved on to the next one;
- attempted: real progress toward it, or a real try that failed (walking to its place, \
gathering what it needs, a failed craft);
- deviated_justified: left it for a reason that holds (see above);
- ignored: no real work on it while it was the subgoal to work on, and no reason that holds;
- not_reached: the segment ended while it was still working on the previous subgoal (that one \
attempted), or right after finishing it. A subgoal that comes after one finished with turns to \
spare, and got no real work, is ignored, not not_reached.
A subgoal code marks "already true at the start of this segment (not scored)" is not scored \
whatever you give it: answer done, and judge the later subgoals as the ones to work on.
Then a one-line rationale."""

GROUP_ADHERENCE_PROMPT = """\
You are grading how well {n} branches of one Craftax game followed the plan they were all \
given. The agent wrote this plan at the end of the previous segment; every branch started \
from that same world with the plan in its context and played one segment. Following the plan \
means working toward its subgoals in order; leaving a subgoal is fine when the agent's \
reasoning gives a reason that holds (danger, a vital running low, a better resource found on \
the way, a subgoal that turned out impossible). Your statuses train the agent by comparing \
the branches; grade what each branch did, not what it wrote.

{guide}

== The plan (shared; its (row,col) places are cells of the view it was written in, the map \
place code read from a subgoal in brackets) ==
{parent_plan}

== The game record at the start (shared) ==
{record_before}

{branches}

For every branch and every subgoal, by its number, give one status: done (achieved, or \
already true and the branch moved on), attempted (real progress or a real try that failed), \
deviated_justified (left for a reason that holds), ignored (no real work on it while it was \
the subgoal to work on, no reason that holds), not_reached (the segment ended while still \
working on the previous subgoal, or right after finishing it; a subgoal after one finished with \
turns to spare that got no real work is ignored). A subgoal code marks "already true at the start of this segment (not \
scored)" is not scored: answer done. Then a one-line rationale per branch. Return one verdict \
per branch, with the branch number as given."""

ADHERENCE_BRANCH_BLOCK = """\
== Branch {branch} ==
-- the game record at the end --
{record_after}
-- code status of each subgoal --
{subgoal_status}
-- position every 5 turns --
{position_track}
-- what happened ({steps} turns) --
{events}
actions: {actions}; forfeited turns: {forfeits}
-- its reasoning at decision points --
{decision_excerpts}"""

STATUSES = ("done", "attempted", "deviated_justified", "ignored", "not_reached", "already_done")
STATUS_SCORE = {"done": 1.0, "attempted": 0.6, "deviated_justified": 1.0, "ignored": 0.0}
SUBGOAL_WEIGHTS = (0.5, 0.3, 0.2)  # later subgoals 0.1 each


class SubgoalStatus(BaseModel):
    n: int
    status: Literal["done", "attempted", "deviated_justified", "ignored", "not_reached"]


class AdherenceVerdict(BaseModel):
    subgoals: list[SubgoalStatus]
    rationale: str = ""


class BranchAdherenceVerdict(AdherenceVerdict):
    branch: int


class GroupAdherenceVerdict(BaseModel):
    verdicts: list[BranchAdherenceVerdict]
    comparison: str = ""


def adherence_fields(info: dict[str, Any]) -> dict[str, Any]:
    """The adherence prompt's fields from a segment's ``trace.info["craftax"]``."""
    base = judge_fields(info)
    digest = info.get("adherence_digest") or {}
    return {k: base[k] for k in ("record_before", "record_after", "steps", "events", "actions", "forfeits")} | {
        k: digest.get(k) or "(not recorded)" for k in ("parent_plan", "subgoal_status", "position_track", "decision_excerpts")}


class CraftaxAdherenceJudge(_CrafterJudge[AdherenceVerdict]):
    prompt = ADHERENCE_JUDGE_PROMPT
    schema = AdherenceVerdict

    def build_messages(self, **fields: Any) -> str:
        return super().build_messages(guide=CRAFTAX_GUIDE, **fields)


class CraftaxGroupAdherenceJudge(_CrafterJudge[GroupAdherenceVerdict]):
    prompt = GROUP_ADHERENCE_PROMPT
    schema = GroupAdherenceVerdict

    def build_messages(self, *, infos: list[dict[str, Any]], **_: Any) -> str:
        blocks = []
        for i, info in enumerate(infos):
            f = adherence_fields(info)
            blocks.append(ADHERENCE_BRANCH_BLOCK.format(branch=i, **{k: f[k] for k in (
                "record_after", "subgoal_status", "position_track", "steps", "events", "actions", "forfeits",
                "decision_excerpts")}))
        shared = adherence_fields(infos[0])
        return vf.Judge.build_messages(self, guide=CRAFTAX_GUIDE, n=len(infos), parent_plan=shared["parent_plan"],
                                       record_before=shared["record_before"], branches="\n\n".join(blocks))


def adherence_score(statuses: dict[int, str], n_subgoals: int, already: set[int] | frozenset = frozenset(),
                    late: set[int] | frozenset = frozenset()) -> dict[str, Any]:
    """Statuses -> the score: done or deviated_justified 1.0, attempted 0.6, ignored 0,
    weighted 0.5 / 0.3 / 0.2 (0.1 for each later subgoal); not_reached subgoals leave the
    sum (renormalized over the rest), and so do the subgoals in `already` (true when the
    segment began, by code: every sibling gets them free). A subgoal without a status counts
    as ignored. not_reached means "the segment ended while still on the previous subgoal":
    it counts as ignored as the first scored subgoal, after an ignored one, and after a done
    one unless code saw that one finish in the last LATE_TURNS turns (`late`): finishing
    subgoal 1 and then doing nothing must not score like finishing them all."""
    out, blocked = {}, True  # the first scored subgoal cannot be "not reached"
    for n in range(1, n_subgoals + 1):
        if n in already:
            out[n] = "already_done"
            continue
        s = statuses.get(n, "ignored")
        if s == "not_reached" and blocked:
            s = "ignored"
        out[n] = s
        blocked = s == "ignored" or (s == "done" and n not in late)
    weights = {n: SUBGOAL_WEIGHTS[n - 1] if n <= len(SUBGOAL_WEIGHTS) else 0.1 for n in out}
    counted = {n: s for n, s in out.items() if s not in ("not_reached", "already_done")}
    total = sum(weights[n] for n in counted)
    score = sum(weights[n] * STATUS_SCORE[s] for n, s in counted.items()) / total if total else 0.0
    return {"score": float(score), "statuses": out}


def _adherence_verdict(info: dict[str, Any], parsed: AdherenceVerdict | None, failed: bool, rationale: str = "",
                       **extra: Any) -> dict[str, Any]:
    subgoals = (info.get("adherence_digest") or {}).get("subgoals") or []
    if parsed is None:  # the judge failed: neutral
        return {"score": 0.5, "statuses": {}, "rationale": rationale, "failed": True, **extra}
    already = {int(g["n"]) for g in subgoals if g.get("done_at_start")}
    late = {int(g["n"]) for g in subgoals if g.get("done_late")}
    s = adherence_score({v.n: v.status for v in parsed.subgoals}, len(subgoals), already, late)
    return {**s, "rationale": parsed.rationale, "failed": failed, **extra}


async def judge_adherence(
    judge: CraftaxAdherenceJudge,
    trace: Any,
    *,
    weight: float,
    gate: asyncio.Semaphore | None = None,
    episode_log: str | None = None,
    price: tuple[float, float, float] | None = None,
) -> float:
    """Grade one segment's plan-following alone and record the ``adherence`` reward."""
    info = trace.info.get("craftax") or {}
    queued = t0 = time.time()
    cost = usage_cost(None, judge.config.model)
    try:
        async with gate or _null():
            t0 = time.time()
            response = await judge.evaluate(trace=trace, **adherence_fields(info))
        cost = usage_cost(getattr(response, "usage", None), judge.config.model, price, backend=getattr(judge.config, "backend", "openai"))
        if response.parsed is None:
            raise RuntimeError("judge returned no verdict")
        verdict = _adherence_verdict(info, response.parsed, False)
    except Exception as exc:
        logger.warning("adherence judge failed (%s); neutral 0.5", exc)
        verdict = _adherence_verdict(info, None, True, f"judge failed: {exc}")
    timing = {"queue_s": max(0.0, t0 - queued), "call_s": time.time() - t0}
    return apply_adherence(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="single", cost=cost)


async def group_adherence_verdicts(
    judge: CraftaxGroupAdherenceJudge,
    infos: list[dict[str, Any]],
    *,
    trace: Any = None,
    gate: asyncio.Semaphore | None = None,
    price: tuple[float, float, float] | None = None,
    costs: list | None = None,
) -> tuple[list[dict[str, Any]] | None, float, str | None]:
    """One comparative call over the branches of one node (they share the plan)."""
    t0 = time.time()
    try:
        async with gate or _null():
            messages = judge.build_messages(infos=infos)
            response = await judge.complete(messages, trace=trace, schema=judge.schema, parse=judge.parse)
        if costs is not None:
            costs.append(usage_cost(getattr(response, "usage", None), judge.config.model, price,
                                    backend=getattr(judge.config, "backend", "openai")))
        parsed: GroupAdherenceVerdict | None = response.parsed
        by_branch = {v.branch: v for v in (parsed.verdicts if parsed else [])}
        if sorted(by_branch) != list(range(len(infos))):
            raise RuntimeError(f"group verdict covers branches {sorted(by_branch)}, expected 0..{len(infos) - 1}")
        out = [_adherence_verdict(infos[i], by_branch[i], False, comparison=parsed.comparison) for i in range(len(infos))]
        return out, time.time() - t0, None
    except Exception as exc:
        logger.warning("group adherence judge failed (%s)", exc)
        return None, time.time() - t0, str(exc)


async def judge_adherence_group(
    rendezvous: GroupRendezvous,
    judge: CraftaxAdherenceJudge,
    group_judge: CraftaxGroupAdherenceJudge,
    trace: Any,
    *,
    weight: float,
    gate: asyncio.Semaphore | None,
    episode_log: str | None,
    price: tuple[float, float, float] | None = None,
) -> float:
    """Group mode in the RL pipeline (its own rendezvous directory), else judged alone."""
    info = trace.info.get("craftax") or {}
    key, member = str(info.get("parent")), str(info.get("node_id"))

    async def call(infos):
        costs: list = []
        out, call_s, err = await group_adherence_verdicts(group_judge, infos, trace=trace, gate=gate, price=price, costs=costs)
        return out, call_s, err, (costs[0] if costs else None)

    payload = {k: v for k, v in info.items() if k not in ("judge", "adherence_judge")}
    verdict, timing = await rendezvous.meet(key, member, payload, call)
    cost = timing.pop("_cost", None) or usage_cost(None, group_judge.config.model)
    if verdict is None:
        await judge_adherence(judge, trace, weight=weight, gate=gate, episode_log=None, price=price)
        single = trace.info["craftax"]["adherence_judge"]
        cost = {k: cost[k] + float(single.get(k, 0.0)) for k in cost}
        verdict = {k: single[k] for k in ("score", "statuses", "rationale", "failed") if k in single}
        timing = {**timing, "call_s": timing.get("call_s", 0.0) + single.get("call_s", 0.0), "group_fallback": 1.0}
    else:
        timing = {**timing, "group_fallback": 0.0}
    return apply_adherence(trace, verdict, weight=weight, timing=timing, episode_log=episode_log, mode="group", cost=cost)


def apply_adherence(
    trace: Any,
    verdict: dict[str, Any],
    *,
    weight: float,
    timing: dict[str, float],
    episode_log: str | None = None,
    mode: str = "single",
    cost: dict[str, float] | None = None,
) -> float:
    """Record a verdict as the ``adherence`` reward (gate x score), its parts as metrics,
    and a ``craftax-adherence`` line in the episode log."""
    info = trace.info.get("craftax") or {}
    gate_value = float(info.get("parent_plan_score") if info.get("parent_plan_score") is not None else 0.5)
    score = float(verdict["score"])
    trace.record_reward("adherence", gate_value * score, weight)
    statuses = verdict.get("statuses") or {}
    trace.record_metrics({
        "adherence_score": score, "adherence_gate": gate_value, "adherence_failed": float(verdict["failed"]),
        "adherence_gate_missing": float(info.get("parent_plan_score") is None),
        **{f"adherence_{s}": float(sum(v == s for v in statuses.values())) for s in STATUSES},
        **{f"adherence_judge_{k}": float(v) for k, v in timing.items()},
        **{f"adherence_judge_{k}": float(v) for k, v in (cost or {}).items()},
    })
    info["adherence_judge"] = {**verdict, "statuses": {str(k): v for k, v in statuses.items()}, "gate": gate_value,
                               "mode": mode, **timing, **(cost or {})}
    trace.info["craftax"] = info
    if episode_log:
        _log(episode_log, {"task": "craftax-adherence", "node_id": info.get("node_id"), "parent": info.get("parent"),
                           "depth": info.get("depth"), "seed": info.get("seed"), "adherence_score": score, "gate": gate_value,
                           "reward": gate_value * score, "statuses": {str(k): v for k, v in statuses.items()},
                           "judge_failed": verdict["failed"], "rationale": verdict.get("rationale", ""), "at": time.time(),
                           "mode": mode, **timing, **(cost or {})})
    return score


def skip_adherence(trace: Any, *, weight: float, reason: str, episode_log: str | None = None) -> None:
    """The parent left no PLAN code could split into subgoals: adherence 0 without a call
    (the siblings share the parent, so they all get it)."""
    trace.record_reward("adherence", 0.0, weight)
    trace.record_metrics({"adherence_score": 0.0, "adherence_skipped": 1.0})
    info = trace.info.get("craftax") or {}
    info["adherence_judge"] = {"score": 0.0, "skipped": reason, "failed": False, "mode": "skipped"}
    trace.info["craftax"] = info
    if episode_log:
        _log(episode_log, {"task": "craftax-adherence", "node_id": info.get("node_id"), "parent": info.get("parent"),
                           "depth": info.get("depth"), "seed": info.get("seed"), "adherence_score": 0.0, "reward": 0.0,
                           "judge_failed": False, "skipped": reason, "mode": "skipped", "rationale": f"not graded: {reason}",
                           "at": time.time()})


def _log(path: str, line: dict[str, Any]) -> None:
    def write():
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")

    retry_io(write, what=f"appending the judge line to {path}")

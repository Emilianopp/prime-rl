"""The code half of the notes reward under `code_facts` (the LLM half is the judge's `findings`
and, when the plan leaves the critical path, whether that is justified).

Offline replay on the 137 sibling groups of 2026-11-02 (children's progress after the end state's
effect is regressed out): `feasible` 0.57 +- 0.03 and `on_path` 0.57 +- 0.04 pairwise agreement; map
precision and finding-event matching anti-predicted (0.41 / 0.40), so they are not here.

- feasible (P2): of the subgoals that name an achievement, the fraction whose tools and stations
  (world.REQUIRES) are held at the end of the segment or made by an EARLIER subgoal; a subgoal for an
  achievement already unlocked whose capability is still held is wasted (infeasible). None: no
  subgoal names an achievement.
- on_path (P3): 1 if the first subgoal is on the critical path (world.critical_path: the next
  ladder step and the providers of the capabilities it lacks; the materials still missing for the
  next two steps, summed, and the features to find them), 0.5 if only the second one is, else 0.
  Logged only since 2026-11-04: under the drop-everything respawn the critical path says "rebuild the
  wood pickaxe" in 9 of 10 segments, so on_path graded the restart, not progress (judged beam
  w16-96k-cf: 0.28 pairwise agreement with the children's progress).
- on_track: 1 if the plan chains to an achievement this lineage has NOT unlocked (the `target`: the
  first such subgoal) and every subgoal before it serves that target (a provider of a capability it
  lacks, or a material or feature run of its requirement chain, world.requirements); 0.5 if a target
  exists but a subgoal before it is a detour; 0 if the plan only repeats unlocked achievements or
  names none. The notes reward's code half is mean(feasible, on_track) (`plan_track_score`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from craftax_env import world
from craftax_env.adherence import Subgoal, parse_plan


@dataclass
class PlanCheck:
    feasible: float | None
    on_path: float
    critical: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    on_track: float = 0.0
    target: str | None = None  # the first subgoal's achievement not yet unlocked (what the plan is for)
    target_n: int | None = None

    def text(self) -> str:
        head = (f"critical path (next tech step and what it lacks, prerequisites first): {', '.join(self.critical) or 'none'}; "
                f"first subgoal on it: {'yes' if self.on_path == 1 else 'only the second subgoal' if self.on_path == 0.5 else 'no'}; "
                f"feasible subgoals: {'n/a' if self.feasible is None else f'{self.feasible:.2f}'}; "
                + (f"new achievement the plan reaches: {self.target} (subgoal {self.target_n}), "
                   f"{'every' if self.on_track == 1 else 'not every'} subgoal before it serves it"
                   if self.target else "the plan reaches no achievement this lineage lacks"))
        return "\n".join([head, *self.lines])


def _on_path(s: Subgoal, achs: list[str], mats: set[str], feats: set[str]) -> bool:
    """A named achievement counts by name only ("make_wood_sword" names wood but is no wood run);
    a gather or search subgoal counts by the material or feature it targets."""
    if s.achievement and not s.achievement.startswith("collect_"):
        return s.achievement in achs
    if s.achievement and s.achievement in achs:
        return True
    if any(item in mats for item, _, _ in s.items):
        return True
    return bool(s.thing and s.thing in feats)


def check_plan(game: Any, explored: np.ndarray, plan_lines: list[str], view: tuple[int, int, int] | None = None) -> PlanCheck:
    names = [a.name.lower() for a in game._rt["achievements"]]
    subgoals = parse_plan(plan_lines, names, game.floor, view=view)
    achs, _, _ = world.critical_path(game, explored, steps=1)
    # The materials and features of the next TWO steps count: wood for the pickaxe right after the
    # table is on the path; a named craft still has to be the next step itself.
    _, mats, feats = world.critical_path(game, explored, steps=2)
    caps, have = world.capabilities(game, explored), set(game.achievements)
    lines, held, ok, named = [], set(caps), 0, 0
    for s in subgoals:
        a = s.achievement
        if not a:
            lines.append(f"- {s.n}. no achievement recognized: {s.goal[:80]}")
            continue
        named += 1
        lacks = [c for c in world.REQUIRES.get(a, ()) if c not in held]
        # A material run is never wasted (wood and stone are spent again and again); a craft or
        # placement already unlocked is wasted only while its capability is still held.
        wasted = (not a.startswith("collect_")) and a in have and (a not in world.PROVIDES or world.PROVIDES[a] in held)
        good = not lacks and not wasted
        ok += good
        why = "ok" if good else ("already unlocked" if wasted else f"lacks {', '.join(lacks)}")
        lines.append(f"- {s.n}. {a}: {why}{' (on the critical path)' if _on_path(s, achs, mats, feats) else ''}")
        if a in world.PROVIDES:
            held.add(world.PROVIDES[a])
    if subgoals:
        on_path = 1.0 if _on_path(subgoals[0], achs, mats, feats) else (
            0.5 if len(subgoals) > 1 and _on_path(subgoals[1], achs, mats, feats) else 0.0)
    else:
        on_path = 0.0
    on_track, target, target_n = _on_track(game, explored, subgoals, have)
    return PlanCheck(feasible=ok / named if named else None, on_path=on_path, critical=achs, lines=lines,
                     on_track=on_track, target=target, target_n=target_n)


def _serves(s: Subgoal, chain: list[str], mats: set[str], feats: set[str]) -> bool:
    """A subgoal before the target serves it when it is a step of the target's requirement chain, a
    run for a material the chain still lacks, or a search for the feature that gives one."""
    if s.achievement and s.achievement in chain:
        return True
    if s.achievement in world._COLLECT_FEATURE and world._COLLECT_FEATURE[s.achievement] in feats:
        return True
    if any(item in mats for item, _, _ in s.items):
        return True
    return bool(s.thing and s.thing in feats)


def _on_track(game: Any, explored: np.ndarray, subgoals: list[Subgoal], have: set[str]) -> tuple[float, str | None, int | None]:
    """(on_track, target, target subgoal number): see the module docstring."""
    for i, s in enumerate(subgoals):
        if s.achievement and s.achievement not in have:
            chain, mats, feats = world.requirements(game, explored, [s.achievement])
            before = subgoals[:i]
            return (1.0 if all(_serves(b, chain, mats, feats) for b in before) else 0.5), s.achievement, s.n
    return 0.0, None, None


def plan_code_score(feasible: float | None, on_path: float, deviation: float | None) -> tuple[float, float]:
    """(plan score, path score). path = on_path, raised to the judge's `deviation` verdict when the
    plan leaves the critical path (a vital or a danger first can be right); the judge's verdict is
    ignored when the first subgoal is on the path, and absent (judge failed) leaves on_path."""
    path = on_path if on_path >= 1.0 or deviation is None else max(on_path, float(deviation))
    plan = path if feasible is None else 0.5 * (path + feasible)
    return plan, path


def plan_track_score(feasible: float | None, on_track: float) -> float:
    """The code half of the notes reward since 2026-11-04: mean(feasible, on_track), on_track alone
    when no subgoal names an achievement (feasible None)."""
    return float(on_track) if feasible is None else 0.5 * (float(feasible) + float(on_track))

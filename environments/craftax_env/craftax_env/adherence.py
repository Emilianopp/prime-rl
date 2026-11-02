"""The adherence digest (docs/craftax-segments-judge.md, Milestone 3): how a segment worked
on the PLAN its parent wrote, as code can see it.

- ``parse_plan``: the parent's PLAN lines -> subgoals, each matched (leniently: the model's
  plans are free text) to an achievement, item-count targets, a place, and (below the
  Overworld) a ladder target: the floor's kills reaching the 8 that open its down-ladder.
  A generic "descend" below the Overworld is the next floor's entry achievement.
- ``AdherenceTracker``: fed once per turn during the segment; per subgoal the turn its
  achievement unlocked, the item counts reached, the closest approach to its place; a
  position every 5 turns with the distance to the active subgoal (the first one code does
  not see done); and at most 6 tails of the agent's reasoning at decision points (it moved
  away from the active subgoal's place for 10+ turns, it finished a later subgoal before
  the active one, it died).
- ``digest``: the texts the adherence judge reads. Code gives the evidence; the judge gives
  each subgoal its status (judge.py turns the statuses into the score).

Every place is written with ``glyph.place``; turns are game turns, as in the timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from craftax_env.glyph import place
from craftax_env.world import claim_feature, parse_places

TRACK_EVERY = 5
AWAY_TURNS = 10  # turns moving away from the active subgoal's place before an excerpt
AWAY_TILES = 3  # ... and at least this much farther than its closest approach
MAX_EXCERPTS = 6
EXCERPT_CHARS = 1200  # ~300 tokens: the tail of the reasoning, where the decision is
MAX_SUBGOALS = 6
LATE_TURNS = 5  # a subgoal finished this close to the end leaves no time for the next one

ITEMS = ("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling", "arrows", "torches")
_ITEM_WORD = {"wood": "wood", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond",
              "sapphire": "sapphire", "ruby": "ruby", "rubie": "ruby", "sapling": "sapling", "arrow": "arrows",
              "arrows": "arrows", "torch": "torches", "torche": "torches", "torches": "torches"}
_ITEM_RE = "|".join(sorted(_ITEM_WORD, key=len, reverse=True))
_COUNT_FIRST = re.compile(rf"\b(\d{{1,2}})\s*\+?\s*(more\s+)?({_ITEM_RE})s?\b", re.IGNORECASE)
_ITEM_FIRST = re.compile(rf"\b({_ITEM_RE})s?\s*(?::|=|>=|≥|>)?\s*(\d{{1,2}})\b(?!\s*[,)])", re.IGNORECASE)

# The verbs of a goal -> the achievement verb they mean ("mine coal" is collect_coal).
_VERBS: dict[str, tuple[str, ...]] = {
    "make": ("make", "craft", "build", "forge", "create"),
    "place": ("place", "put"),
    "collect": ("collect", "mine", "gather", "harvest", "get", "chop", "dig", "obtain", "cut", "drink"),
    "defeat": ("defeat", "kill", "fight", "slay", "attack", "hunt"),
    "eat": ("eat",),
    "enter": ("enter", "descend", "go down"),
    "find": ("find",), "fire": ("fire", "shoot"), "learn": ("learn", "read"), "cast": ("cast",),
    "open": ("open",), "enchant": ("enchant",), "wake": ("sleep", "wake"),
}
# Extra noun sets for achievements whose name words are not what a plan says.
_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "collect_wood": (("tree",),), "collect_drink": (("water",),), "wake_up": (("sleep",), ("wake",)),
    "enter_dungeon": (("dungeon",), ("ladder",), ("descend",)), "make_iron_armour": (("iron", "armor"),),
    "make_diamond_armour": (("diamond", "armor"),), "enchant_armour": (("armor",),),
    "defeat_orc_solider": (("orc", "soldier"),),
}
# Floor -> the achievement for entering it (Craftax's LEVEL_ACHIEVEMENT_MAP; floor 0 has none).
# A plan's generic "descend" / "ladder" matches enter_dungeon; below the Overworld it means the
# next floor down.
FLOOR_ENTRY = (None, "enter_dungeon", "enter_gnomish_mines", "enter_sewers", "enter_vault", "enter_troll_mines",
               "enter_fire_realm", "enter_ice_realm", "enter_graveyard")
LADDER_KILLS = 8  # Craftax's MONSTERS_KILLED_TO_CLEAR_LEVEL: kills on a floor that open its down-ladder
# "open the ladder", "until the ladder opens", "Ladder: OPEN", "kill 8 monsters", "8 kills", "clear the floor".
_LADDER_OPEN = re.compile(
    r"\b(?:open|unlock|clear)\w*\b[^.|;]{0,30}?\bladder\b|\bladder\b[^.|;]{0,20}?\b(?:open|unlock)\w*\b"
    r"|\bkill\w*\s+(?:\d+\s+(?:more\s+)?)?(?:monsters|enemies|mobs|creatures)\b|\b\d+\s+kills\b|\bclear\w*\s+(?:the\s+|this\s+)?(?:floor|level)\b",
    re.IGNORECASE)
_FIELD = re.compile(r"^\s*(goal|where|at|near|destination|needs|need|done when|done)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_NUMBERED = re.compile(r"(?:^|(?<=\s))([1-9])\s*[.:)]\s+")


@dataclass
class Subgoal:
    n: int
    text: str
    goal: str
    where: str = ""
    done_when: str = ""
    achievement: str | None = None
    items: list[tuple[str, int, bool]] = field(default_factory=list)  # (item, count, relative to the start)
    place: tuple[int, int, int] | None = None  # (floor, row, col)
    thing: str | None = None
    place_seen: bool = True  # False: the agent never saw that cell (a grid-view coordinate, or made up)
    ladder_floor: int | None = None  # open this floor's down-ladder (LADDER_KILLS kills on it)


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower().replace("_", " ")) + " "


def _has(t: str, word: str) -> int:
    """Position of `word` (or its plural) in the normalized text `t`, else -1."""
    m = re.search(rf" {re.escape(word)}(?:s|es)? ", t)
    return m.start() if m else -1


def match_achievement(text: str, names: list[str]) -> str | None:
    """The achievement a goal names: every noun of the name (or of an alias) appears; a
    name whose verb matches the goal's first verb wins, then more nouns, then the earliest."""
    t = _words(text)
    verb, at = None, len(t)
    for v, words in _VERBS.items():
        for w in words:
            p = _has(t, w)
            if 0 <= p < at:
                verb, at = v, p
    best: tuple[int, int, int, str] | None = None
    for name in names:
        head, *nouns = name.split("_")
        for alt in ((tuple(nouns),) if nouns else ()) + _ALIASES.get(name, ()):
            pos = [_has(t, w) for w in alt]
            if not alt or min(pos) < 0:
                continue
            key = (int(head == verb or (verb == "wake" and name == "wake_up")), len(alt), -min(pos), name)
            if best is None or key[:3] > best[:3]:
                best = key
    return best[3] if best else None


def _targets(text: str) -> list[tuple[str, int, bool]]:
    out: dict[str, tuple[str, int, bool]] = {}
    for m in _COUNT_FIRST.finditer(text):
        item = _ITEM_WORD[m.group(3).lower()]
        out.setdefault(item, (item, int(m.group(1)), bool(m.group(2))))
    for m in _ITEM_FIRST.finditer(text):
        item = _ITEM_WORD[m.group(1).lower()]
        out.setdefault(item, (item, int(m.group(2)), False))
    return [v for v in out.values() if v[1] > 0]


def _split_items(lines: list[str]) -> list[str]:
    """The PLAN's subgoals: numbered items (also several on one line, "1: ... 2: ..."),
    numbered in order; without numbers, one per line."""
    text = "\n".join(lines)
    starts, expect = [], 1
    for m in _NUMBERED.finditer(text):
        if int(m.group(1)) == expect:
            starts.append((m.start(1), m.end()))
            expect += 1
    if not starts:
        return [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in lines if ln.strip()][:MAX_SUBGOALS]
    items = [text[s[1]:(starts[i + 1][0] if i + 1 < len(starts) else len(text))] for i, s in enumerate(starts)]
    return [" ".join(it.split()) for it in items][:MAX_SUBGOALS]


def parse_plan(lines: list[str], achievement_names: list[str], default_floor: int = 0,
               view: tuple[int, int, int] | None = None) -> list[Subgoal]:
    """The parent's PLAN lines -> subgoals (see the module docstring). `view`: the player's
    (floor, row, col) when the plan was written, for its view-frame places (world.parse_places)."""
    out = []
    for n, text in enumerate(_split_items(lines), start=1):
        parts = [p.strip() for p in text.split("|")]
        goal, where, done = parts[0], "", ""
        for p in parts:
            m = _FIELD.match(p)
            if not m:
                continue
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "goal":
                goal = value
            elif key in ("where", "at", "near", "destination"):
                where = value
            elif key.startswith("done"):
                done = value
        m = _FIELD.match(goal)
        if m and m.group(1).lower() == "goal":
            goal = m.group(2).strip()
        if len(parts) == 1:
            low = text.lower()
            cut = low.find("done when")
            if cut >= 0:
                done, goal = text[cut + len("done when"):].lstrip(" :"), text[:cut].rstrip(" |,;-–—")
            where = text
        places = parse_places(where or goal, default_floor, view) or parse_places(text, default_floor, view)
        achievement = match_achievement(goal, achievement_names) or (match_achievement(done, achievement_names) if done else None)
        if achievement == "enter_dungeon" and default_floor >= 1:  # "descend" below the Overworld: the next floor
            achievement = FLOOR_ENTRY[default_floor + 1] if default_floor + 1 < len(FLOOR_ENTRY) else None
        # The Overworld's ladder starts open, so a kill target means something only below it.
        opens_ladder = default_floor >= 1 and bool(_LADDER_OPEN.search(f"{goal} | {done}"))
        out.append(Subgoal(
            n=n, text=text, goal=goal, where=where if len(parts) > 1 else "", done_when=done,
            achievement=achievement,
            items=_targets(f"{goal} | {done}"),
            place=places[0] if places else None,
            thing=claim_feature(where or goal),
            ladder_floor=default_floor if opens_ladder else None,
        ))
    return out


def plan_text(subgoals: list[Subgoal]) -> str:
    """The plan as written; a subgoal with a place also gets the map place code read from it
    (the plan writes view-frame places)."""
    return "\n".join(f"{s.n}. {s.text}" + (f"  [map place: {place(s.place[0], s.place[1:])}]" if s.place else "")
                     for s in subgoals) or "(no subgoals)"


def _inventory(game: Any, items: set[str]) -> dict[str, int]:
    inv = game.state.inventory
    return {k: int(np.asarray(getattr(inv, k))) for k in items}


def _kills(game: Any, floors: set[int]) -> dict[int, int]:
    killed = np.asarray(game.state.monsters_killed)
    return {f: int(killed[f]) for f in floors if 0 <= f < killed.shape[0]}


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class AdherenceTracker:
    """Per-turn evidence of how the segment worked on the parent's subgoals."""

    def __init__(self, subgoals: list[Subgoal], game: Any, achievements_before: list[str],
                 explored: np.ndarray | None = None, tolerance: int = 1) -> None:
        self.subgoals = subgoals
        if explored is not None:  # a plan place the agent never saw is not tracked as a place
            for s in subgoals:
                if s.place is not None:
                    f, r, c = s.place
                    inside = 0 <= f < explored.shape[0] and 0 <= r < explored.shape[1] and 0 <= c < explored.shape[2]
                    s.place_seen = bool(inside and explored[f, max(0, r - tolerance):r + tolerance + 1,
                                                            max(0, c - tolerance):c + tolerance + 1].any())
        self.before = set(achievements_before)
        self.items = {item for s in subgoals for item, _, _ in s.items}
        self.start_inv = _inventory(game, self.items)
        self.max_inv = dict(self.start_inv)
        self.end_inv = dict(self.start_inv)
        self.ladder_floors = {s.ladder_floor for s in subgoals if s.ladder_floor is not None}
        self.start_kills = _kills(game, self.ladder_floors)
        self.end_kills = dict(self.start_kills)
        self.ladder_open_at: dict[int, int] = {}  # floor -> the turn its kills reached LADDER_KILLS
        self.first_turn = self.last_turn = game.turn
        self.unlocked: dict[str, int] = {}
        self.item_reached: dict[tuple[int, str], int] = {}
        self.closest: dict[int, tuple[int, int]] = {}  # subgoal n -> (tiles, turn)
        self.start_dist: dict[int, int | None] = {s.n: self._dist(s, game) for s in subgoals}
        self.end_dist: dict[int, int | None] = dict(self.start_dist)
        for s in subgoals:
            if self.start_dist[s.n] is not None:
                self.closest[s.n] = (self.start_dist[s.n], game.turn)
        self.done_at: dict[int, int] = {s.n: game.turn for s in subgoals if self._done(s)}
        self.done_at_start = set(self.done_at)  # already true when the segment began: not scored
        self.track: list[str] = []
        self._observed = 0
        self.excerpts: list[tuple[int, str, str]] = []
        self.tails: dict[int, str] = {}
        self._pending: str | None = None
        self._active = self.active()
        self._best, self._streak, self._away_from, self._armed = None, 0, 0, True
        self._track(game)

    # --- code's view of a subgoal ----------------------------------------------------
    def _target(self, item: str, count: int, relative: bool) -> int:
        return self.start_inv.get(item, 0) + count if relative else count

    def _done(self, s: Subgoal) -> bool | None:
        """True / False when code can tell, else None. The achievement unlocking in this
        segment is done; an achievement unlocked before it counts only without item targets
        ("place a table" when one is placed), else the targets decide ("wood 1 -> 3" is not
        done because collect_wood was unlocked long ago); item targets alone decide too."""
        items = [self.max_inv.get(item, 0) >= self._target(item, count, rel) for item, count, rel in s.items]
        if s.ladder_floor is not None:  # kills only grow, so the end count says it
            items.append(self.end_kills.get(s.ladder_floor, 0) >= LADDER_KILLS)
        if s.achievement and s.achievement in self.unlocked and s.achievement not in self.before:
            return True
        if items:
            return all(items)
        if s.achievement:
            return s.achievement in self.before
        return None

    @staticmethod
    def _dist(s: Subgoal, game: Any) -> int | None:
        if s.place is None or not s.place_seen or s.place[0] != game.floor:
            return None
        return _cheb(tuple(game.position), (s.place[1], s.place[2]))

    def active(self) -> Subgoal | None:
        return next((s for s in self.subgoals if s.n not in self.done_at), None)

    # --- per turn ----------------------------------------------------------------------
    def reply(self, content: str | None, reasoning: str | None) -> None:
        """The reply that chooses the next action (its reasoning tail, for the excerpts)."""
        text = (reasoning or "").strip() or (content or "").strip()
        self._pending = text[-EXCERPT_CHARS:] if text else None

    def _excerpt(self, turn: int, why: str) -> None:
        if len(self.excerpts) >= MAX_EXCERPTS:
            return
        tail = self.tails.get(turn) or self.tails.get(max(self.tails, default=-1), "")
        self.excerpts.append((turn, why, tail or "(no reasoning recorded)"))

    def died(self, turn: int, where: str) -> None:
        if self._pending is not None:
            self.tails[turn] = self._pending
        self._excerpt(turn, f"died at {where}")

    def observe(self, game: Any, unlocked: dict[str, int]) -> None:
        """After the turn's actions (and a respawn): `unlocked` is the segment's
        achievement -> game turn."""
        turn = self.last_turn = game.turn
        if self._pending is not None:
            self.tails[turn] = self._pending
            self._pending = None
            for t in [t for t in self.tails if t < turn - AWAY_TURNS - 2]:
                del self.tails[t]
        self.unlocked = dict(unlocked)
        inv = _inventory(game, self.items)
        self.end_inv = inv
        for item, v in inv.items():
            self.max_inv[item] = max(self.max_inv[item], v)
        if self.ladder_floors:
            self.end_kills = _kills(game, self.ladder_floors)
            for f, k in self.end_kills.items():
                if k >= LADDER_KILLS:
                    self.ladder_open_at.setdefault(f, turn)
        for s in self.subgoals:
            for item, count, rel in s.items:
                if inv[item] >= self._target(item, count, rel):
                    self.item_reached.setdefault((s.n, item), turn)
            d = self._dist(s, game)
            self.end_dist[s.n] = d
            if d is not None and (s.n not in self.closest or d < self.closest[s.n][0]):
                self.closest[s.n] = (d, turn)
        active = self._active
        for s in self.subgoals:
            if s.n not in self.done_at and self._done(s):
                self.done_at[s.n] = turn
                if active is not None and s.n > active.n:
                    self._excerpt(turn, f"finished subgoal {s.n} before subgoal {active.n}")
        now = self.active()
        if now is not active:
            self._active, self._best, self._streak, self._armed = now, None, 0, True
        self._away(game, turn)
        self._observed += 1
        if self._observed % TRACK_EVERY == 0:
            self._track(game)

    def _away(self, game: Any, turn: int) -> None:
        s = self._active
        d = self._dist(s, game) if s is not None else None
        if d is None:
            self._streak = 0
            return
        if self._best is None or d < self._best:
            self._best, self._streak, self._armed = d, 0, True
        elif d > self._best:
            if self._streak == 0:
                self._away_from = turn
            self._streak += 1
            if self._armed and self._streak >= AWAY_TURNS and d >= self._best + AWAY_TILES:
                self._excerpt(self._away_from, f"moved away from subgoal {s.n}'s place {place(*s.place[:1], s.place[1:])} "
                                               f"for {self._streak} turns (closest {self._best} tiles, now {d})")
                self._armed = False
        else:
            self._streak = 0

    def _track(self, game: Any) -> None:
        s = self.active()
        where = place(game.floor, game.position)
        if s is None:
            note = "every subgoal done by code"
        else:
            d = self._dist(s, game)
            note = (f"active subgoal {s.n}, {d} tiles from its place" if d is not None else
                    f"active subgoal {s.n} (" + ("its place is a cell the agent never saw" if s.place and not s.place_seen else
                                                  f"its place is on F{s.place[0]}" if s.place else "no place given") + ")")
        self.track.append(f"turn {game.turn}: {where}, {note}")

    # --- the digest ------------------------------------------------------------------
    def status_lines(self) -> list[str]:
        out = []
        for s in self.subgoals:
            ev = []
            if s.achievement:
                a = s.achievement
                ev.append(f"{a}: " + ("already unlocked before this segment" if a in self.before else
                                      f"unlocked at turn {self.unlocked[a]}" if a in self.unlocked else "not unlocked"))
            for item, count, rel in s.items:
                target = self._target(item, count, rel)
                reached = self.item_reached.get((s.n, item))
                ev.append(f"{item} {self.start_inv[item]} at start, max {self.max_inv[item]}, {self.end_inv[item]} at end; "
                          f"target {target}: " + (f"reached at turn {reached}" if reached is not None else "not reached"))
            if s.ladder_floor is not None:
                f = s.ladder_floor
                start, end = self.start_kills.get(f, 0), self.end_kills.get(f, 0)
                ev.append(f"F{f}'s down-ladder: {start} kills at start, {end} at end ({LADDER_KILLS} open it): "
                          + ("open at the start" if start >= LADDER_KILLS else
                             f"opened at turn {self.ladder_open_at[f]}" if f in self.ladder_open_at else "still closed"))
            if s.place is not None:
                at = f"{s.thing or 'its place'} {place(s.place[0], s.place[1:])}"
                if not s.place_seen:
                    ev.append(f"{at}: a cell the agent never saw (not a real place it knew of: distances not tracked)")
                elif s.n in self.closest:
                    tiles, t = self.closest[s.n]
                    start, end = self.start_dist[s.n], self.end_dist[s.n]
                    ev.append(f"{at}: closest {tiles} tiles at turn {t} "
                              f"({'start ' + str(start) if start is not None else 'not on its floor at the start'}, "
                              f"{'end ' + str(end) if end is not None else 'not on its floor at the end'})")
                else:
                    ev.append(f"{at}: never on F{s.place[0]} this segment")
            done = self._done(s)
            verdict = ("already true at the start of this segment (not scored)" if s.n in self.done_at_start else
                       "no code check (no achievement, item count, ladder kills or place recognized)" if done is None and (s.place is None or not s.place_seen) else
                       "code cannot tell done (a place only)" if done is None else
                       f"done by code at turn {self.done_at[s.n]}" if done else "not done by code")
            out.append(f"{s.n}. {s.goal[:160]}\n   code: {'; '.join(ev) or '-'} -> {verdict}")
        return out

    def digest(self) -> dict[str, Any]:
        return {
            "parent_plan": plan_text(self.subgoals),
            "subgoal_status": "\n".join(self.status_lines()) or "(no subgoals)",
            "position_track": "\n".join(self.track) or "(no positions)",
            "decision_excerpts": "\n".join(f"-- turn {t} ({why}) --\n...{tail}" for t, why, tail in sorted(self.excerpts))
                                 or "(none: the agent never moved away from the active subgoal's place for long, "
                                    "finished subgoals out of order or died)",
            "subgoals": [{"n": s.n, "goal": s.goal, "achievement": s.achievement, "items": s.items,
                          "place": list(s.place) if s.place else None, "place_seen": s.place_seen,
                          "ladder_floor": s.ladder_floor,
                          "code_done": self._done(s), "done_at": self.done_at.get(s.n),
                          "done_at_start": s.n in self.done_at_start,
                          "done_late": s.n in self.done_at and s.n not in self.done_at_start
                                       and self.done_at[s.n] > self.last_turn - LATE_TURNS} for s in self.subgoals],
            "end_turn": self.last_turn,
            "excerpt_turns": [t for t, _, _ in sorted(self.excerpts)],
        }

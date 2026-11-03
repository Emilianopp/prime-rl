"""Segments for craftax-env: the game record, the notes turn and the continuation prompt.

docs/craftax-segments-judge.md is the design. A segment is one episode: it restores the
game a previous segment saved, plays until the cut, asks the policy for notes in the same
session, and saves the game. The next segment's protected first message is the system
prompt, a header, then the `summary`: the game's own record of the state at the cut plus
the policy's notes. The tree sampler (prime_rl ... samplers/tree.py) builds that prompt
itself from ``trace.info["craftax"]["summary"]``, with ``CONTINUATION_HEADER`` as its
``summary_header``; the in-env beam builds it with ``continuation_prompt``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from craftax_env.glyph import FLOOR_NAMES, FURNACE_BLOCK, TABLE_BLOCK, place

NOTES_PROTOCOL = """\
SEGMENTS
This game is played in segments of about 100 turns. On every normal turn, reply with one \
action as described above. When a message starts with "END OF SEGMENT", the game is paused \
and no action will be played: reply only with your notes for the next segment, inside \
<notes>...</notes>, and no <action> tag. The next segment starts from the same world with a \
fresh context that holds only this system prompt, the game's record of your state and those \
notes."""

CONTINUATION_HEADER = (
    "You are continuing a game already in progress. Below is the game's own record of your state, "
    "then the notes you wrote at the end of the previous segment of this same game; the world is "
    "exactly as you left it."
)

NOTES_OPENER = ("END OF SEGMENT. The game is paused: below is the last observation, then what to write. "
                "This reply is your notes, not a game turn.")

SUMMARY_PROMPT = """\
Your notes for the next segment. This reply is not a game turn: no action will be played, \
and the observation above is shown only so you can write your notes. The game continues from \
exactly this state in a fresh context, which will show the game's own record of your state \
(floor, position, vitals, inventory, equipment, XP, spells, achievements, deaths, the stations \
you placed), so do not copy it. Write what the record cannot hold, inside <notes>...</notes>, \
under these four headers:

MAP: one line per place that matters (water, trees, stone, coal, iron, diamond, sapphire, ruby, \
ladders, chests, fountains, enchantment tables, lava, where enemies gather). For a place in the \
[Grid] above, write its (row,col) in that grid: row 0 is the grid's top line, col 0 its first \
character, and you are the arrow at (4,5). For a place you saw earlier but cannot see now, write \
its direction and rough distance from you instead, e.g. "about 10 tiles west". Only places you \
actually saw.

FINDINGS: up to 5 facts you learned in this segment that the rules do not state, each tied to \
something that happened: what a failed craft or placement needed, what hurt or killed you and \
where, which actions did nothing. No general advice.

PLAN: 2 to 4 numbered subgoals in order, one per line: \
`N. <achievement or item count> at <place, or a direction to search> - done when <what the HUD will show>`. \
Plan only achievements that are not in [Achievements already unlocked] above, or item counts you \
do not hold yet.

SURVIVAL: one or two rules, each `if <vital or condition> then <what to do, where>`.

An example from a different game (its places and facts are not yours, do not copy them):
<notes>
MAP:
- table (5,3)
- stone (2,7) and (3,8)
- water (8,1)
- coal (0,9), behind the stone
- trees: a forest about 12 tiles east
FINDINGS:
1. make stone pickaxe failed at turn 41: it needs 1 wood and 1 stone, and I had no wood.
2. A zombie hit me twice at night on the open grass east of the table (HP 9 to 5).
PLAN:
1. 2 wood at the forest about 12 tiles east - done when Wood:2
2. make_stone_pickaxe at the table (5,3) - done when the HUD shows a stone pickaxe
3. collect_coal at (0,9) - done when Coal:1
SURVIVAL:
if drink < 4 then drink at the water (8,1)
if it is night then wall in with stone next to the table and sleep
</notes>

Be factual and specific. Reply with <notes>...</notes> only, no <action> tag."""

PLAN_SECTIONS = ("FINDINGS", "PLAN", "SURVIVAL")
"""The notes' sections under `code_facts`: the game writes the places and the event log itself."""

PLAN_PROMPT = """\
Your notes for the next segment. This reply is not a game turn: no action will be played. \
The game continues from exactly this state in a fresh context, which will show the game's own \
record of your state, the KNOWN PLACES, TURNS LOST and EVENT LOG above, so do not copy any of \
them. Write only what the game cannot: what you will do and why, inside <notes>...</notes>, under \
these three headers:

FINDINGS: up to 3 lessons, one line each: \
`N. <what went wrong, with its turn or place> because <the cause> -> <the fix> (in PLAN k or SURVIVAL k)`. \
If TURNS LOST counts 10 or more lost turns, finding 1 explains its first (largest) line. The cause \
must be what really happened in this game (the event log and the game's messages show it), not a \
guess about the rules. Carry each fix out in the plan: in how a subgoal is done (the way you go, \
what you face first) or as a SURVIVAL rule, and name where at the end. Do not add a subgoal just \
for a fix. No general advice; write "none" only if nothing went wrong.

PLAN: 2 to 4 numbered subgoals in order, one per line: \
`N. <achievement or item count> at <place> - done when <what the HUD will show>`. \
Take every place from KNOWN PLACES as written there, F<floor> (row,col); for something never \
seen, give a direction to search. Subgoal 1 is the step that is marked "needed next" unless a \
vital or a danger comes first, and then say so; your fixes change how you do the subgoals, not \
which comes first. Plan only achievements that are not in \
[Achievements already unlocked], or item counts you do not hold yet, and make sure each subgoal \
has its tools and station from the inventory or an earlier subgoal.

SURVIVAL: one or two rules, each `if <vital or condition> then <what to do, where>`, with places \
from KNOWN PLACES.

An example from a different game (its places and facts are not yours, do not copy them):
<notes>
FINDINGS:
1. Turns 30-52: 14 moves blocked by water near F0 (18,30) because I kept walking west into the lake -> walk around it by the south shore (in PLAN 1)
2. Turn 41: make_stone_pickaxe failed because I had used my last wood on a sword -> keep 1 wood for each craft (in PLAN 1)
3. Turns 60-75: zombies hit me on open grass at night far from the table -> be at the table by dusk (in SURVIVAL 2)
PLAN:
1. 2 wood from the trees near F0 (22,31), going by the south shore of the lake - done when Wood:2
2. make_stone_pickaxe at the table F0 (25,30) - done when the HUD shows a stone pickaxe
3. collect_iron at F0 (14,36) - done when Iron:1
SURVIVAL:
if drink < 4 then drink at the water F0 (27,26)
if it is night then wall in with stone next to the table F0 (25,30) and sleep
</notes>

Be specific. Reply with <notes>...</notes> only, no <action> tag."""

PLAN_RETRY = """\
END OF SEGMENT, still: the game is paused and no action will be played now. Write the notes \
for the next segment inside <notes>...</notes>, under the headers FINDINGS, PLAN and SURVIVAL as \
asked above, with places as KNOWN PLACES writes them. Do not output an <action> tag."""

EVENT_LOG_LINES = 15
EVENT_LOG_HEADER = "EVENT LOG of the previous segment (written by the game{tail})"


def event_log(events: list[str], n: int = EVENT_LOG_LINES) -> str:
    """The game's own log of the segment for the next one: unlocks, failed crafts and why, stations
    placed or destroyed, floors, sleeps, deaths (the timeline's lines, the last `n`)."""
    tail = events[-n:]
    more = f"; the last {len(tail)} of {len(events)} notable events" if len(events) > len(tail) else ""
    return "\n".join([EVENT_LOG_HEADER.format(tail=more), *(tail or ["(nothing notable happened)"])])


LOST_MIN = 10
"""Fewer lost turns than this: no finding has to explain them (and the judge gives no `waste` score)."""

LOST_KINDS = {
    "forfeit": "forfeited turns (no action played)",
    "bump": "moves into a blocked cell that did not even turn you",
    "turn": "turns in place that the next action did not use",
    "revisit": "moves back to a cell already visited since your last useful action (back and forth)",
    "do": "DO with no effect",
    "build": "failed crafts and placements",
    "idle": "NOOP or no visible effect",
}


class TurnsLost:
    """Turns that changed nothing, counted from the game's own feedback, by kind and detail. A blocked
    move that turns the player is a turn in place: lost unless the next action uses it (a DO, a craft,
    a placement that does something); a move on, or another turn, means it was not. A move back to a
    cell already visited since the last useful action is lost too (pacing back and forth); a useful
    action, a floor change or a respawn starts the count of visited cells again. Eating, drinking and
    sleeping (food, drink or energy up) are useful."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self._turned: str | None = None  # the pending turn in place
        self._visited: set | None = None  # cells walked through since the last useful action

    def add(self, kind: str, detail: str, n: int = 1) -> None:
        self.counts[(kind, detail)] = self.counts.get((kind, detail), 0) + n

    def forfeit(self, why: str) -> None:
        self._flush()
        self.add("forfeit", why)

    def _flush(self) -> None:
        if self._turned is not None:
            self.add("turn", self._turned)
            self._turned = None

    def observe(self, name: str, text: str, before: dict, after: dict) -> None:
        """One played action; `text` is `CraftaxGame.feedback`'s sentence for it."""
        what = text.split(": ", 1)[1] if ": " in text else text
        what = what.rstrip(".").split("; ")[0]  # drop the vitals' changes
        if after["floor"] != before["floor"]:
            self._turned = self._visited = None
            return
        if name.startswith("MOVE_"):
            self._flush()  # moving on (or turning again) did not use the last turn
            pos, start = tuple(after["pos"]), tuple(before["pos"])
            if pos == start:
                if after["facing"] == before["facing"]:
                    self.add("bump", what)
                else:
                    self._turned = what
                return
            if self._visited is None or (abs(pos[0] - start[0]) + abs(pos[1] - start[1]) > 1):  # a respawn jump
                self._visited = {start}
            if pos in self._visited:
                self.add("revisit", f"F{after['floor']} ({pos[0]},{pos[1]})")
            self._visited.add(pos)
            return
        vb, va = before.get("vitals") or {}, after.get("vitals") or {}
        refreshed = any(va.get(k, 0) > vb.get(k, 0) for k in ("food", "drink", "energy"))
        lost = None
        if refreshed:
            pass
        elif name == "DO" and what.startswith(("no effect", "nothing collected")):
            lost = ("do", what)
        elif what.startswith(("nothing made", "nothing placed")):
            lost = ("build", f"{name}: {what}")
        elif name == "NOOP" or what.startswith(("no visible effect", "no effect")):
            lost = ("idle", name)
        if lost is None:
            self._turned = self._visited = None  # this action used the turn, and the walk to it
        else:
            self._flush()
            self.add(*lost)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def by_kind(self) -> list[tuple[str, int, list[tuple[str, int]]]]:
        """[(kind, turns, [(detail, turns) largest first])], largest kind first."""
        kinds: dict[str, list[tuple[str, int]]] = {}
        for (kind, detail), n in self.counts.items():
            kinds.setdefault(kind, []).append((detail, n))
        out = [(k, sum(n for _, n in ds), sorted(ds, key=lambda d: -d[1])) for k, ds in kinds.items()]
        return sorted(out, key=lambda k: -k[1])

    def text(self, turns: int) -> str:
        head = f"TURNS LOST in the previous segment (counted by the game): {self.total} of {turns} turns changed nothing"
        rows = [f"- {n} {LOST_KINDS[k]}: " + ", ".join(f"{d} x{c}" for d, c in ds[:3]) + (", ..." if len(ds) > 3 else "")
                for k, n, ds in self.by_kind()]
        return "\n".join([head, *rows]) if rows else head

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {k: dict(ds) for k, _, ds in self.by_kind()}


def count_findings(lines: list[str]) -> int:
    """Numbered lines of a FINDINGS section ("none" or unnumbered prose: 0)."""
    return sum(1 for line in lines if re.match(r"\s*\d+\s*[.)]", line))


SUMMARY_RETRY = """\
END OF SEGMENT, still: the game is paused and no action will be played now. Write the notes \
for the next segment inside <notes>...</notes>, under the headers MAP, FINDINGS, PLAN and \
SURVIVAL as asked above, with places as (row,col) in the last [Grid] (you at (4,5)) or as a \
direction and distance from you. Do not output an <action> tag."""

NOTES_MIN_CHARS = 80
NOTES_RETRY_REASONS = ("action", "no_notes_tag", "short")
SECTIONS = ("MAP", "FINDINGS", "PLAN", "SURVIVAL")

_NOTES = re.compile(r"<notes>\s*(.*?)\s*(?:</notes>|$)", re.DOTALL | re.IGNORECASE)
_ACTION = re.compile(r"<\s*action\s*>.*?<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE)
_HEADER = re.compile(r"^\s*(?:#+\s*|\*\*)?(MAP|FINDINGS|PLAN|SURVIVAL)\s*(?:\*\*)?\s*:?\s*(?:\*\*)?\s*(.*)$", re.IGNORECASE)


# --- the system prompt -----------------------------------------------------------------


def with_notes_protocol(system_prompt: str | None) -> str:
    """The system prompt with NOTES_PROTOCOL after the game rules, before the record and
    notes of a previous segment (if any)."""
    sp = system_prompt or ""
    if NOTES_PROTOCOL in sp:
        return sp
    head, sep, tail = sp.partition(CONTINUATION_HEADER)
    if sep:
        return f"{head.rstrip()}\n\n---\n{NOTES_PROTOCOL}\n\n{CONTINUATION_HEADER}{tail}"
    return f"{sp}\n\n---\n{NOTES_PROTOCOL}"


def continuation_prompt(base: str, summary: str | None) -> str:
    """The protected first message of a continuing segment, as the tree sampler builds it
    (base, the header, the summary)."""
    body = summary.strip() if summary and summary.strip() else "(no record or notes were written)"
    return f"{base}\n\n{CONTINUATION_HEADER}\n\n{body}"


# --- the game record -------------------------------------------------------------------


STATION_BLOCKS = {"table": TABLE_BLOCK, "furnace": FURNACE_BLOCK}


def standing_stations(game: Any, stations: list[dict[str, Any]], floor: int | None = None) -> list[dict[str, Any]]:
    """The placed stations whose block still stands (on `floor` only, when given). Craftax
    turns a table or furnace into path when the player DOes facing it or an arrow hits it."""
    if not stations:
        return []
    keep = [s for s in stations if floor is None or s["floor"] == floor]
    if not keep:
        return []
    blocks = np.asarray(game.state.map if floor is None else game.state.map[floor])
    at = (lambda s: blocks[s["floor"], s["pos"][0], s["pos"][1]]) if floor is None else (lambda s: blocks[s["pos"][0], s["pos"][1]])
    return [s for s in keep if int(at(s)) == STATION_BLOCKS[s["kind"]]]


def game_record(
    game: Any,
    *,
    stations: list[dict[str, Any]],
    record_stations: bool,
    depth: int,
    first_turn: int,
    deaths: list[dict[str, Any]],
    death_inventory: str,
    previous: bool = True,
) -> str:
    """The env-written block at the top of the next segment: the HUD as the game prints it
    (without its turn counter), the achievements by name, the stations the agent placed
    that still stand, and what the segment just played did to the player (deaths, respawns)."""
    hud = [line for line in game.hud().splitlines() if not line.startswith("Step:")]
    achievements = sorted(game.achievements)
    total = len(game._rt["achievements"])
    lines = ["GAME RECORD (written by the game at the start of this segment, not by you)",
             f"You are at {place(game.floor, game.position)} ({floor_name(game.floor)}). "
             "Places are written F<floor> (row,col): the HUD's Floor and Pos.",
             f"Game step: {game.timestep}  (model turns so far: {game.turn})", *hud,
             f"Achievements ({len(achievements)}/{total}): {', '.join(achievements) or 'none'}"]
    if record_stations:
        shown = [f"{s['kind']} {place(s['floor'], s['pos'])}" for s in standing_stations(game, stations)[-8:]]
        lines.append(f"Stations you placed: {', '.join(shown) if shown else 'none'}")
    if not previous:  # the record at a segment's start (the judge's "before")
        return "\n".join(lines)
    lost = {"drop_all": "inventory lost", "drop_materials": "materials lost, tools and armour kept", "keep": "inventory kept"}
    died = "; ".join(
        f"turn {d['turn']} at {place(d['floor'], d['pos'])}, respawned at {place(0, d['respawn'])}"
        for d in deaths[-5:]
    )
    lines.append(
        f"Previous segment: segment {depth}, turns {first_turn + 1}-{game.turn}, "
        + (f"{len(deaths)} death{'s' if len(deaths) != 1 else ''} ({died}; {lost.get(death_inventory, death_inventory)})" if deaths else "no deaths")
    )
    return "\n".join(lines)


class Timeline:
    """The segment's events for the judge, one line each; a run of identical events (the
    same failed craft five times) becomes one line with its turn span and count."""

    def __init__(self, limit: int = 80) -> None:
        self.items: list[list[Any]] = []  # [first turn, last turn, text, count]
        self.limit = limit

    def add(self, turn: int, text: str) -> None:
        if self.items and self.items[-1][2] == text:
            self.items[-1][1], self.items[-1][3] = turn, self.items[-1][3] + 1
        elif len(self.items) < self.limit:
            self.items.append([turn, turn, text, 1])

    def lines(self) -> list[str]:
        return [f"turn {a}: {t}" if n == 1 else f"turns {a}-{b}: {t} (x{n})" for a, b, t, n in self.items]


NOTES_HEADER = "YOUR NOTES FROM THE PREVIOUS SEGMENT:"
NOTES_FRAME = ("(Places written (row,col) in these notes are cells of the [Grid] of your first observation "
               "in this segment, where you are the arrow at (4,5); they shift as soon as you move.)")
FACTS_NOTES_FRAME = ("(Places in these notes are F<floor> (row,col) as KNOWN PLACES writes them: the HUD's Floor "
                     "and Pos, fixed on the map. Your [Grid] and HUD are always the truth; these notes and places "
                     "may be stale (a station can be destroyed, creatures move). If your recent moves keep "
                     "returning to the same cells, stop and try another way.)")


def summary_text(record: str, notes: str | None, facts: str | None = None) -> str:
    """What the next segment starts from: the record, the game-written facts (`code_facts`: the
    known places and the event log), then the notes. Its first observation is the view the notes
    were written in, so their view-frame places read off it; under `code_facts` the notes take their
    places from KNOWN PLACES, map places that do not shift."""
    notes = (notes or "").strip()
    head = f"{record}\n\n{facts.strip()}" if facts else record
    if not notes:
        return f"{head}\n\n{NOTES_HEADER}\n(you wrote no notes)"
    return f"{head}\n\n{NOTES_HEADER}\n{FACTS_NOTES_FRAME if facts else NOTES_FRAME}\n{notes}"


def notes_of_summary(summary: str | None) -> str:
    """The parent's notes inside a summary (what `summary_text` wrote), else ""."""
    _, sep, notes = (summary or "").partition(NOTES_HEADER)
    notes = notes.strip().removeprefix(NOTES_FRAME).removeprefix(FACTS_NOTES_FRAME).strip()
    return "" if not sep or notes == "(you wrote no notes)" else notes


# --- the notes -------------------------------------------------------------------------


def extract_notes(content: str | None, reasoning: str | None = None) -> str:
    """The notes of a notes-turn reply: inside <notes> in the content, else inside <notes>
    in the reasoning (the model sometimes ends its turn inside its thinking), else the
    content minus any action tags."""
    for text in (content, reasoning):
        m = _NOTES.search(text or "")
        if m and m.group(1).strip():
            return m.group(1).strip()
    return _ACTION.sub("", content or "").strip()


def notes_problem(content: str | None, reasoning: str | None, notes: str) -> str | None:
    """Why a notes-turn reply gets the retry, else None: "action" (an action instead of
    notes), "no_notes_tag" (text without a <notes> tag) or "short" (under 80 characters).
    Notes of 80+ characters inside a <notes> tag (in the content or the reasoning) pass, and
    an action after them is ignored."""
    tagged = any(_NOTES.search(t or "") for t in (content, reasoning))
    if tagged and len(notes) >= NOTES_MIN_CHARS:
        return None
    if not tagged and _ACTION.search(content or ""):
        return "action"
    if not tagged:
        return "no_notes_tag"
    return "short"


def parse_notes(notes: str) -> dict[str, list[str]]:
    """The four sections of the notes, line by line (headers case-insensitive, markdown
    decoration tolerated; text before the first header goes under ""; a missing section is
    an empty list)."""
    out: dict[str, list[str]] = {"": [], **{s: [] for s in SECTIONS}}
    current = ""
    for raw in (notes or "").splitlines():
        m = _HEADER.match(raw)
        if m:
            current = m.group(1).upper()
            rest = m.group(2).strip()
            if rest:
                out[current].append(rest)
            continue
        line = raw.strip()
        if line:
            out[current].append(line)
    return out


def notes_format(sections: dict[str, list[str]], expected: tuple[str, ...] = SECTIONS) -> float:
    """The fraction of the expected sections (the four, or PLAN_SECTIONS) that hold at least one line."""
    return sum(bool(sections.get(s)) for s in expected) / len(expected)


def floor_name(floor: int) -> str:
    return FLOOR_NAMES[floor] if 0 <= floor < len(FLOOR_NAMES) else f"Floor {floor}"

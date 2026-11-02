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

from craftax_env.glyph import FLOOR_NAMES, place

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
    and what the segment just played did to the player (deaths, respawns)."""
    hud = [line for line in game.hud().splitlines() if not line.startswith("Step:")]
    achievements = sorted(game.achievements)
    total = len(game._rt["achievements"])
    lines = ["GAME RECORD (written by the game at the start of this segment, not by you)",
             f"You are at {place(game.floor, game.position)} ({floor_name(game.floor)}). "
             "Places are written F<floor> (row,col): the HUD's Floor and Pos.",
             f"Game step: {game.timestep}  (model turns so far: {game.turn})", *hud,
             f"Achievements ({len(achievements)}/{total}): {', '.join(achievements) or 'none'}"]
    if record_stations:
        shown = [f"{s['kind']} {place(s['floor'], s['pos'])}" for s in stations[-8:]]
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


def summary_text(record: str, notes: str | None) -> str:
    """What the next segment starts from: the record, then the notes. Its first observation
    is the view the notes were written in, so their view-frame places read off it."""
    notes = (notes or "").strip()
    if not notes:
        return f"{record}\n\n{NOTES_HEADER}\n(you wrote no notes)"
    return f"{record}\n\n{NOTES_HEADER}\n{NOTES_FRAME}\n{notes}"


def notes_of_summary(summary: str | None) -> str:
    """The parent's notes inside a summary (what `summary_text` wrote), else ""."""
    _, sep, notes = (summary or "").partition(NOTES_HEADER)
    notes = notes.strip().removeprefix(NOTES_FRAME).strip()
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


def notes_format(sections: dict[str, list[str]]) -> float:
    """The fraction of the four sections that hold at least one line."""
    return sum(bool(sections.get(s)) for s in SECTIONS) / len(SECTIONS)


def floor_name(floor: int) -> str:
    return FLOOR_NAMES[floor] if 0 <= floor < len(FLOOR_NAMES) else f"Floor {floor}"

"""Round 2 variants: v7_combo (round-1 leader) as the control, and mutations aimed at
round 1's failure modes: viewport-relative coordinates, append-only logs and throwaway
keys, tags written inside the reasoning (ignored), fewer replies (long reasoning)."""
import re

r1 = {}
for name in ("v3_checklist", "v5_terse", "v7_combo"):
    r1[name] = open(f"r1/{name}.toml").read()


def field(toml, key):
    m = re.search(rf"^env\.player\.memory\.{key} = '''\n(.*?)'''", toml, re.S | re.M)
    return m.group(1) if m else None


V7_PROMPT, V7_TURN, V7_SUM = field(r1["v7_combo"], "prompt"), field(r1["v7_combo"], "turn_note"), field(r1["v7_combo"], "summary_note")
V3_PROMPT, V3_TURN = field(r1["v3_checklist"], "prompt"), field(r1["v3_checklist"], "turn_note")
V5_PROMPT = field(r1["v5_terse"], "prompt")

COORDS = '''POSITIONS: always store absolute positions, never positions on your screen (the screen moves with you). The HUD's Pos (R,C) is your row and column: MOVE_UP lowers R, MOVE_DOWN raises it, MOVE_LEFT lowers C, MOVE_RIGHT raises it. The [Grid] is 9 rows by 11 columns with you in the middle (row 4, column 5, counting from 0 at the top left), so a cell at grid row r and column c is at absolute (R + r - 4, C + c - 5). Example: at Pos (20,23), a tree in the grid's top row (r=0) and fourth column (c=3) is at (16,21).'''
VISIBLE = '''Write memory tags only in your visible reply, after you have finished thinking: tags written while thinking are thrown away.'''
TIDY = '''KEEP MEMORY TIDY: use at most 6 entries. MAP is a list with one line per place ("table (24,22)"); when a place changes or turns out wrong, fix its line with <replace>, never append a duplicate. Rewrite STATUS and PLAN instead of adding to them. Do not make an entry for a single event.'''
BRIEF = '''Keep your thinking short (a few sentences): long thinking runs out of tokens and loses the turn.'''
MUST = '''Your reply must contain at least one memory edit (<write>, <rewrite> or <replace>) before its action tags: add what you just learned, with absolute positions.'''
SCHEMA_TERSE = '''Keep exactly these entries: MAP (one line per place, absolute positions), STATUS (tools, inventory, achievements), PLAN (goal and next steps).'''


def ml(s):
    assert "'''" not in s
    return "'''\n" + s + "'''"


def variant(prompt, turn=None, summary=None):
    out = f"env.player.memory.prompt = {ml(prompt)}\n"
    if turn:
        out += f"env.player.memory.turn_note = {ml(turn)}\n"
    if summary:
        out += f"env.player.memory.summary_note = {ml(summary)}\n"
    return out


V = {
    "c0_v7": r1["v7_combo"].split("\n", 1)[1],
    "a_v7_coords_visible": variant(V7_PROMPT + "\n" + COORDS + "\n" + VISIBLE, V7_TURN, V7_SUM),
    "b_v7_coords_tidy": variant(V7_PROMPT + "\n" + COORDS + "\n" + TIDY + "\n" + VISIBLE, V7_TURN, V7_SUM),
    "c_v3_coords_visible": variant(V3_PROMPT + "\n" + COORDS + "\n" + VISIBLE, V3_TURN),
    "d_terse_schema_coords": variant(V5_PROMPT + "\n" + SCHEMA_TERSE + "\n" + COORDS + "\n" + VISIBLE),
    "e_v7_brief": variant(V7_PROMPT + "\n" + VISIBLE + "\n" + BRIEF, V7_TURN, V7_SUM),
    "f_v7_coords_tidy_brief": variant(V7_PROMPT + "\n" + COORDS + "\n" + TIDY + "\n" + VISIBLE + "\n" + BRIEF, V7_TURN, V7_SUM),
    "g_v3_must_coords": variant(V3_PROMPT + "\n" + COORDS + "\n" + VISIBLE, V3_TURN + " " + MUST),
}
for k, v in V.items():
    open(f"r2/{k}.toml", "w").write(f"# memopt round 2: {k}\n" + v)
print(sorted(V))

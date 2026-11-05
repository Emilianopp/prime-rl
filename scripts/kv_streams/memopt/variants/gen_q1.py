"""Quality search, round 1 (q1): the memory is an option, not a duty - no "edit every
reply" - and the score is Gemini's grade of the memory at each reset (judge_memory.py),
with play (achievements, replies) and tag tokens per reply as guards against memory
edits eating the turn."""
import tomllib


def load(path):
    return tomllib.loads("[[source]]\n" + open(path).read())["source"][0]["env"]["player"]["memory"]


F = load("r4/f.toml")
BASE = F["prompt"].split("\nWHY YOUR MEMORY MATTERS")[0].replace(
    " After a context reset, your first reply must read every entry (one <read> tag per key) before doing anything else.", "")
COORDS = F["prompt"].split("\nPOSITIONS: ")[1].split("\n")[0]
COORDS = "POSITIONS: " + COORDS

OPTIONAL = '''Memory is optional: most turns need no edit. Write only when you learn something you will still need after your context is reset.'''
WORTH = '''WHAT IS WORTH REMEMBERING: things you cannot see again from your screen and will need later - where resources, water, your table and furnace and ladders are (absolute positions), rules and recipes you confirmed, what failed and why, what hurt you and where, your plan. NOT worth remembering: what is on your screen now, your vitals or inventory (the HUD shows them), step-by-step logs of your moves.'''
SCHEMA = '''A good layout: MAP (one line per place, absolute positions), RULES (confirmed mechanics and recipes), LESSONS (what failed and why, dangers), PLAN (goal and next steps). Fix wrong or outdated lines with <replace> or <rewrite> instead of adding new ones.'''
BAR = '''Before writing a fact, ask: will it still be true and useful 100 turns from now? If not, leave it out.'''
RESET_READ = '''After a context reset, read the entries you need (MAP and PLAN first) before acting.'''
CONSOLIDATE_NOTE = '''Your memory is kept across the reset. Use this reply to make it the best guide for your next self: fix wrong lines, add places and lessons you have not saved yet, drop what is outdated. In the summary, say which entries to read first.'''
QUIET_TURN = '''You may also edit your memory in this reply if you learned something worth keeping.'''


def ml(s):
    assert "'''" not in s
    return "'''\n" + s + "'''"


def emit(name, prompt=None, turn=None, summary=None, no_turn=False):
    out = f"# memopt quality round 1: {name}\n"
    if prompt:
        out += f"env.player.memory.prompt = {ml(prompt)}\n"
    if no_turn:
        out += 'env.player.memory.turn_note = ""\n'
    elif turn:
        out += f"env.player.memory.turn_note = {ml(turn)}\n"
    if summary:
        out += f"env.player.memory.summary_note = {ml(summary)}\n"
    open(f"q1/{name}.toml", "w").write(out)


W = BASE + "\n" + OPTIONAL + "\n" + WORTH + "\n" + COORDS
emit("q0_default")
emit("q1_f_optional", F["prompt"].replace(" Players who write to memory on almost every turn do far better, because they never search twice for the same thing.", "") + "\n" + OPTIONAL,
     turn=F["turn_note"].replace(" Write at least one memory tag in every reply.", ""), summary=F["summary_note"])
emit("q2_worth", W, turn=QUIET_TURN)
emit("q3_worth_schema", W + "\n" + SCHEMA, turn=QUIET_TURN)
emit("q4_worth_consolidate", W + "\n" + SCHEMA, turn=QUIET_TURN, summary=CONSOLIDATE_NOTE)
emit("q5_worth_noturn", W + "\n" + SCHEMA, no_turn=True, summary=CONSOLIDATE_NOTE)
emit("q6_worth_schema_read", W + "\n" + SCHEMA + "\n" + RESET_READ, turn=QUIET_TURN, summary=CONSOLIDATE_NOTE)
emit("q7_worth_bar", W + "\n" + SCHEMA + "\n" + BAR, turn=QUIET_TURN, summary=CONSOLIDATE_NOTE)

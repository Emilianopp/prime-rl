"""Quality search, round 2 (q2). Round 1: when the model keeps a memory it is decent
(judge usefulness 6-7.4 when non-empty); the score is driven by how often the memory is
empty at a reset, and "only write what is worth it" made it empty far more often. Lever
here: the reset itself. The summary reply is not a game turn, so memory edits there cost
no play - ask for the memory update at the reset instead of during turns."""
import tomllib


def load(path):
    return tomllib.loads("[[source]]\n" + open(path).read())["source"][0]["env"]["player"]["memory"]


Q1 = load("q1/q1_f_optional.toml")
Q6 = load("q1/q6_worth_schema_read.toml")
WORTH = [l for l in Q6["prompt"].split("\n") if l.startswith("WHAT IS WORTH REMEMBERING")][0]
WHY = '''WHY YOUR MEMORY MATTERS
You see only your last 50 replies, and every ~25 replies your whole context is reset: after a reset you keep only your handoff summary and your memory. Every fact you did not save is gone: where the trees, water, stone, your table and furnace are, which actions worked, what hurt you.'''
HANDOFF = '''This reply is your moment to save what matters, and it costs no turn: before the summary, update your memory with what you learned in this context - add new places to MAP (absolute positions), confirmed rules to RULES, failures and dangers to LESSONS, and rewrite PLAN. Fix or delete lines that turned out wrong. The summary reaches only your next context; your memory lasts the whole game. In the summary, say which entries to read first.'''
LAYOUT = '''A good layout: MAP (one line per place, absolute positions), RULES (confirmed mechanics and recipes), LESSONS (what failed and why, dangers), PLAN (goal and next steps).'''


def ml(s):
    assert "'''" not in s
    return "'''\n" + s + "'''"


def emit(name, prompt=None, turn=None, summary=None, no_turn=False):
    out = f"# memopt quality round 2: {name}\n"
    if prompt:
        out += f"env.player.memory.prompt = {ml(prompt)}\n"
    if no_turn:
        out += 'env.player.memory.turn_note = ""\n'
    elif turn:
        out += f"env.player.memory.turn_note = {ml(turn)}\n"
    if summary:
        out += f"env.player.memory.summary_note = {ml(summary)}\n"
    open(f"q2/{name}.toml", "w").write(out)


emit("a_q1_control", Q1["prompt"], Q1["turn_note"], Q1["summary_note"])
emit("b_q6_control", Q6["prompt"], Q6["turn_note"], Q6["summary_note"])
emit("c_q6_handoff", Q6["prompt"], Q6["turn_note"], HANDOFF)
emit("d_q1_handoff", Q1["prompt"], Q1["turn_note"], HANDOFF)
emit("e_default_handoff", None, None, HANDOFF.replace(" - add new places to MAP (absolute positions), confirmed rules to RULES, failures and dangers to LESSONS, and rewrite PLAN", ": where things are, rules you confirmed, what failed, your plan") + "")
emit("f_q6_handoff_noturn", Q6["prompt"], None, HANDOFF, no_turn=True)
emit("g_q6_why_handoff", Q6["prompt"].replace(WORTH, WHY + "\n" + WORTH), Q6["turn_note"], HANDOFF)
emit("h_q1_worth_handoff", Q1["prompt"] + "\n" + WORTH + "\n" + LAYOUT, Q1["turn_note"], HANDOFF)

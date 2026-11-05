"""Quality search, round 3 (q3), around q2's winner g_q6_why_handoff (judge usefulness
6.67 vs 5.59 / 4.81 for the controls, memory empty at 3/64 resets, 0.08 edits per reply):
the why-section plus the free memory update at the reset. Mutations: positions (only ~34%
within a tile), writing only at resets, the why-reminder at the reset, off-screen focus,
and whether why + handoff alone lift the default prompt."""
import tomllib


def load(path):
    return tomllib.loads("[[source]]\n" + open(path).read())["source"][0]["env"]["player"]["memory"]


G = load("q2/g_q6_why_handoff.toml")
WHY = G["prompt"][G["prompt"].index("WHY YOUR MEMORY MATTERS"):].split("\nWHAT IS WORTH")[0]
ADJ = '''Record a position when you are next to the thing: it is your Pos plus one step in the direction you face (facing up: row - 1; down: row + 1; left: column - 1; right: column + 1). Positions you work out from far away on the grid are often wrong.'''
OFFSCREEN = '''Prefer what your screen cannot show you after the reset: places that will be off-screen, rules you had to discover, mistakes you should not repeat.'''
WHY_AT_RESET = '''Everything you learned in this context that is not in your memory is about to be lost. '''


def ml(s):
    assert "'''" not in s
    return "'''\n" + s + "'''"


def emit(name, prompt=None, turn=None, summary=None, no_turn=False):
    out = f"# memopt quality round 3: {name}\n"
    if prompt:
        out += f"env.player.memory.prompt = {ml(prompt)}\n"
    if no_turn:
        out += 'env.player.memory.turn_note = ""\n'
    elif turn:
        out += f"env.player.memory.turn_note = {ml(turn)}\n"
    if summary:
        out += f"env.player.memory.summary_note = {ml(summary)}\n"
    open(f"q3/{name}.toml", "w").write(out)


P, T, S = G["prompt"], G["turn_note"], G["summary_note"]
DEFAULT_TAGS = P.split("\nMemory is optional")[0]  # the default memory section, as in every q variant
emit("g0_control", P, T, S)
emit("g1_noturn", P, None, S, no_turn=True)
emit("g2_adjpos", P + "\n" + ADJ, T, S)
emit("g3_whyreset", P, T, WHY_AT_RESET + S)
emit("g4_offscreen", P + "\n" + OFFSCREEN, T, S)
emit("g5_default_why_handoff", DEFAULT_TAGS + "\n" + WHY, None, S)
emit("g6_adj_whyreset", P + "\n" + ADJ, T, WHY_AT_RESET + S)
emit("g7_adj_whyreset_offscreen", P + "\n" + ADJ + "\n" + OFFSCREEN, T, WHY_AT_RESET + S)

"""Round 3 variants: f_v7_coords_tidy_brief (round-2 balance of memory use, absolute
positions and achievements) as the control; mutations for the two failures left: ~1/3 of
tags written inside the reasoning (ignored), and few reads after a reset in the
checklist family."""
import re
import tomllib

def load(path):
    d = tomllib.loads("[[source]]\n" + open(path).read())
    return d["source"][0]["env"]["player"]["memory"]

F = load("r2/f_v7_coords_tidy_brief.toml")
B = load("r2/b_v7_coords_tidy.toml")
G = load("r2/g_v3_must_coords.toml")

THINK_WORDS = '''While thinking, decide your memory edits in plain words only: never type <write>, <rewrite>, <replace>, <read> or <delete> while thinking. Type the tags once, after your thinking ends, in your visible reply just before your action tags.'''
MUST = '''Your reply must contain at least one memory edit (<write>, <rewrite> or <replace>) before its action tags: add what you just learned, with absolute positions.'''
RESET_READ = '''After a context reset, your first reply must begin with <read key="MAP"/> <read key="PLAN"/> <read key="STATUS"/> before anything else.'''
SHORT_TURN = '''Memory (in your visible reply, before actions): new place? fix MAP with its absolute position. New rule? add to RULES. Goal or inventory changed? rewrite PLAN or STATUS.'''


def ml(s):
    assert "'''" not in s
    return "'''\n" + s + "'''"


def emit(name, m):
    out = f"# memopt round 3: {name}\n"
    for k in ("prompt", "turn_note", "summary_note"):
        if m.get(k):
            out += f"env.player.memory.{k} = {ml(m[k])}\n"
    open(f"r3/{name}.toml", "w").write(out)


def with_(m, **kw):
    m = dict(m)
    for k, v in kw.items():
        m[k] = v
    return m


emit("c0_f", F)
emit("h_f_must", with_(F, turn_note=F["turn_note"] + " " + MUST))
emit("i_f_resetread", with_(F, prompt=F["prompt"] + "\n" + RESET_READ))
emit("j_f_thinkwords", with_(F, prompt=F["prompt"] + "\n" + THINK_WORDS, turn_note=F["turn_note"] + " " + THINK_WORDS.split(". ")[0] + "."))
emit("k_f_all", with_(F, prompt=F["prompt"] + "\n" + RESET_READ + "\n" + THINK_WORDS, turn_note=F["turn_note"] + " " + MUST))
emit("l_b_thinkwords", with_(B, prompt=B["prompt"] + "\n" + THINK_WORDS))
emit("m_f_shortturn_must", with_(F, turn_note=SHORT_TURN + " " + MUST))
emit("n_g_reset_tidy_thinkwords", with_(G, prompt=G["prompt"] + "\n" + RESET_READ + "\n" + THINK_WORDS, summary_note=F["summary_note"]))

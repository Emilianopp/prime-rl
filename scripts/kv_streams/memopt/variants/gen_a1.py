"""Ahead check (a1): does showing the faced cell's absolute position in the HUD
(env.ahead_in_hud) fix remembered positions (~20% within a tile with g7)? g7 with and
without it, g7 with its position rules rewritten around the HUD's Ahead, the default
prompt with it."""
import tomllib

G7 = open("q4/g7.toml").read().split("\n", 1)[1]
mem = tomllib.loads("[[source]]\n" + G7)["source"][0]["env"]["player"]["memory"]
P = mem["prompt"]
coords = [l for l in P.split("\n") if l.startswith("POSITIONS: ")][0]
adj = [l for l in P.split("\n") if l.startswith("Record a position when you are next to the thing")][0]
AHEAD = '''POSITIONS: always store absolute positions, never positions on your screen. The HUD's "Ahead: (row,col) thing" names the cell in front of you and its absolute position: to remember where something is, face it and copy that position.'''
P2 = P.replace(coords, AHEAD).replace("\n" + adj, "")
assert P2 != P and "Ahead" in P2


def ml(s):
    return "'''\n" + s + "'''"


ON = "env.ahead_in_hud = true\n"
open("a1/g7.toml", "w").write("# ahead check: g7, HUD unchanged\n" + G7)
open("a1/g7_hud.toml", "w").write("# ahead check: g7 + Ahead in the HUD\n" + ON + G7)
out = "# ahead check: g7 with position rules around the HUD's Ahead\n" + ON + f"env.player.memory.prompt = {ml(P2)}\n"
out += f"env.player.memory.turn_note = {ml(mem['turn_note'])}\n" if mem.get("turn_note") else ""
out += f"env.player.memory.summary_note = {ml(mem['summary_note'])}\n" if mem.get("summary_note") else ""
open("a1/g7_ahead.toml", "w").write(out)
open("a1/default_hud.toml", "w").write("# ahead check: default prompts + Ahead in the HUD\n" + ON)

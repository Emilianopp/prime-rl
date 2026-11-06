"""Visible-list check (a3): neither the HUD's Ahead (a1) nor grid labels (a2) fixed
remembered positions (~25% within a tile): the model notes things seen at a distance and
cannot index glyph columns. env.visible_list adds a [Visible] line listing what is on screen
with absolute positions, so remembering a place is a copy."""
import tomllib

G7 = open("q4/g7.toml").read().split("\n", 1)[1]
mem = tomllib.loads("[[source]]\n" + G7)["source"][0]["env"]["player"]["memory"]
P = mem["prompt"]
coords = [l for l in P.split("\n") if l.startswith("POSITIONS: ")][0]
adj = [l for l in P.split("\n") if l.startswith("Record a position when you are next to the thing")][0]
VIS = '''POSITIONS: always store absolute positions, and copy them from the [Visible] line, which lists what is on your screen with its absolute (row,col) position, nearest first. Do not work positions out from the grid yourself.'''
P2 = P.replace(coords, VIS).replace("\n" + adj, "")
assert P2 != P


def ml(s):
    return "'''\n" + s + "'''"


ON = "env.visible_list = true\n"
open("a3/g7.toml", "w").write("# visible check: g7, no list\n" + G7)
open("a3/g7_visible.toml", "w").write("# visible check: g7 + [Visible]\n" + ON + G7)
out = "# visible check: g7 with position rules around [Visible]\n" + ON + f"env.player.memory.prompt = {ml(P2)}\n"
out += f"env.player.memory.turn_note = {ml(mem['turn_note'])}\n" + f"env.player.memory.summary_note = {ml(mem['summary_note'])}\n"
open("a3/g7_vis.toml", "w").write(out)
open("a3/default_visible.toml", "w").write("# visible check: default prompts + [Visible]\n" + ON)

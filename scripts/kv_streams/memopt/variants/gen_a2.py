"""Grid-label check (a2): the HUD's Ahead (a1) did not fix remembered positions, because
the player mostly notes things it sees at a distance on the grid. env.grid_coords labels
the grid's rows and columns with absolute positions. g7 with and without labels, g7 with its
position rules rewritten around the labels, the default prompt with labels."""
import tomllib

G7 = open("q4/g7.toml").read().split("\n", 1)[1]
mem = tomllib.loads("[[source]]\n" + G7)["source"][0]["env"]["player"]["memory"]
P = mem["prompt"]
coords = [l for l in P.split("\n") if l.startswith("POSITIONS: ")][0]
adj = [l for l in P.split("\n") if l.startswith("Record a position when you are next to the thing")][0]
GRID = '''POSITIONS: always store absolute positions. The [Grid] is labeled with them: the number at the left of a row is its absolute row, the two digits above a column (tens over ones) its absolute column. Read a thing's position off the labels as (row,col), e.g. a tree in row 19 under the column labeled 2/3 is at (19,23).'''
P2 = P.replace(coords, GRID).replace("\n" + adj, "")
assert P2 != P


def ml(s):
    return "'''\n" + s + "'''"


ON = "env.grid_coords = true\n"
open("a2/g7.toml", "w").write("# grid check: g7, unlabeled grid\n" + G7)
open("a2/g7_labels.toml", "w").write("# grid check: g7 + labeled grid\n" + ON + G7)
out = "# grid check: g7 with position rules around the grid labels\n" + ON + f"env.player.memory.prompt = {ml(P2)}\n"
out += f"env.player.memory.turn_note = {ml(mem['turn_note'])}\n" + f"env.player.memory.summary_note = {ml(mem['summary_note'])}\n"
open("a2/g7_grid.toml", "w").write(out)
open("a2/default_labels.toml", "w").write("# grid check: default prompts + labeled grid\n" + ON)

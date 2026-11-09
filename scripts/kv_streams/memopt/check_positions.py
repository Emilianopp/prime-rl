"""Check the positions players wrote to memory against the true map, without a judge: for
every "(row,col)" in a memory snapshot at a reset (judge_memory.snapshots) whose line names
a static thing (tree, water, stone, coal, iron, diamond, lava, sand, table, furnace, ladder),
is that thing on the floor's map within 1 tile? Only the current floor's map is known, so
positions are checked only while the player is on the Overworld.

usage: check_positions.py ROUND_OUT_DIR
"""

import collections
import glob
import os
import re
import sys

from judge_memory import snapshots

GLYPH = {"tree": "♣", "water": "≈", "stone": "S", "coal": "C", "iron": "I", "diamond": "D", "lava": "♨",
         "sand": "░", "table": "t", "furnace": "f", "ladder": "▼"}
WORD = re.compile(r"\b(" + "|".join(GLYPH) + r")s?\b", re.I)
PAIR = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


def grid(state):
    rows = {}
    for line in state["map"].splitlines()[2:]:
        m = re.match(r"\s*(\d+) (.*)", line)
        if m:
            rows[int(m.group(1))] = m.group(2)
    return rows


def near(rows, r, c, g):
    return any(rows.get(rr, "")[cc:cc + 1] == g for rr in range(r - 1, r + 2) for cc in range(max(0, c - 1), c + 2))


def main():
    out = sys.argv[1]
    print("variant".ljust(28) + "checked  within1  exact   by thing (within 1 / checked)")
    for vdir in sorted(glob.glob(f"{out}/*/")):
        n = ok = exact = 0
        per = collections.Counter(), collections.Counter()
        for _world, _k, entries, state in snapshots(vdir.rstrip("/")):
            if "Overworld" not in state["hud"]:
                continue
            rows = grid(state)
            for text in entries.values():
                for piece in re.split(r"[;\n]|,\s(?=[A-Za-z])", text):
                    w = WORD.search(piece)
                    if not w:
                        continue
                    g = GLYPH[w.group(1).lower()]
                    for a, b in PAIR.findall(piece):
                        r, c = int(a), int(b)
                        n += 1
                        per[1][w.group(1).lower()] += 1
                        if near(rows, r, c, g):
                            ok += 1
                            per[0][w.group(1).lower()] += 1
                        exact += rows.get(r, "")[c:c + 1] == g
        things = " ".join(f"{t} {per[0][t]}/{per[1][t]}" for t, _ in per[1].most_common(5))
        print(f"{os.path.basename(vdir.rstrip('/')):28}{n:7}  {100 * ok / max(1, n):6.0f}%  {100 * exact / max(1, n):5.0f}%   {things}")


if __name__ == "__main__":
    main()

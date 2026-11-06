"""Memory reads per block: anywhere in a segment, right after a reset, and by reset number
within the game. usage: reads.py LABEL:BLOCK_DIR ..."""
import collections, sys
sys.path.insert(0, sys.path[0])
from cmp import block, txt, OPS

print("blk   segs  reads/100rep  %segs_any_read  %reset_read_first3 | by reset#: 1  2  3  4-6  7+ (% reading in first 3, n)")
for spec in sys.argv[1:]:
    label, bdir = spec.split(":")
    segs = block(bdir)
    rep = sum(s["replies"] for s in segs)
    later = [s for s in segs if s["t0"] not in (0, 1, None) and s["opened_with_mem"]]
    games = collections.defaultdict(list)
    for s in segs:
        if s["t0"] is not None:
            games[s["idx"]].append(s)
    byk = collections.defaultdict(list)
    for ss in games.values():
        ss.sort(key=lambda s: s["t0"])
        for k, s in enumerate(ss):
            if k and s["opened_with_mem"]:
                byk["1" if k == 1 else "2" if k == 2 else "3" if k == 3 else "4-6" if k <= 6 else "7+"].append(s["read_first3"])
    cells = "  ".join(f"{100 * sum(byk[k]) / max(1, len(byk[k])):3.0f}% n{len(byk[k]):<3}" for k in ("1", "2", "3", "4-6", "7+"))
    print(f"{label:4} {len(segs):5}  {100 * sum(s['reads'] for s in segs) / max(1, rep):9.2f}  {100 * sum(s['reads'] > 0 for s in segs) / max(1, len(segs)):11.0f}"
          f"  {100 * sum(s['read_first3'] for s in later) / max(1, len(later)):14.0f} | {cells}")

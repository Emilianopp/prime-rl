"""Per-segment judge reward and per-game achievements for every block of a qv chain, as
JSON for plot_rewards.py. Blocks come from CHAIN/chain.log ("[chain block JOB] start wN"),
their outputs from OUT_ROOT/JOB (qv.jsonl, qv_states/, traces).

usage: extract_rewards.py CHAIN_DIR OUT_ROOT OUT_JSON
writes {"segments": [[update, R, points_gained], ...], "achievements": {start_update: [per game, ...]}}"""
import glob
import json
import os
import re
import sys


def main():
    chain, root, out = sys.argv[1:4]
    log = open(f"{chain}/chain.log").read()
    starts = {m.group(1): int(m.group(2)) for m in re.finditer(r"chain block (\d+)\] start w(\d+)", log)}
    segs, ach = [], {}
    for job, w in starts.items():
        qv = f"{root}/{job}/qv.jsonl"
        if not os.path.exists(qv):
            continue
        world = {}  # trace id -> world index, from each trace's opening record
        for t in glob.glob(f"{root}/{job}/run/*/monitors/file/traces/live/*.jsonl"):
            with open(t) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if "open" in d:
                            world[os.path.basename(t)[:-6]] = d["open"]["task"]["data"]["idx"]
                            break
        last = {}
        for line in open(qv):
            d = json.loads(line)
            i = d.get("info", {})
            if "B" not in i:
                continue
            segs.append([w + d["policy"], d["reward"], i["B"][1] - i["B"][0]])
            g = world.get(d["trace"])
            if g is not None and (g not in last or d["time"] > last[g]["time"]):
                last[g] = d
        ach[w] = []
        for d in last.values():
            p = f"{root}/{job}/qv_states/{d['trace']}.end.json"
            if os.path.exists(p):
                ach[w].append(len(json.load(open(p))["state"]["achievements"]))
    json.dump({"segments": segs, "achievements": ach}, open(out, "w"))


if __name__ == "__main__":
    main()

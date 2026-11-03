"""Drop tasks with no learning signal from a staged tinyswe directory, by measured solve rate.

A task whose every attempt got the same reward (always solved or never solved) gives a GRPO
group zero advantage: the orchestrator generates the whole group and then discards it. With
episode logs from earlier runs (the env's episode_log, one JSON line per episode), this writes
an index next to index.jsonl that keeps the mixed tasks and the ones not yet tried enough to
judge; point env.taskset.index at it.

  <dataset>/<out>            the kept tasks (index.jsonl lines)
  <dataset>/<out>.drop.json  the dropped ids, with their attempts and solve rate

Usage: python filter_tasks.py <dataset_dir> <episodes.jsonl>... [--min-tries 6] [--out index.filtered.jsonl]
"""

import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("episodes", nargs="+")
ap.add_argument("--min-tries", type=int, default=6, help="attempts before a uniform task counts as judged")
ap.add_argument("--out", default="index.filtered.jsonl")
a = ap.parse_args()

rewards = collections.defaultdict(list)
for path in a.episodes:
    for line in open(path):
        e = json.loads(line)
        rewards[e["id"]].append(e["reward"])

kept, dropped = [], {}
for line in open(f"{a.dataset}/index.jsonl"):
    row = json.loads(line)
    r = rewards.get(row["id"], [])
    if len(r) >= a.min_tries and min(r) == max(r):
        dropped[row["id"]] = {"tries": len(r), "solve_rate": r[0]}
    else:
        kept.append(line)
open(f"{a.dataset}/{a.out}", "w").writelines(kept)
json.dump(dropped, open(f"{a.dataset}/{a.out}.drop.json", "w"), indent=1)
solved = sum(d["solve_rate"] == 1 for d in dropped.values())
judged = sum(len(r) >= a.min_tries for r in rewards.values())
print(
    f"kept {len(kept)}, dropped {len(dropped)} ({solved} always solved, {len(dropped) - solved} never solved); "
    f"{judged} tasks had >= {a.min_tries} attempts"
)

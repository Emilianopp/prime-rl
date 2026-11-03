"""Stage SWE-smith tasks (HF SWE-bench/SWE-smith) into the directory tinyswe-env reads.

  <out>/index.jsonl          id, repo, image, problem_statement, suite_s
  <out>/tasks/<id>.json      FAIL_TO_PASS, PASS_TO_PASS, test_files (pytest node ids; the grade runs test_files)
  <out>/patches/<id>.diff    the bug, applied at sandbox start in its image's /testbed

Only tasks with a problem statement, on the images listed (one image name per line, e.g. the ones
pulled or saved on the cluster), at most --per-repo per image, shuffled.

Usage: python stage_smith.py <images.txt> <out_dir> [--per-repo 200] [--suite-s 30] [--seed 0]
       [--skip-repos sqlglot,trio,...]
"""

import argparse
import collections
import json
import os
import random

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("out")
ap.add_argument("--per-repo", type=int, default=200)
ap.add_argument("--suite-s", type=float, default=30, help="grade time estimate; the grade timeout is max(120, 5x)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-repos", default="", help="comma-separated substrings of image names to leave out")
a = ap.parse_args()
skip = [s for s in a.skip_repos.split(",") if s]
images = {l.split()[0].removesuffix(":latest") for l in open(a.images) if l.strip()}
images = {i for i in images if not any(s in i for s in skip)}
cols = ["instance_id", "image_name", "repo", "patch", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS"]
ds = load_dataset("SWE-bench/SWE-smith", split="train").select_columns(cols)
by = collections.defaultdict(list)
for r in ds:
    if r["image_name"] in images and (r["problem_statement"] or "").strip():
        by[r["image_name"]].append(r)
rng = random.Random(a.seed)
rows = []
for img, rs in sorted(by.items()):
    rng.shuffle(rs)
    rows += rs[: a.per_repo]
rng.shuffle(rows)
os.makedirs(f"{a.out}/tasks", exist_ok=True)
os.makedirs(f"{a.out}/patches", exist_ok=True)
with open(f"{a.out}/index.jsonl", "w") as idx:
    for r in rows:
        f2p, p2p = list(r["FAIL_TO_PASS"]), list(r["PASS_TO_PASS"])
        files = sorted({t.split("::")[0] for t in f2p + p2p})
        open(f"{a.out}/patches/{r['instance_id']}.diff", "w").write(r["patch"])
        json.dump(
            {"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p, "test_files": files},
            open(f"{a.out}/tasks/{r['instance_id']}.json", "w"),
        )
        repo = r["repo"].split("/")[-1]
        idx.write(
            json.dumps(
                {
                    "id": r["instance_id"],
                    "repo": repo,
                    "image": f"docker.io/{r['image_name']}:latest",
                    "suite_s": a.suite_s,
                    "problem_statement": r["problem_statement"],
                }
            )
            + "\n"
        )
print(
    f"staged {len(rows)} tasks from {len(by)} images: "
    + ", ".join(f"{i.split('.x86_64.')[-1]} {min(len(v), a.per_repo)}" for i, v in sorted(by.items()))
)

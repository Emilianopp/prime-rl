"""Stage tiny-swe tasks.jsonl into the directory tinyswe-env reads.

  <out>/index.jsonl          id, repo, problem_statement, suite_s (one task per line, grouped by repo)
  <out>/tasks/<id>.json      FAIL_TO_PASS, PASS_TO_PASS (read inside the sandbox by the grader)
  <out>/patches/<id>.diff    the bug, applied and committed at sandbox start

Usage: python stage_tasks.py <tasks.jsonl> <baseline_dir> <out_dir> [--suite-max 60]
(baseline_dir holds <repo>.wall.json: the repo's full-suite time in the store image.)
"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("tasks")
ap.add_argument("baseline")
ap.add_argument("out")
ap.add_argument("--suite-max", type=float, default=60, help="drop repos whose full suite takes longer (s)")
a = ap.parse_args()
os.makedirs(f"{a.out}/tasks", exist_ok=True)
os.makedirs(f"{a.out}/patches", exist_ok=True)
kept, skipped = 0, set()
with open(f"{a.out}/index.jsonl", "w") as idx:
    for line in open(a.tasks):
        t = json.loads(line)
        suite = json.load(open(f"{a.baseline}/{t['repo']}.wall.json"))
        if suite > a.suite_max:
            skipped.add(t["repo"])
            continue
        open(f"{a.out}/patches/{t['instance_id']}.diff", "w").write(t["patch"])
        json.dump(
            {k: t[k] for k in ("FAIL_TO_PASS", "PASS_TO_PASS")}, open(f"{a.out}/tasks/{t['instance_id']}.json", "w")
        )
        idx.write(
            json.dumps(
                {
                    "id": t["instance_id"],
                    "repo": t["repo"],
                    "suite_s": suite,
                    "problem_statement": t["problem_statement"],
                }
            )
            + "\n"
        )
        kept += 1
print(f"staged {kept} tasks; skipped repos with suite > {a.suite_max:.0f} s: {sorted(skipped)}")

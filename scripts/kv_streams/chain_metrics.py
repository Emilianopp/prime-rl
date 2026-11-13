"""Per-global-step trainer metrics for a chained RL run (scripts/kv_streams/chain.sbatch).

Each block restarts its step count at 1 from the chain's latest weights; chain.log records
the start (`[chain block N/M, job J] start wS`). Global step = S + local step. Reads every
block's trainer.log under OUT_ROOT/<config name>/<job>/run/*/logs/latest/ and prints JSON
rows {step, job, mismatch_kl, kl_evicted, kl_full, entropy, step_time_s}.

    python chain_metrics.py <chain dir> <block output root>
"""

import glob
import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
START = re.compile(r"\[chain block \d+/\d+, job (\d+)\] start w(\d+)")
STEP = re.compile(
    r"Step (\d+) \|\s*(?:(\d+)m )?\s*(\d+)s \|.*?Entropy ([0-9.]+) \| Mismatch KL ([0-9.]+)"
    r"(?: \(evicted context ([0-9.]+), full context ([0-9.]+))?"
)


def main(chain_dir: str, out_root: str) -> None:
    starts = {}
    for line in open(f"{chain_dir}/chain.log"):
        if m := START.search(line):
            starts[m.group(1)] = int(m.group(2))
    rows = {}
    for job, s0 in starts.items():
        for path in glob.glob(f"{out_root}/{job}/run/*/logs/latest/trainer.log"):
            for line in open(path, errors="replace"):
                m = STEP.search(ANSI.sub("", line))
                if not m or "SUCCESS" not in line:
                    continue
                local, mins, secs, ent, kl, kl_ev, kl_full = m.groups()
                step = s0 + int(local)
                rows[step] = {
                    "step": step,
                    "job": job,
                    "mismatch_kl": float(kl),
                    "kl_evicted": float(kl_ev) if kl_ev else None,
                    "kl_full": float(kl_full) if kl_full else None,
                    "entropy": float(ent),
                    "step_time_s": int(mins or 0) * 60 + int(secs),
                }
    print(json.dumps([rows[k] for k in sorted(rows)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

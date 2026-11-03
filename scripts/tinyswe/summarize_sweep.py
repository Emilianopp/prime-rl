"""Per-level summary of a kve_sweep.sbatch output dir: one row per concurrency level, over its
steady window (from WARMUP s after the level starts to 60 s before it ends).

  engine:  decode tokens/s (all engines), running and queued requests, KV and DeltaNet-slot use
  node:    CPU in use (fraction of the job's CPUs, and cores above the idle-engine baseline),
           memory, live sandboxes
  env:     episodes finished per minute, tool calls per second, sandbox boot/grade times,
           seconds per command, reward

Usage: python3 summarize_sweep.py <out_dir> [warmup_s=300]
"""

import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime

OUT = sys.argv[1]
WARMUP = float(sys.argv[2]) if len(sys.argv) > 2 else 300
DECODE = re.compile(
    r"^\[(\S+ \S+) DP(\d+)[^\]]*\] Decode batch, #running-req: (\d+), #full token: (\d+), "
    r"full token usage: ([\d.]+), mamba num: \d+, mamba usage: ([\d.]+).*gen throughput \(token/s\): "
    r"([\d.]+), #queue-req: (\d+)"
)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


levels = [l.split("\t") for l in open(f"{OUT}/levels.tsv").read().splitlines()[1:]]
levels = [(int(c), float(a), float(b)) for c, a, b in levels]
node = [l.split("\t") for l in open(f"{OUT}/node.tsv").read().splitlines()[1:]]
node = [(float(t), float(c), float(m), int(s)) for t, c, m, s, _ in node]
ncpu = int(re.search(r"(\d+) cpus", open(f"{OUT}/cell.txt").read()).group(1))
base = [c for t, c, m, s in node if levels and levels[0][1] - 60 <= t < levels[0][1]]
base = st.median(base) if base else 0.0

# engine log: per 10 s bucket, sum over DP ranks of each rank's last line in the bucket
buckets = defaultdict(dict)
for line in open(f"{OUT}/inference.log", errors="replace"):
    m = DECODE.match(line)
    if m:
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        buckets[int(t // 10) * 10][m.group(2)] = tuple(float(x) for x in m.group(3, 4, 5, 6, 7, 8))

print(f"node {ncpu} cpus; idle-engine CPU baseline {base:.2f} of the job ({base * ncpu:.1f} cores)\n")
hdr = (
    "conc",
    "tok/s",
    "run",
    "queue",
    "kv_use",
    "slots",
    "cpu",
    "sbx_cores",
    "mem_gb",
    "sandboxes",
    "ep/min",
    "calls/s",
    "s/call",
    "boot_p50",
    "boot_p95",
    "grade_s",
    "ep_min",
    "steps",
    "reward",
    "n_ep",
)
print("  ".join(f"{h:>9}" for h in hdr))
for conc, a, b in levels:
    w0, w1 = a + WARMUP, b - 60
    win = max(w1 - w0, 1)
    eng = [v for t, v in buckets.items() if w0 <= t < w1 and v]
    tok = [sum(r[4] for r in v.values()) for v in eng]
    run = [sum(r[0] for r in v.values()) for v in eng]
    que = [sum(r[5] for r in v.values()) for v in eng]
    kv = [st.mean(r[2] for r in v.values()) for v in eng]
    slots = [st.mean(r[3] for r in v.values()) for v in eng]
    nd = [(c, m, s) for t, c, m, s in node if w0 <= t < w1]
    cpu = st.median(c for c, _, _ in nd) if nd else float("nan")
    ep_file = f"{OUT}/c{conc}/episodes.jsonl"
    eps = [json.loads(l) for l in open(ep_file)] if os.path.exists(ep_file) else []
    done = [e for e in eps if w0 <= e["end"] < w1]
    calls = 0.0  # each episode's commands spread evenly over its life, the part inside the window
    for e in eps:
        dur = max(e["end"] - e["start"], 1)
        calls += e["steps"] * max(0.0, min(e["end"], w1) - max(e["start"], w0)) / dur
    boots = [e["boot_s"] for e in eps if e.get("boot_s") is not None and w0 <= e["start"] < w1]
    row = (
        conc,
        q(tok, 0.5),
        q(run, 0.5),
        q(que, 0.5),
        q(kv, 0.5),
        q(slots, 0.5),
        cpu,
        (cpu - base) * ncpu,
        max((m for _, m, _ in nd), default=float("nan")),
        q([s for _, _, s in nd], 0.5),
        len(done) / win * 60,
        calls / win,
        sum(e["exec_s"] for e in done) / max(1, sum(e["steps"] for e in done)),
        q(boots, 0.5),
        q(boots, 0.95),
        st.mean(e["grade_s"] or 0 for e in done) if done else float("nan"),
        st.mean((e["end"] - e["start"]) / 60 for e in done) if done else float("nan"),
        st.mean(e["steps"] for e in done) if done else float("nan"),
        st.mean(e["reward"] for e in done) if done else float("nan"),
        len(done),
    )
    print("  ".join(f"{x:>9.2f}" if isinstance(x, float) else f"{x:>9}" for x in row))
print(
    "\ntok/s, run, queue: sums over engines; kv_use, slots: mean fraction per engine; cpu: fraction of the job;"
    "\nsbx_cores: cores above the idle-engine baseline (sandboxes + env workers + the engines' extra);"
    "\ncalls/s, s/call, boot, grade: the env's own timings; ep_min: episode length in minutes."
)

"""Engine + node timeline of one kve_sweep run dir: every 3 minutes, episodes running/queued
(summed over engines), the fullest engine's KV use, decode tok/s, node CPU and live sandboxes."""
import collections, re, sys, time
d = sys.argv[1]
R = re.compile(r"^\[\S+ (\d\d:\d\d):\d\d DP(\d+)[^\]]*\] Decode batch, #running-req: (\d+), #full token: \d+, "
               r"full token usage: ([\d.]+).*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
b = collections.defaultdict(dict)
for line in open(d + "/inference.log", errors="replace"):
    m = R.match(line)
    if m:
        b[m.group(1)][m.group(2)] = (int(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6)))
node = {}
for line in open(d + "/node.tsv").read().splitlines()[1:]:
    t, cpu, mem, sbx, _ = line.split("\t")
    node[time.strftime("%H:%M", time.localtime(int(t)))] = (float(cpu), int(sbx), float(mem))
keys = sorted(b)
for k in keys[::3] + keys[-1:]:
    v = b[k].values()
    c, s, m = node.get(k, (float("nan"), -1, float("nan")))
    print(f"{k}  running {sum(x[0] for x in v):4d}  queued {sum(x[3] for x in v):4d}  kv_max {max(x[1] for x in v):.2f}  "
          f"tok/s {sum(x[2] for x in v):6.0f}  cpu {c:.2f}  sandboxes {s:4d}  mem_gb {m:.0f}")

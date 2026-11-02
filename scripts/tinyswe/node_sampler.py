"""Every INTERVAL s, one TSV row for this Slurm job: CPU use as a fraction of its CPUs (job cgroup),
memory, live tinyswe sandboxes (socket dirs) and load average. Runs until killed.

Usage: python3 node_sampler.py <out.tsv> [interval_s=10]      (env: TINYSWE_SOCKETS)
"""

import os
import sys
import time

out, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10
rel = open("/proc/self/cgroup").read().strip().split("::")[-1].strip("/").split("/")
cg = "/sys/fs/cgroup/" + "/".join(rel[: next((i + 1 for i, p in enumerate(rel) if p.startswith("job_")), len(rel))])
ncpu = len(os.sched_getaffinity(0))
socks = os.environ.get("TINYSWE_SOCKETS", "")


def usage():
    return int(next(l.split()[1] for l in open(f"{cg}/cpu.stat") if l.startswith("usage_usec")))


with open(out, "w") as f:
    f.write("t\tcpu_frac\tmem_gb\tsandboxes\tload1\n")
    u0, t0 = usage(), time.time()
    while True:
        time.sleep(interval)
        u, t = usage(), time.time()
        mem = int(open(f"{cg}/memory.current").read()) / 2**30
        n = len(os.listdir(socks)) if socks and os.path.isdir(socks) else 0
        f.write(f"{t:.0f}\t{(u - u0) / 1e6 / (t - t0) / ncpu:.3f}\t{mem:.1f}\t{n}\t{os.getloadavg()[0]:.1f}\n")
        f.flush()
        u0, t0 = u, t

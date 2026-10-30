"""Per-training-step reply breakdown of a training run dir (buckets of N steps; uses reply_breakdown.category).
usage: step_breakdown.py <run dir> [bucket=5]"""
import collections, glob, io, json, os, subprocess, sys
sys.path.insert(0, "/network/scratch/y/yuns/factory/agent-rl/prime-rl/scripts/crafter")
from reply_breakdown import category, text, CATS, A
run, bucket = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
step_of = {}
for f in glob.glob(f"{run}/**/annotations/orch/*.jsonl", recursive=True):
    for line in open(f):
        r = json.loads(line); step_of[r["trace_id"]] = r["info"]["ship"]["step"]
per = collections.defaultdict(lambda: {"traces": 0, "replies": 0, "cats": collections.Counter(), "chars": collections.Counter(), "turns": 0, "malf": 0, "notes_mid": 0, "tok_out": 0})
for f in sorted(glob.glob(f"{run}/**/stream/*.jsonl.zst", recursive=True)):
    blob = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
    for line in io.BytesIO(blob):
        d = json.loads(line)
        for tr in d["traces"]:
            c = (tr.get("info") or {}).get("crafter")
            if not c or tr["id"] not in step_of: continue
            b = (step_of[tr["id"]] - 1) // bucket
            p = per[b]; p["traces"] += 1; p["turns"] += c["steps"]
            p["malf"] += (c.get("metrics") or {}).get("malformed_replies", 0) if isinstance(c.get("metrics"), dict) else 0
            nodes = [n for n in tr["nodes"] if n.get("message") and n["message"]["role"] == "assistant" and n.get("sampled", True)]
            for i, n in enumerate(nodes):
                t = text(n["message"]); k = category(t, i == len(nodes) - 1)
                p["replies"] += 1; p["cats"][k] += 1; p["chars"][k] += len(t)
                p["tok_out"] += len(n.get("token_ids") or n.get("tokens") or [])
print(f"steps      | traces | turns/tr | replies/tr | " + " ".join(f"{k:>12}" for k in CATS) + " | chars: action +text +notes prose")
for b in sorted(per):
    p = per[b]; n = p["replies"]
    print(f"{b*bucket+1:3d}-{(b+1)*bucket:3d}    | {p['traces']:6d} | {p['turns']/p['traces']:8.1f} | {n/p['traces']:10.1f} | " + " ".join(f"{p['cats'][k]/n:12.3f}" for k in CATS)
          + " | " + " ".join(f"{p['chars'][k]/max(1,p['cats'][k]):6.0f}" for k in ("action","action+text","action+notes","prose")))

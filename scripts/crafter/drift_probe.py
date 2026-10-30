"""On-node probe of a running run: per third of the segment start order, the repeat rate,
tag-less reply share and their texts, and the `do` share. usage: drift_probe.py <stream dir> <episodes.jsonl>"""
import glob, io, json, re, subprocess, sys, collections
stream, episodes = sys.argv[1], sys.argv[2]
eps = sorted((json.loads(l) for l in open(episodes)), key=lambda e: e.get("start", 0))
eps = [e for e in eps if e.get("task") == "crafter"]; n = len(eps)
rank = {e["node_id"]: i * 3 // max(1, n) for i, e in enumerate(eps)}
def text(m):
    c = m.get("content"); return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in (c or []) if isinstance(x, dict))
A = re.compile(r"<action>\s*(\w+)\s*</action>")
third = collections.defaultdict(lambda: {"traces": 0, "replies": 0, "tagless": 0, "rep": 0, "pairs": 0, "do": 0, "texts": collections.Counter()})
files = sorted(glob.glob(f"{stream}/*.jsonl.zst")); print("stream files:", [f.rsplit('/', 1)[-1] for f in files])
for f in files:
    blob = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
    for line in io.BytesIO(blob):
        for tr in json.loads(line).get("traces", []):
            t = third[rank.get(tr["id"], -1)]; t["traces"] += 1
            acts = []
            for nd in tr.get("nodes", []):
                m = nd.get("message")
                if not m or m.get("role") != "assistant": continue
                r = text(m); a = A.search(r); t["replies"] += 1
                if a: acts.append(a.group(1)); t["do"] += a.group(1) == "do"
                else:
                    t["tagless"] += 1; t["texts"][re.sub(r"\s+", " ", r)[:50]] += 1
            acts = acts[:-1] if acts else acts
            for x, y in zip(acts, acts[1:]): t["pairs"] += 1; t["rep"] += x == y
for k in sorted(third):
    t = third[k]
    print(f"third {k + 1}: {t['traces']} traces, tag-less {t['tagless'] / max(1, t['replies']):.1%} of {t['replies']} replies, P(repeat) {t['rep'] / max(1, t['pairs']):.2f}, do share {t['do'] / max(1, t['replies']):.2f}")
    print("   top tag-less texts:", [(c, s) for s, c in t["texts"].most_common(4)])

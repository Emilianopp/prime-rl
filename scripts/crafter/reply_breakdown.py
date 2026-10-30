"""Per-depth breakdown of what the policy replies with: pure <action> tags, tags with extra
text, tags next to spontaneous <notes>, bare "<" (empty), tag-less prose, lessons, the notes
turn. Reads every trace stream under the given dirs (a training run dir, or a beam chain's
traces/ dir), so a no-trainer beam and a training run compare on the same footing.

usage: reply_breakdown.py <dir> [<dir> ...]
"""
import collections
import glob
import io
import json
import os
import re
import subprocess
import sys

A = re.compile(r"<action>\s*(\w+)\s*</action>")
CATS = ["action", "action+text", "action+notes", "empty", "prose", "lesson", "notes"]


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(x.get("text", "") or "" for x in (c or []) if isinstance(x, dict))


def category(content, last):
    if last and not A.search(content):
        return "notes"
    if "<lesson>" in content:
        return "lesson"
    if A.search(content) is None:
        return "empty" if len(content.strip()) < 20 else "prose"
    if "<notes>" in content:
        return "action+notes"
    return "action" if len(content.strip()) <= 40 else "action+text"


def main(dirs):
    files, depth_of = [], {}
    for d in dirs:
        files += sorted(glob.glob(f"{d}/**/stream/*.jsonl.zst", recursive=True))
        for ep in [p for p in (os.path.join(d, "episodes.jsonl"), os.path.join(os.path.dirname(d.rstrip("/")), "episodes.jsonl")) if os.path.exists(p)]:  # a beam chain: the stream's depth is per level, the log's is the tree's
            for line in open(ep):
                r = json.loads(line)
                if r.get("task") == "crafter" and r.get("node_id"):
                    depth_of[r["node_id"]] = r["depth"]
    print(f"{len(files)} stream files")
    per = collections.defaultdict(lambda: {"traces": 0, "replies": 0, "cats": collections.Counter(), "chars": collections.Counter(),
                                           "prose_chars": [], "turns": 0, "steps": 0})
    for f in files:
        blob = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
        for line in io.BytesIO(blob):
            d = json.loads(line)
            for tr in d["traces"]:
                c = (tr.get("info") or {}).get("crafter")
                if not c:
                    continue
                p = per[depth_of.get(tr["id"], c["depth"])]
                p["traces"] += 1
                p["steps"] += c["steps"]
                nodes = [n for n in tr["nodes"] if n.get("message") and n["message"]["role"] == "assistant" and n.get("sampled", True)]
                for i, n in enumerate(nodes):
                    t = text(n["message"])
                    k = category(t, i == len(nodes) - 1)
                    p["replies"] += 1
                    p["cats"][k] += 1
                    p["chars"][k] += len(t)
                    if k == "prose":
                        p["prose_chars"].append(len(t))
    print("depth | traces | replies/trace | share: action, +text, +notes, empty, prose, lesson | chars/reply: +text, +notes, prose | prose p90 chars")
    for dpt in sorted(per):
        p = per[dpt]
        r = p["replies"] or 1
        sh = {k: p["cats"][k] / r for k in CATS}
        ch = {k: p["chars"][k] / max(1, p["cats"][k]) for k in CATS}
        pc = sorted(p["prose_chars"])
        p90 = pc[int(0.9 * (len(pc) - 1))] if pc else 0
        print(f"{dpt:5d} | {p['traces']:6d} | {r / p['traces']:13.0f} | {sh['action']:.3f} {sh['action+text']:.3f} {sh['action+notes']:.3f} {sh['empty']:.3f} {sh['prose']:.3f} {sh['lesson']:.3f} | "
              f"{ch['action+text']:5.0f} {ch['action+notes']:5.0f} {ch['prose']:5.0f} | {p90:5d}")
    tot = sum(p["replies"] for p in per.values()) or 1
    allc = collections.Counter()
    for p in per.values():
        allc.update(p["cats"])
    print("all depths:", {k: f"{allc[k] / tot:.3f}" for k in CATS})


if __name__ == "__main__":
    main(sys.argv[1:])

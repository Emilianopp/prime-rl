"""Verifiers traces -> compact JSON for the Trace Inspector (viewer.html), for any env.

Reads an eval or RL output dir (anything holding */monitors/file/traces/stream/*.jsonl[.zst]): a
summary row for every episode, and the full dialogue of a spread of them (token ids dropped; long
messages clipped to their head and tail). Per message: role, token span, the call that evicted it
(KV-Streams `eviction` spans on the nodes), text, reasoning and tool calls. Per model call: its node, wall
times, prompt/completion tokens and finish reason, so the viewer can split each turn into model
time and env time.

usage: extract_traces.py <run dir> <out.json> [--pick K] [--ids ID ...] [--clip CHARS]
  --pick K   K episodes spread over (reward, turns) (default 40)
  --ids      these episode or trace ids instead
"""

import argparse
import glob
import json
import os
import statistics as st
import subprocess
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("out")
ap.add_argument("--pick", type=int, default=40)
ap.add_argument("--ids", nargs="*")
ap.add_argument("--clip", type=int, default=4000)
args = ap.parse_args()

streams = sorted(glob.glob(os.path.join(args.run_dir, "**/monitors/file/traces/stream/*"), recursive=True))
streams = [p for p in streams if p.endswith((".jsonl", ".zst"))]
if not streams:
    raise SystemExit(f"no trace streams under {args.run_dir}")


def records():
    for path in streams:
        if path.endswith(".zst"):
            proc = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, text=True)
            yield from (json.loads(l) for l in proc.stdout)
            proc.wait()
        else:
            yield from (json.loads(l) for l in open(path) if l.strip())


def scalars(d, prefix=""):
    """Flat scalar fields of an env's trace.info (nested one level), for the episode table."""
    out = {}
    for k, v in (d or {}).items():
        if k in ("kind", "dispatch", "arrival"):
            continue
        if isinstance(v, dict):
            out |= {f"{k}.{kk}": vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool)) or vv is None}
        elif isinstance(v, (int, float, str, bool)):
            out[prefix + k] = v
    return out


def clip(s):
    s = s if isinstance(s, str) else ("" if s is None else json.dumps(s))
    if len(s) <= args.clip:
        return s
    h = args.clip * 2 // 3
    return s[:h] + f"\n[... {len(s) - args.clip:,} characters clipped ...]\n" + s[-(args.clip - h) :]


def ep_row(i, ep):
    trs = ep["traces"]
    t0 = trs[0]
    calls = [c for tr in trs for c in tr.get("calls") or []]
    rewards = [sum(r["score"] * r.get("weight", 1) for r in (tr.get("rewards") or {}).values()) for tr in trs]
    times = [c["time"] for c in calls if c.get("time")]
    return {
        "i": i,
        "id": ep["id"],
        "task": (t0["task"]["data"].get("name") or t0["task"]["data"].get("idx")),
        "env": (ep.get("env") or {}).get("name"),
        "reward": round(sum(rewards), 4),
        "turns": len(calls),
        "traces": len(trs),
        "in_tok": sum(tr.get("num_input_tokens") or 0 for tr in trs),
        "out_tok": sum(tr.get("num_output_tokens") or 0 for tr in trs),
        "dur": round(max(t["end"] for t in times) - min(t["start"] for t in times), 1) if times else None,
        "stop": t0.get("stop_condition"),
        "ok": ep.get("ok"),
        "info": scalars(t0.get("info")),
    }


def trace_full(ep_i, tr):
    nodes = tr["nodes"]
    death = []
    for n in nodes:
        death.extend([-1] * len(n.get("token_ids") or []))
    evs = []
    for n in nodes:
        ev = n.get("eviction")
        if ev:
            for s, e in ev["spans"]:
                evs.append([ev["at"], s, e])
                for j in range(s, min(e, len(death))):
                    if death[j] < 0:
                        death[j] = ev["at"]
    msgs, pos = [], 0
    for n in nodes:
        m = n["message"]
        k = len(n.get("token_ids") or [])
        d = [x for x in death[pos : pos + k] if x >= 0]
        msgs.append(
            [
                m.get("role", "?"),
                pos,
                k,
                min(d) if d else -1,
                round(len(d) / k, 3) if k else 0,
                clip(m.get("content")),
                clip(m.get("reasoning_content") or ""),
                n.get("timestamp"),
                n.get("parent"),
                [[c.get("name"), clip(c.get("arguments"))] for c in m.get("tool_calls") or []],
            ]
        )
        pos += k
    calls = [
        [
            c.get("node"),
            (c.get("time") or {}).get("start"),
            (c.get("time") or {}).get("end"),
            (c.get("usage") or {}).get("prompt_tokens"),
            (c.get("usage") or {}).get("completion_tokens"),
            c.get("finish_reason"),
        ]
        for c in tr.get("calls") or []
    ]
    data = tr["task"]["data"]
    return {
        "ep": ep_i,
        "id": tr["id"],
        "task": data.get("name"),
        "ntok": pos,
        "protected": len(nodes[0].get("token_ids") or [])
        if nodes and nodes[0]["message"].get("role") == "system"
        else 0,
        "msgs": msgs,
        "calls": calls,
        "ev": evs,
        "rewards": tr.get("rewards"),
        "info": tr.get("info"),
        "timing": tr.get("timing"),
        "stop": tr.get("stop_condition"),
        "errors": tr.get("errors"),
        "agent": (tr.get("agent") or {}).get("name"),
    }


# pass 1: every episode's summary row
rows, meta = [], None
for i, ep in enumerate(records()):
    rows.append(ep_row(i, ep))
    if meta is None:
        t0 = ep["traces"][0]
        cfg = (t0.get("agent") or {}).get("config") or {}
        meta = {
            "model": cfg.get("model"),
            "sampling": cfg.get("sampling"),
            "max_total_tokens": cfg.get("max_total_tokens"),
            "harness": (cfg.get("harness") or {}).get("id"),
            "run": (ep.get("run") or {}).get("name"),
            "kind": ((ep.get("run") or {}).get("work") or {}).get("type"),
            "env": (ep.get("env") or {}).get("name"),
            "verifiers": (t0.get("verifiers") or {}).get("version"),
        }

if args.ids:
    want = {r["i"] for r in rows if r["id"] in args.ids}
else:  # a spread over reward then turns: evenly spaced through the sorted list
    order = sorted(rows, key=lambda r: (r["reward"], r["turns"]))
    k = min(args.pick, len(order))
    want = {order[round(j * (len(order) - 1) / max(1, k - 1))]["i"] for j in range(k)}
    want |= {order[0]["i"], order[-1]["i"]}

# pass 2: full traces of the picked episodes
traces = []
for i, ep in enumerate(records()):
    if i in want:
        for tr in ep["traces"]:
            traces.append(trace_full(i, tr))
            for r in rows:
                if r["i"] == i:
                    r.setdefault("t", []).append(len(traces) - 1)

rew = [r["reward"] for r in rows]
data = {
    "run": meta
    | {
        "path": os.path.abspath(args.run_dir),
        "episodes": len(rows),
        "reward_mean": round(st.mean(rew), 4) if rew else None,
        "turns_mean": round(st.mean(r["turns"] for r in rows), 1) if rows else None,
        "stops": dict(Counter(r["stop"] for r in rows)),
        "shown": len(want),
        "clip": args.clip,
    },
    "episodes": rows,
    "traces": traces,
}
json.dump(data, open(args.out, "w"), separators=(",", ":"))
print(
    f"{len(rows)} episodes, {len(want)} with full traces ({len(traces)} traces), {os.path.getsize(args.out) / 1e6:.1f} MB"
)

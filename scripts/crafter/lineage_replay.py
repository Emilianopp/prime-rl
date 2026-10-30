"""Lineage replay data for a Crafter beam chain: every leaf of the last depth, backtracked to
its root through `parent`, as one continuous game.

    python scripts/crafter/lineage_replay.py <chain dir> [<chain dir> ...] [--out DIR]

A chain dir is $SCRATCH/factory/runs/crafter-beam/<chain> (episodes.jsonl, states/beam/*.json,
and traces/depth-*/ when the chain ran with KEEP_TRACES=1). Writes
<out>/data/<chain>/<seed>.json per seed and <out>/data/chains.json; serve <out> with the
viewer (scripts/crafter/lineage_replay.html, copied there) and open index.html.

With traces every segment on a leaf's lineage carries its turns (each observation and the
reply to it); without, the viewer steps segment by segment (end map, deaths, events, notes).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

REPLY_CAP = 6000  # characters kept of a reply (notes turns can be long)


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def load_traces(chain: str) -> dict[str, dict]:
    """node_id -> trace (the stream of every saved depth), when the chain kept traces."""
    import zstandard

    traces: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(chain, "traces", "depth-*", "stream", "*.jsonl*"))):
        if path.endswith(".zst"):
            with open(path, "rb") as f:
                text = zstandard.ZstdDecompressor().stream_reader(f).read().decode()
        else:
            text = open(path).read()
        for line in text.splitlines():
            try:
                episode = json.loads(line)
            except ValueError:
                continue
            for trace in episode.get("traces") or []:
                node_id = ((trace.get("info") or {}).get("crafter") or {}).get("node_id")
                if node_id:
                    traces[node_id] = trace
    return traces


def turns_of(trace: dict) -> list[dict]:
    """The segment's turns: each user message (an observation, a notes request, a probe) with
    the assistant reply that follows it. A system prompt that carries the opening observation
    (the env's include_initial_obs_in_system) opens the list."""
    turns: list[dict] = []
    pending: dict | None = None
    for node in trace.get("nodes") or []:
        message = node.get("message") or {}
        role, content = message.get("role"), message.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = content or ""
        if role == "system":
            if "Local map" in content:
                pending = {"obs": content[content.find("Step:"):] if "Step:" in content else content, "reply": ""}
        elif role == "user":
            if pending is not None:
                turns.append(pending)
            pending = {"obs": content, "reply": ""}
        elif role == "assistant" and pending is not None:
            reply = content if len(content) <= REPLY_CAP else content[:REPLY_CAP] + " [...]"
            pending["reply"] = (pending["reply"] + "\n" + reply).strip() if pending["reply"] else reply
            turns.append(pending)
            pending = None
    if pending is not None:
        turns.append(pending)
    return turns


SEGMENT_KEYS = (
    "depth", "branch", "parent", "reward", "achievements", "segment_achievements", "achievements_unlocked",
    "steps", "game_step", "ended_by", "continues", "deaths", "position_before", "position", "local_map",
    "events", "summary", "inventory", "next_tier", "world_nearby", "malformed_replies", "invalid_actions",
    "cells_visited", "unlocked_at",
)


def build(chain: str, out: str) -> dict:
    name = os.path.basename(chain.rstrip("/"))
    lines = read_jsonl(os.path.join(chain, "episodes.jsonl"))
    segments: dict[str, dict] = {}
    judge: dict[str, dict] = {}
    for line in lines:
        if line.get("task") == "crafter-judge" and line.get("node_id"):
            judge[line["node_id"]] = line
        elif line.get("node_id"):
            segments[line["node_id"]] = line  # the last record of a node wins (a resumed level)
    traces = load_traces(chain)
    beams = {}
    for path in glob.glob(os.path.join(chain, "states", "beam", "*.json")):
        try:
            beam = json.load(open(path))
            beams[int(beam["seed"])] = beam
        except (OSError, ValueError, KeyError):
            pass

    children: dict[str, int] = defaultdict(int)
    for seg in segments.values():
        children[seg.get("parent")] += 1
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for node_id, seg in segments.items():
        by_seed[int(seg["seed"])].append({"id": node_id, **seg})

    cell = open(os.path.join(chain, "cell.txt")).read().splitlines()[0] if os.path.exists(os.path.join(chain, "cell.txt")) else ""
    os.makedirs(os.path.join(out, "data", name), exist_ok=True)
    seeds_index = []
    for seed, nodes in sorted(by_seed.items()):
        max_depth = max(n["depth"] for n in nodes)
        leaves = [n for n in nodes if n["depth"] == max_depth]
        survivors = {n["id"] for n in (beams.get(seed) or {}).get("beam") or []}
        ids = {n["id"]: n for n in nodes}
        needed: set[str] = set()
        for leaf in leaves:
            node = leaf
            while node is not None:
                needed.add(node["id"])
                node = ids.get(node.get("parent"))
        out_nodes = {}
        for node_id in needed:
            seg = ids[node_id]
            j = judge.get(node_id) or {}
            row = {k: seg.get(k) for k in SEGMENT_KEYS if k in seg}
            row["children"] = children.get(node_id, 0)
            row["judge"] = {k: j.get(k) for k in ("judge_score", "progress", "plan", "consistency", "tier", "rationale", "judge_failed") if k in j}
            if node_id in traces:
                row["turns"] = turns_of(traces[node_id])
            out_nodes[node_id] = row
        leaves.sort(key=lambda n: -float((judge.get(n["id"]) or {}).get("judge_score") or n.get("reward") or 0))
        payload = {
            "chain": name, "cell": cell, "seed": seed, "max_depth": max_depth, "has_turns": any("turns" in n for n in out_nodes.values()),
            "leaves": [{"id": n["id"], "survivor": n["id"] in survivors} for n in leaves],
            "nodes": out_nodes,
        }
        with open(os.path.join(out, "data", name, f"{seed}.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        with_turns = sum(1 for n in out_nodes.values() if "turns" in n)
        seeds_index.append({"seed": seed, "max_depth": max_depth, "leaves": len(leaves), "has_turns": payload["has_turns"],
                            "segments": len(out_nodes), "segments_with_turns": with_turns})
        print(f"{name} seed {seed}: depth {max_depth}, {len(leaves)} leaves, {len(needed)} lineage segments, "
              f"{sum(1 for n in out_nodes.values() if 'turns' in n)} with turns", file=sys.stderr)
    return {"chain": name, "cell": cell, "seeds": seeds_index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chains", nargs="+")
    parser.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "factory", "lineage-replay"))
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    index_path = os.path.join(args.out, "data", "chains.json")
    index = {c["chain"]: c for c in (json.load(open(index_path)) if os.path.exists(index_path) else [])}
    for chain in args.chains:
        entry = build(chain, args.out)
        index[entry["chain"]] = entry
    with open(index_path, "w") as f:
        json.dump(sorted(index.values(), key=lambda c: c["chain"]), f, indent=1)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lineage_replay.html"), os.path.join(args.out, "index.html"))
    print(f"wrote {args.out}/index.html and data for {len(args.chains)} chain(s)", file=sys.stderr)


if __name__ == "__main__":
    main()

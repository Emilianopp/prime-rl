"""Lineage replay for Craftax beam runs: every leaf backtracked to the root through `parent`,
replayed frame by frame in a browser viewer.

    python scripts/craftax/lineage_replay.py <run dir> [<run dir> ...] [--name NAME] [--out DIR]

Run dirs of one chain (a job and the jobs that resumed it) are merged: episodes.jsonl from
each, turns/<node_id>.json.gz (env.turn_log, on by default since 2026-11-04) and
state/beam/*.json. Writes <out>/data/<name>/<seed>.json per world, the per-segment turn files
and frame sheets under <out>/data/<name>/<seed>/, <out>/data/runs.json, and the viewer as
<out>/index.html (scripts/craftax/lineage_replay.html). Serve <out> (python -m http.server)
and open index.html.

A segment whose whole path from the root has turn logs is replayed from the seed: the game is
deterministic given the played actions, so every turn gets its pixel frame (Craftax's own
renderer), what the model saw and wrote, and where it stood; each turn's position and each
segment's end-state hash are checked against the log. Older runs (no turn logs) fall back to
one card per segment: the floor map, game record, notes, events and judge scores at its end.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

SEGMENT_KEYS = (
    "depth", "branch", "parent", "reward", "return", "achievement_score", "achievements", "achievements_before",
    "achievement_reward_delta", "turns", "game_steps", "first_turn", "ended_by", "continues", "deaths", "floor",
    "position", "max_floor", "events", "stations", "notes", "record", "record_before", "map_ascii", "nearest",
    "next_step", "actions_hist", "forfeits", "forfeit_kinds", "explored_cells", "state_value", "map_check",
    "plan_check", "plan_feasible", "plan_on_path", "plan_on_track", "plan_target", "turns_lost_total", "trace_tokens",
    "survived", "unlocked_at",
)
JUDGE_KEYS = ("reward", "notes_score", "rationale", "plan", "findings", "grounding", "judge_failed", "skipped",
              "map_check", "plan_feasible", "plan_on_path")
ADHERENCE_KEYS = ("adherence_score", "deviation", "rationale", "skipped", "judge_failed")
SHEET_COLS, SHEET_FRAMES = 10, 100


def read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


class Renderer:
    """Craftax's pixel renderer, jitted once."""

    def __init__(self, px: int) -> None:
        import jax
        from craftax.craftax.renderer import render_craftax_pixels

        self.px = px
        self._fn = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def __call__(self, game: Any) -> np.ndarray:
        return np.clip(np.asarray(self._fn(game.state, self.px)), 0, 255).astype(np.uint8)


def write_sheets(frames: list[np.ndarray], stem: str) -> dict:
    """Frames into WebP sheets of SHEET_FRAMES each, SHEET_COLS per row."""
    from PIL import Image

    h, w = frames[0].shape[:2]
    names = []
    for j in range(0, len(frames), SHEET_FRAMES):
        chunk = frames[j:j + SHEET_FRAMES]
        rows = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
        sheet = np.zeros((rows * h, SHEET_COLS * w, 3), dtype=np.uint8)
        for k, frame in enumerate(chunk):
            r, c = divmod(k, SHEET_COLS)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        path = f"{stem}-{j // SHEET_FRAMES}.webp"
        Image.fromarray(sheet).save(path, lossless=True, quality=20, method=0)
        names.append(os.path.basename(path))
    return {"sheets": names, "n": len(frames), "cols": SHEET_COLS, "per_sheet": SHEET_FRAMES, "w": w, "h": h}


def turn_row(row: dict, names: list[str]) -> dict:
    out = {k: row.get(k) for k in ("i", "turn", "obs", "reasoning", "content", "forfeit", "pos", "health", "message",
                                   "new", "death", "finish") if row.get(k) is not None}
    out["acts"] = [names[a] for a in row.get("acts") or []]
    if row.get("frozen"):
        out["frozen"] = True
    return out


def build(dirs: list[str], name: str, out: str, px: int, all_leaves: bool, seeds: set[int] | None) -> dict:
    segments: dict[str, dict] = {}
    judge: dict[str, dict] = {}
    adherence: dict[str, dict] = {}
    logs: dict[str, str] = {}
    beams: dict[int, dict] = {}
    for d in dirs:
        for line in read_jsonl(os.path.join(d, "episodes.jsonl")):
            node_id, task = line.get("node_id"), line.get("task")
            if not node_id:
                continue
            if task == "craftax-judge":
                judge[node_id] = line
            elif task == "craftax-adherence":
                adherence[node_id] = line
            elif task == "craftax":
                segments[node_id] = line  # the last record of a node wins (a resumed level)
        for path in glob.glob(os.path.join(d, "turns", "*.json.gz")):
            logs[os.path.basename(path)[:-len(".json.gz")]] = path
        for path in glob.glob(os.path.join(d, "state", "beam", "*.json")):
            try:
                beam = json.load(open(path))
                if int(beam.get("depth", 0)) >= int((beams.get(int(beam["seed"])) or {}).get("depth", -1)):
                    beams[int(beam["seed"])] = beam
            except (OSError, ValueError, KeyError):
                pass
    cell = ""
    for d in dirs:
        if os.path.exists(os.path.join(d, "cell.txt")):
            cell = open(os.path.join(d, "cell.txt")).read().strip().splitlines()[0]
            break

    children: dict[str, list[str]] = defaultdict(list)
    for node_id, seg in segments.items():
        children[seg.get("parent")].append(node_id)
    by_seed: dict[int, list[str]] = defaultdict(list)
    for node_id, seg in segments.items():
        by_seed[int(seg["seed"])].append(node_id)

    renderer = None
    root_dir = os.path.join(out, "data", name)
    os.makedirs(root_dir, exist_ok=True)
    index = []
    for seed, ids in sorted(by_seed.items()):
        if seeds and seed not in seeds:
            continue
        t0 = time.time()
        max_depth = max(int(segments[i]["depth"]) for i in ids)
        survivors = {n["id"] for n in (beams.get(seed) or {}).get("beam") or []}
        leaves = [i for i in ids if (int(segments[i]["depth"]) == max_depth or i in survivors
                                     or (all_leaves and not children.get(i)))]
        needed: set[str] = set()
        for leaf in leaves:
            node = leaf
            while node in segments and node not in needed:
                needed.add(node)
                node = segments[node].get("parent")
        nodes = {}
        for node_id in needed:
            seg = segments[node_id]
            row = {k: seg.get(k) for k in SEGMENT_KEYS if k in seg}
            row["children"] = len(children.get(node_id, []))
            if node_id in judge:
                row["judge"] = {k: judge[node_id].get(k) for k in JUDGE_KEYS if k in judge[node_id]}
            if node_id in adherence:
                row["adherence"] = {k: adherence[node_id].get(k) for k in ADHERENCE_KEYS if k in adherence[node_id]}
            nodes[node_id] = row

        # Frame-by-frame replay of every needed node whose path from the root has turn logs.
        seed_dir = os.path.join(root_dir, str(seed))
        replayed = problems_total = 0
        roots = [i for i in needed if segments[i].get("parent") not in segments]
        if any(i in logs for i in needed):
            from craftax_env import glyph, turn_log, world

            if renderer is None:
                renderer = Renderer(px)
            names = list(glyph.ACTION_NAMES)
            os.makedirs(seed_dir, exist_ok=True)
            stack: list[tuple[str, Any, Any]] = []
            for r in roots:
                if r in logs:
                    stack.append((r, None, None))
            while stack:
                node_id, saved, explored = stack.pop()
                seg = turn_log.read(logs[node_id])
                game = turn_log.new_game(seg) if saved is None else glyph.CraftaxGame.restore(saved, seg.get("max_turns"))
                if explored is None:
                    explored = np.zeros(np.asarray(game.state.map).shape, dtype=bool)
                floor, mask = game.visible()
                explored[floor] |= mask
                frames = [renderer(game)]
                floors = {game.floor}
                problems: list[str] = []

                def on_turn(row: dict, g: Any) -> None:
                    fl, m = g.visible()
                    explored[fl] |= m
                    floors.add(g.floor)
                    frames.append(renderer(g))

                turn_log.replay_segment(game, seg, on_turn, problems.append, label=node_id)
                maps = {str(f): world.ascii_map(game, explored, f) for f in sorted(floors)}
                sheets = write_sheets(frames, os.path.join(seed_dir, node_id))
                with open(os.path.join(seed_dir, f"{node_id}.json"), "w") as f:
                    json.dump({"turns": [turn_row(r, names) for r in seg["turns"]],
                               "notes": seg.get("notes") or [], "maps": maps, "problems": problems},
                              f, separators=(",", ":"))
                nodes[node_id]["frames"] = sheets
                nodes[node_id]["replay_problems"] = len(problems)
                replayed += 1
                problems_total += len(problems)
                for p in problems[:3]:
                    print(f"  replay mismatch: {p}", file=sys.stderr)
                kids = [c for c in children.get(node_id, []) if c in needed and c in logs]
                if kids:
                    state = game.save()
                    for c in kids:
                        stack.append((c, state, explored.copy()))

        def ach(i: str) -> int:
            return len(segments[i].get("achievements") or [])

        leaves.sort(key=lambda i: (-int(segments[i]["depth"]), -ach(i), -float(segments[i].get("achievement_score") or 0)))
        payload = {
            "name": name, "cell": cell, "seed": seed, "max_depth": max_depth, "px": px,
            "leaves": [{"id": i, "depth": int(segments[i]["depth"]), "achievements": ach(i),
                        "score": segments[i].get("achievement_score"), "survivor": i in survivors,
                        "deaths": sum(len(segments[a].get("deaths") or []) for a in _path(i, segments))}
                       for i in leaves],
            "nodes": nodes,
        }
        with open(os.path.join(root_dir, f"{seed}.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        index.append({"seed": seed, "max_depth": max_depth, "leaves": len(leaves), "segments": len(nodes),
                      "replayed": replayed, "problems": problems_total,
                      "best": max((ach(i) for i in leaves), default=0)})
        print(f"{name} seed {seed}: depth {max_depth}, {len(leaves)} leaves, {len(nodes)} lineage segments, "
              f"{replayed} replayed ({problems_total} mismatches) in {time.time() - t0:.0f}s", file=sys.stderr)
    return {"name": name, "cell": cell, "dirs": [os.path.abspath(d) for d in dirs], "seeds": index}


def _path(node_id: str, segments: dict[str, dict]) -> list[str]:
    out = []
    while node_id in segments:
        out.append(node_id)
        node_id = segments[node_id].get("parent")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="run dirs of one chain (merged)")
    parser.add_argument("--name", help="name in the viewer (default: the first dir's basename)")
    parser.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "factory", "agent-rl", "craftax-replay"))
    parser.add_argument("--px", type=int, default=16, help="pixels per tile of the frames")
    parser.add_argument("--all-leaves", action="store_true", help="also every pruned node (no children), not just the last depth")
    parser.add_argument("--seeds", type=int, nargs="*", help="only these worlds")
    args = parser.parse_args()
    name = args.name or os.path.basename(os.path.normpath(args.dirs[0]))
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    entry = build(args.dirs, name, args.out, args.px, args.all_leaves, set(args.seeds or []) or None)
    index_path = os.path.join(args.out, "data", "runs.json")
    runs = {r["name"]: r for r in (json.load(open(index_path)) if os.path.exists(index_path) else [])}
    runs[name] = entry
    with open(index_path, "w") as f:
        json.dump(sorted(runs.values(), key=lambda r: r["name"]), f, indent=1)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lineage_replay.html"),
                os.path.join(args.out, "index.html"))
    print(f"wrote {args.out}/index.html; serve it with: cd {args.out} && python -m http.server 8765", file=sys.stderr)


if __name__ == "__main__":
    main()

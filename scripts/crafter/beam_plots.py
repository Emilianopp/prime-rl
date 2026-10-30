"""Plots over deep-beam chains: at what depth each method discovers (and keeps) each achievement.

Usage (matplotlib is not in the project venv; use uv):

    uv run --with matplotlib python scripts/crafter/beam_plots.py OUT_DIR label=CHAIN_DIR [label=CHAIN_DIR ...]

Each CHAIN_DIR is a beam results directory holding episodes.jsonl (segment records with
`depth`, `seed`, `node_id`, `parent`, `achievements_unlocked`). Writes:

  OUT_DIR/discovery.png   one row per achievement (tech-ladder order), one marker per method at
                          the first depth any node of the level holds it; filled = still held by
                          the last level, hollow = lost later; two panels, one per seed
  OUT_DIR/progress.png    max cumulative achievements per level vs depth, per method and seed
  OUT_DIR/discovery.tsv   the numbers behind the markers

Labels keep the order given; the first five get the categorical palette in fixed order.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LADDER = [  # Crafter's 22 achievements in tech-tree order (hand tools, then stone, then iron)
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink", "make_wood_pickaxe",
    "make_wood_sword", "place_plant", "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
    "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe", "make_iron_sword",
]
# dataviz reference palette, categorical slots 1-5 (light mode) + marker shapes as the second encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(chain_dir: str):
    """Per seed: {depth: [records]} plus the parent ids (nodes that were kept and branched)."""
    levels: dict[int, dict[int, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    parents: dict[int, set[str]] = collections.defaultdict(set)
    done: dict[int, int] = {}  # the beam checkpoint's depth per seed: records deeper than it are a level cut mid-way
    for f in glob.glob(os.path.join(chain_dir, "states", "beam", "*.json")):
        with open(f) as fh:
            b = json.load(fh)
        done[b["seed"]] = b["depth"]
    with open(os.path.join(chain_dir, "episodes.jsonl")) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task") == "crafter-judge" or "achievements_unlocked" not in r:
                continue
            if r["depth"] > done.get(r["seed"], r["depth"]):
                continue
            levels[r["seed"]][r["depth"]].append(r)
            if r.get("parent"):
                parents[r["seed"]].add(r["parent"])
    return levels, parents


def discovery(levels, parents):
    """Per seed: achievement -> (first depth seen in any node, first depth seen in a kept node, held at the last level)."""
    out = {}
    for seed, by_depth in levels.items():
        last = max(by_depth)
        first_any, first_kept = {}, {}
        for d in sorted(by_depth):
            for r in by_depth[d]:
                kept = r["node_id"] in parents[seed] or d == last
                for a in r["achievements_unlocked"]:
                    first_any.setdefault(a, d)
                    if kept:
                        first_kept.setdefault(a, d)
        held = {a for r in by_depth[last] for a in r["achievements_unlocked"]}
        out[seed] = {a: (first_any[a], first_kept.get(a), a in held) for a in first_any}
    return out


def progress(levels):
    """Per seed: depth -> (max cumulative achievements in the level, max among the kept nodes)."""
    return {seed: {d: max(len(r["achievements_unlocked"]) for r in recs) for d, recs in by_depth.items()} for seed, by_depth in levels.items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(out_dir: str, chains: list[tuple[str, str]]):
    os.makedirs(out_dir, exist_ok=True)
    data = {label: load(path) for label, path in chains}
    seeds = sorted({s for levels, _ in data.values() for s in levels})
    disc = {label: discovery(*lp) for label, lp in data.items()}
    prog = {label: progress(lp[0]) for label, lp in data.items()}
    depth_of = {label: {s: max(lp[0][s]) for s in lp[0]} for label, lp in data.items()}
    n = len(chains)

    # --- discovery.png ---
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.2 * len(seeds) + 1.5, 7.2), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(seeds) > 1 else [axes]
    rows = [a for a in LADDER if any(a in disc[l].get(s, {}) for l in disc for s in seeds)]
    xmax = max(max(dd.values()) for dd in depth_of.values())
    xdisc = max(v[0] for d in disc.values() for s in d.values() for v in s.values())
    for ax, seed in zip(axes, seeds):
        style(ax)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.replace("_", " ") for r in rows], fontsize=9, color=INK)
        ax.invert_yaxis()
        for i, (label, _) in enumerate(chains):
            off = (i - (n - 1) / 2) * 0.16
            for y, a in enumerate(rows):
                if a not in disc[label].get(seed, {}):
                    continue
                d_any, d_kept, held = disc[label][seed][a]
                ax.plot(
                    d_any, y + off, marker=MARKERS[i], markersize=7, linestyle="none",
                    markerfacecolor=COLORS[i] if held else SURFACE, markeredgecolor=COLORS[i], markeredgewidth=1.6,
                    zorder=3,
                )
        ax.set_xlim(0.3, xdisc + 1.2)
        ax.set_xlabel("depth of the first level holding it", color=INK2, fontsize=9)
        ax.set_title(f"seed {seed}", color=INK, fontsize=11, loc="left")
    handles = [
        Line2D([], [], marker=MARKERS[i], color=COLORS[i], linestyle="none", markersize=7,
               label=f"{label} (depth {', '.join(str(depth_of[label][s]) for s in seeds if s in depth_of[label])})")
        for i, (label, _) in enumerate(chains)
    ]
    handles += [
        Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=7, label="held by the last level"),
        Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE, linestyle="none", markersize=7, label="lost later"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Depth at which each method first discovers each achievement", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "discovery.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # --- progress.png ---
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.2 * len(seeds) + 1.5, 4.2), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(seeds) > 1 else [axes]
    for ax, seed in zip(axes, seeds):
        style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        for i, (label, _) in enumerate(chains):
            series = prog[label].get(seed)
            if not series:
                continue
            xs = sorted(series)
            ax.plot(xs, [series[x] for x in xs], color=COLORS[i], linewidth=2, marker=MARKERS[i], markersize=3.5, label=label, zorder=3)
            if xs[-1] == xmax:  # direct labels only where the line ends at the right edge (the rest are told apart by length and the legend)
                ax.annotate(label, (xs[-1], series[xs[-1]]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.set_xlim(0.5, xmax + 0.5)
        ax.set_ylim(0, 22)
        ax.set_xlabel("depth", color=INK2, fontsize=9)
        ax.set_title(f"seed {seed}", color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("max cumulative achievements in the level", color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, n), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Beam progress by depth", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "progress.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # --- discovery.tsv ---
    with open(os.path.join(out_dir, "discovery.tsv"), "w") as fh:
        fh.write("method\tseed\tachievement\tfirst_depth\tfirst_kept_depth\theld_at_last\tlast_depth\n")
        for label, _ in chains:
            for seed in seeds:
                for a in LADDER:
                    if a in disc[label].get(seed, {}):
                        d_any, d_kept, held = disc[label][seed][a]
                        fh.write(f"{label}\t{seed}\t{a}\t{d_any}\t{d_kept if d_kept is not None else ''}\t{int(held)}\t{depth_of[label][seed]}\n")
    print(f"wrote {out_dir}/discovery.png, progress.png, discovery.tsv")
    for label, _ in chains:
        for seed in seeds:
            if seed in disc[label]:
                held = sum(h for _, _, h in disc[label][seed].values())
                print(f"  {label} seed {seed}: depth {depth_of[label][seed]}, discovered {len(disc[label][seed])}, held at the end {held}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or any("=" not in a for a in sys.argv[2:]):
        sys.exit(__doc__)
    main(sys.argv[1], [tuple(a.split("=", 1)) for a in sys.argv[2:]])

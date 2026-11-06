"""Judge reward vs true game progress per 3 h block of a qv chain, side by side (mean +- s.e.).
Blocks with games still running are left out (MAX_UPDATE).

usage: plot_rewards.py REWARDS_JSON OUT_PNG [TITLE] [MAX_UPDATE]
       (REWARDS_JSON from extract_rewards.py)"""
import collections
import json
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ms(v):
    return st.mean(v), st.stdev(v) / len(v) ** 0.5


def main():
    data = json.load(open(sys.argv[1]))
    title = sys.argv[3] if len(sys.argv) > 3 else ""
    max_update = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
    blocks = collections.defaultdict(list)
    for upd, R, _ in data["segments"]:
        blocks[upd // 4 * 4].append(R)
    ach = {int(k): v for k, v in data["achievements"].items()}
    xs = sorted(k for k in blocks if k < max_update and len(ach.get(k, [])) > 1)
    ink, muted, grid, c = "#1f2328", "#6e7781", "#e6e8eb", "#2f6fdb"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=160)
    panels = ((axes[0], [ms(blocks[x]) for x in xs], "Judge reward R (what we train on)", "mean R per segment"),
              (axes[1], [ms(ach[x]) for x in xs], "Total achievements per game (end of 3 h game)", "achievements unlocked (of 67)"))
    for ax, m, head, ylab in panels:
        ax.errorbar(range(len(xs)), [a for a, _ in m], yerr=[e for _, e in m], color=c, lw=2, marker="o", ms=7,
                    capsize=3, elinewidth=1.2, mec="white", mew=1.5)
        for i, (a, e) in enumerate(m):
            ax.annotate(f"{a:.2f}", (i, a + e), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=ink)
        ax.set_xticks(range(len(xs)), [f"{x}-{x + 3}" for x in xs], fontsize=8, color=muted)
        ax.set_xlabel("updates (one 3 h block each, same 64 worlds)", fontsize=8, color=muted)
        ax.set_ylabel(ylab, fontsize=8, color=muted)
        ax.set_title(head, fontsize=10, color=ink, loc="left")
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.12)
    if title:
        fig.suptitle(title, fontsize=9, color=muted, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()

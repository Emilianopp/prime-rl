"""Summarize a pokemon-env episode log: python scripts/pokemon/summarize_episodes.py episodes.jsonl"""

import collections
import json
import statistics
import sys


def main(path: str):
    eps = [json.loads(line) for line in open(path)]
    if not eps:
        sys.exit(f"no episodes in {path}")
    n = len(eps)
    reached = collections.Counter(m for e in eps for m in e["milestones"])
    first_at = collections.defaultdict(list)  # milestone -> model turn it was reached at
    for e in eps:
        for m, turn in e["milestones"].items():
            first_at[m].append(turn)
    print(f"episodes {n}  reward mean {statistics.mean(e['reward'] for e in eps):.4f}  "
          f"milestones mean {statistics.mean(len(e['milestones']) for e in eps):.2f}  "
          f"max {max(len(e['milestones']) for e in eps)}")
    print(f"turns median {statistics.median(e['steps'] for e in eps)}  ended_by {dict(collections.Counter(e['ended_by'] for e in eps))}")
    turns = sum(e["steps"] for e in eps)
    print(f"malformed replies {sum(e['malformed_replies'] for e in eps) / max(turns, 1):.3%} of turns  "
          f"trace tokens median {statistics.median(e['trace_tokens'] for e in eps):.0f}  "
          f"wall median {statistics.median(e['end'] - e['start'] for e in eps):.0f}s")
    print("milestone            episodes   median turn")
    for m, k in sorted(reached.items(), key=lambda kv: (-kv[1], statistics.median(first_at[kv[0]]))):
        print(f"  {m:<20} {k:>3}/{n:<4} {statistics.median(first_at[m]):>8.0f}")


if __name__ == "__main__":
    main(sys.argv[1])

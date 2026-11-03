"""Compare judge models graded on the same states by reliability.py.

For each quantity: the candidate's per-state mean against the reference's (Pearson r,
mean absolute difference, mean signed difference), the candidate's own ICC, and the
cost per call from the logged usage (thinking tokens, hidden from the completion count,
billed as output).

    python compare_judges.py <ref.json> <cand.json> [<cand.json> ...] --price model=in,out ...
"""

import json
import statistics as st
import sys

QUANTITIES = ("P", "V", "G_star", "D", "Q", "R", "n_flaws")


def pearson(xs, ys):
    if len(xs) < 3 or st.pstdev(xs) == 0 or st.pstdev(ys) == 0:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * st.pstdev(xs) * st.pstdev(ys))


def usage(path):
    n = p = out = 0
    for line in open(path):
        u = json.loads(line).get("usage") or {}
        n += 1
        p += u.get("prompt_tokens", 0) or 0
        out += (u.get("total_tokens", 0) or 0) - (u.get("prompt_tokens", 0) or 0)
    return n, p, out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--price") and "=" not in a]
    prices = dict((k, tuple(map(float, v.split(",")))) for k, v in (a.split("=") for a in sys.argv[1:] if "=" in a))
    ref = json.load(open(args[0]))
    rows = []
    for path in [args[0], *args[1:]]:
        run = json.load(open(path))
        model = run["report"]["model"] if "model" in run["report"] else "gemini-3.8-flash"
        n, p, out = usage(path + ".calls.jsonl")
        pin, pout = prices.get(model, (None, None))
        cost = (p * pin + out * pout) / 1e6 / n if pin is not None and n else None
        row = {"model": model, "calls": n, "in/call": p / n if n else None, "out/call": out / n if n else None,
               "$/call": cost}
        for q in QUANTITIES:
            a = [st.mean(g) if g else None for g in ref["groups"][q]]
            b = [st.mean(g) if g else None for g in run["groups"][q]]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
            row[q] = {"r": pearson(xs, ys), "mad": st.mean(abs(x - y) for x, y in pairs) if pairs else None,
                      "bias": st.mean(y - x for x, y in pairs) if pairs else None,
                      "icc": run["report"][q]["icc"], "within_sd": run["report"][q]["within_sd"]}
        rows.append(row)
    for row in rows:
        f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
        print(f"\n== {row['model']}: {row['calls']} calls, {f(row['in/call'], 0)} in + {f(row['out/call'], 0)} out per call, "
              f"${f(row['$/call'], 4)}/call")
        for q in QUANTITIES:
            r = row[q]
            print(f"  {q:8s} r vs ref {f(r['r'])}  |diff| {f(r['mad'])}  bias {f(r['bias'])}  ICC {f(r['icc'])}  repeat sd {f(r['within_sd'])}")
    json.dump(rows, open(args[0].replace(".json", "_compare.json"), "w"))


if __name__ == "__main__":
    main()

"""Build the Handoff Ledger page (one HTML file, data inlined) from judge.py output.

    python ledger.py <judged.json> <out.html> [branch results.json]

With branch_test.py's results it adds the branch test: per test reset, the progress of
the six handoffs played on from it.
"""
import json
import sys
from pathlib import Path


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    a, b = ranks(xs), ranks(ys)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else None


def main():
    data = json.load(open(sys.argv[1]))
    pts = []  # one per segment with both ends judged
    flaw_counts: dict[str, int] = {}
    summaries = with_false = 0
    for ep in data:
        st = ep["states"]
        for s in st:
            j = s.get("judge")
            if j and s["kind"] == "reset" and s.get("summary"):
                summaries += 1
                types = [f.get("type") for f in j["flaws"]]
                with_false += "false_fact" in types
                for t in types:
                    flaw_counts[t] = flaw_counts.get(t, 0) + 1
        for t, seg in enumerate(ep["segments"]):
            a, b = st[t].get("judge"), st[t + 1].get("judge")
            if not a or not b:
                continue
            pts.append({"seed": ep["seed"], "t": t, "G": a["G_star"], "dV": b["V"] - a["V"],
                        "dB": st[t + 1]["B"] - st[t]["B"], "D": a["D"], "R": seg["R"],
                        "S": bool(st[t].get("summary"))})
    with_d = [p for p in pts if p["D"] is not None and p["S"]]  # graded summaries only
    stats = {
        "games": len(data),
        "states": sum(1 for ep in data for s in ep["states"] if s.get("judge")),
        "segments": len(pts),
        "summaries": summaries,
        "false_share": with_false / summaries if summaries else None,
        "mean_D": sum(p["D"] for p in with_d) / len(with_d) if with_d else None,
        "mean_G": sum(p["G"] for p in pts) / len(pts) if pts else None,
        "mean_dV": sum(p["dV"] for p in pts) / len(pts) if pts else None,
        "d_saturated": sum(1 for p in with_d if p["D"] >= 0.999) / len(with_d) if with_d else None,
        "rho_G_dV": spearman([p["G"] for p in pts], [p["dV"] for p in pts]),
        "rho_D_dV": spearman([p["D"] for p in with_d], [p["dV"] for p in with_d]),
        "rho_D_dB": spearman([p["D"] for p in with_d], [p["dB"] for p in with_d]),
        "flaws": dict(sorted(flaw_counts.items(), key=lambda x: -x[1])),
    }
    branch = None
    if len(sys.argv) > 3:
        res = json.load(open(sys.argv[3]))
        branch = {"stats": res["stats"], "groups": [
            {"seed": g["seed"], "k": g["k"], "V0": g["V0"], "turn": g["state"]["turn"], "B": g["state"]["B"],
             "original": g["original"], "rho": g["rho_D_dV"], "branches": [
                 {"label": b["label"], "summary": b["summary"], "D": b["start"]["D"], "flaws": b["start"]["flaws"],
                  "dV": b["dV"], "dB": b["dB"], "deaths": b["deaths"], "Vend": b["end"]["V"],
                  "unlocked": sorted(set(b["end_state"]["achievements"]) - set(g["state"]["achievements"]))}
                 for b in g["branches"]]} for g in res["groups"]]}
    html = (Path(__file__).parent / "ledger_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps({"episodes": data, "points": pts, "stats": stats, "branch": branch}))
    open(sys.argv[2], "w").write(html)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

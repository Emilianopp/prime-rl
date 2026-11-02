"""Does a handoff summary the judge scores better make the next segment go better?

Holds the game fixed and varies only the handoff. At each test reset (one per game,
spread over early, mid and late resets) the game is played on for one segment from six
handoffs: four summaries the model samples again at that moment (its own prompt as the
turn window left it), the judge's ideal handoff, and none at all. Then the judge grades
each handoff (D) at the shared start and each branch's end state (V).

  capture   (CPU)   replay the chosen games to each test reset; keep the env save and
                    the summary turn's prompt ids                      -> capture.pkl
  sample    (GPU)   with the policy served on --base-url: sample 4 summaries per reset,
                    write each branch's env save to <out>/state/{seed}-{variant}.pkl
                    and the manifest                                   -> manifest.json
  (eval)            branch.sbatch plays every branch for one segment (taskset
                    episodes x 6 variants, env max_segments = 1, state_dir = <out>/state)
  states    (CPU)   the end state of every branch from its save      -> states.json
  judge     (login) judge every handoff and end state, the statistics -> results.json

    python branch_test.py capture <judged.json> <eval_dir> <out> [n_resets]
    python branch_test.py sample <out> [--base-url URL] [--samples 4]
    python branch_test.py states <out>
    GEMINI_API_KEY=... python branch_test.py judge <out>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

VARIANTS = ["sample 1", "sample 2", "sample 3", "sample 4", "judge ideal", "empty"]
EVAL_SEED_BASE = 1_000_042  # CraftaxConfig: eval seed = seed (42) + 1_000_000 + episode index
THINK_END = "</think>"


def pick_resets(judged: list[dict], n: int) -> list[tuple[int, int]]:
    """(seed, reset) pairs: one reset per game, cycling early / mid / late, each with a
    summary and a following segment."""
    picks = []
    for i, ep in enumerate(sorted(judged, key=lambda e: e["seed"])):
        ks = [k for k, s in enumerate(ep["states"])
              if s["kind"] == "reset" and s.get("summary") and k < len(ep["segments"]) and s.get("judge")]
        if not ks:
            continue
        phase = len(picks) % 3
        picks.append((ep["seed"], ks[0] if phase == 0 else ks[len(ks) // 2] if phase == 1 else ks[-1]))
        if len(picks) == n:
            break
    return picks


def capture(args):
    import replay

    judged = json.load(open(args.judged))
    picks = pick_resets(judged, args.n)
    wanted = {seed: k for seed, k in picks}
    ideal = {ep["seed"]: {k: s["judge"]["ideal_handoff"] for k, s in enumerate(ep["states"]) if s.get("judge")}
             for ep in judged}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resets = []
    for ep in replay.episodes(args.eval_dir):
        seed = int(ep["traces"][0]["task"]["data"]["info"]["seed"])
        if seed not in wanted:
            continue
        k = wanted.pop(seed)
        rec = replay.replay(ep, capture={k})
        cap = rec["captured"][k]
        resets.append({"seed": seed, "episode": seed - EVAL_SEED_BASE, "k": k, "rules": rec["rules"],
                       "state": rec["resets"][k], "original": rec["resets"][k]["summary"],
                       "inherited": cap["inherited"], "ideal": ideal[seed][k],
                       "save": cap["save"], "prompt_ids": cap["prompt_ids"]})
        print(f"seed {seed} reset {k}: prompt {len(cap['prompt_ids']):,} tokens, "
              f"HUD mismatches {rec['hud_mismatched']}/{rec['hud_checked']}", flush=True)
        if not wanted:
            break
    pickle.dump(resets, open(out / "capture.pkl", "wb"))
    print(f"captured {len(resets)} resets -> {out / 'capture.pkl'}")


def summary_of(text: str) -> tuple[str, str]:
    """(summary, reply) from a sampled completion that starts inside the think block, as
    the segment switch reads it: the reply after the think block, its <summary> text (the
    whole reply without the tag), else the reasoning's."""
    from verifiers.v1.segments import _summary_text

    reasoning, _, reply = text.partition(THINK_END) if THINK_END in text else ("", "", text)
    reply = reply.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    summary = _summary_text(reply)
    return (summary or _summary_text(reasoning)), reply


async def sample_one(client, base_url: str, tokenizer, ids: list[int], max_tokens: int) -> str:
    """One completion of prompt `ids` from SGLang's native /generate. The engines run
    token-in/token-out (no detokenizer), so the reply comes back as ids and is decoded here."""
    root = base_url.rsplit("/v1", 1)[0]
    r = await client.post(f"{root}/generate", json={"input_ids": ids, "sampling_params": {
        "temperature": 1.0, "max_new_tokens": max_tokens}}, timeout=3600)
    r.raise_for_status()
    out = r.json()
    out = out[0] if isinstance(out, list) else out
    if out.get("text") is not None:
        return out["text"]
    return tokenizer.decode(out["output_ids"], skip_special_tokens=False)


def sample(args):
    import httpx
    from verifiers.v1.memory import Memory

    out = Path(args.out)
    resets = pickle.load(open(out / "capture.pkl", "rb"))
    state = out / "state"
    state.mkdir(exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    async def run():
        async with httpx.AsyncClient() as client:
            jobs = [sample_one(client, args.base_url, tokenizer, r["prompt_ids"], args.max_tokens)
                    for r in resets for _ in range(args.samples)]
            return await asyncio.gather(*jobs)

    texts = asyncio.run(run())
    manifest = []
    for i, r in enumerate(resets):
        branches = []
        for v, label in enumerate(VARIANTS):
            memory = dict(r["save"]["memory"] or {})
            if v < args.samples:
                summary, reply = summary_of(texts[i * args.samples + v])
                mem = Memory(memory, max_key_tokens=4096)
                mem.apply(reply)  # the summary turn's own memory edits
                memory = dict(mem.entries)
            else:
                summary, reply = (r["ideal"], None) if label == "judge ideal" else ("", None)
            save = {**r["save"], "summary": summary, "last_summary": summary, "memory": memory}
            pickle.dump(save, open(state / f"{r['seed']}-{v}.pkl", "wb"))
            branches.append({"variant": v, "label": label, "summary": summary, "reply": reply})
        manifest.append({**{x: r[x] for x in ("seed", "episode", "k", "rules", "state", "original", "inherited")},
                         "branches": branches})
        print(f"seed {r['seed']} reset {r['k']}: " + " | ".join(f"{len(b['summary'])} ch" for b in branches), flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"))
    (out / "episodes.txt").write_text(",".join(str(r["episode"]) for r in resets))
    print(f"{len(resets)} resets x {len(VARIANTS)} branches -> {state}")


def states(args):
    import replay
    from craftax_env import glyph

    out = Path(args.out)
    manifest = json.load(open(out / "manifest.json"))
    ends = {}
    for r in manifest:
        for b in r["branches"]:
            path = out / "state" / f"{r['seed']}-{b['variant']}.pkl"
            save = pickle.load(open(path, "rb"))
            game = glyph.CraftaxGame(r["seed"], None)
            game.restore(save["game"])
            rec = replay.state_record(game, len(save["deaths"]), "end")
            rec.update(turns=save["turns"], ended_summary=save.get("last_summary"))
            ends[f"{r['seed']}:{b['variant']}"] = rec
    json.dump(ends, open(out / "states.json", "w"))
    print(f"{len(ends)} end states -> {out / 'states.json'}")


def boot_mean(xs, n=4000, rng=random.Random(0)):
    """Mean and 95% bootstrap interval."""
    if not xs:
        return None, None, None
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n)]


def judge_cmd(args):
    import judge
    from ledger import spearman

    out = Path(args.out)
    manifest = json.load(open(out / "manifest.json"))
    ends = json.load(open(out / "states.json"))
    jobs = {}
    for r in manifest:
        for b in r["branches"]:
            key = f"{r['seed']}:{b['variant']}"
            jobs[f"start:{key}"] = judge.prompt(r["rules"], r["state"], True, b["summary"], r["inherited"])
            jobs[f"end:{key}"] = judge.prompt(r["rules"], ends[key], False)
    replies = asyncio.run(judge.judge_all(jobs, str(out / "judge_calls.jsonl"), args.concurrency))
    groups, rows = [], []
    for r in manifest:
        start = r["state"]
        bs = []
        for b in r["branches"]:
            key = f"{r['seed']}:{b['variant']}"
            js, je = replies.get(f"start:{key}"), replies.get(f"end:{key}")
            if js is None or je is None:
                continue
            s = judge.score(start, js, True, b["summary"])
            e = judge.score(ends[key], je, False)
            bs.append({**b, "start": s, "end": e, "end_state": ends[key]})
        if not bs:
            continue
        v0 = sum(b["start"]["V"] for b in bs) / len(bs)  # one state, judged once per branch: average
        for b in bs:
            b["dV"] = b["end"]["V"] - v0
            b["dB"] = b["end_state"]["B"] - start["B"]
            b["deaths"] = b["end_state"]["deaths"] - start["deaths"]
            rows.append({"seed": r["seed"], "label": b["label"], "D": b["start"]["D"], "dV": b["dV"], "dB": b["dB"]})
        sampled = [b for b in bs if b["label"].startswith("sample")]
        by = {b["label"]: b for b in bs}
        groups.append({**{x: r[x] for x in ("seed", "k", "state", "original", "inherited")}, "V0": v0, "branches": bs,
                       "rho_D_dV": spearman([b["start"]["D"] for b in sampled], [b["dV"] for b in sampled]),
                       "rho_D_dB": spearman([b["start"]["D"] for b in sampled], [b["dB"] for b in sampled]),
                       "ideal_minus_empty_dV": by["judge ideal"]["dV"] - by["empty"]["dV"] if "judge ideal" in by and "empty" in by else None,
                       "sampled_minus_empty_dV": (sum(b["dV"] for b in sampled) / len(sampled) - by["empty"]["dV"]) if sampled and "empty" in by else None,
                       "sd_sampled_dV": (sum((b["dV"] - sum(x["dV"] for x in sampled) / len(sampled)) ** 2 for b in sampled) / max(1, len(sampled) - 1)) ** 0.5 if len(sampled) > 1 else None})

    def col(key):
        return [g[key] for g in groups if g[key] is not None]

    stats = {
        "resets": len(groups),
        "ideal_minus_empty_dV": boot_mean(col("ideal_minus_empty_dV")),
        "sampled_minus_empty_dV": boot_mean(col("sampled_minus_empty_dV")),
        "rho_D_dV_within": boot_mean(col("rho_D_dV")),
        "rho_D_dB_within": boot_mean(col("rho_D_dB")),
        "sd_sampled_dV": boot_mean(col("sd_sampled_dV")),
        "mean_dV": {l: boot_mean([x["dV"] for x in rows if x["label"] == l]) for l in VARIANTS},
        "mean_dB": {l: boot_mean([x["dB"] for x in rows if x["label"] == l]) for l in VARIANTS},
        "mean_D": {l: boot_mean([x["D"] for x in rows if x["label"] == l]) for l in VARIANTS},
    }
    json.dump({"stats": stats, "groups": groups}, open(out / "results.json", "w"))
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("judged")
    c.add_argument("eval_dir")
    c.add_argument("out")
    c.add_argument("n", type=int, nargs="?", default=20)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--base-url", default="http://localhost:8000/v1")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
    s.add_argument("--samples", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=32768)
    st = sub.add_parser("states")
    st.add_argument("out")
    j = sub.add_parser("judge")
    j.add_argument("out")
    j.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    {"capture": capture, "sample": sample, "states": states, "judge": judge_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Check a kve-summary RL run against the intended segment pipeline.

    python scripts/crafter/check_pipeline.py $SCRATCH/factory/runs/crafter-kve-summary/<job>

The scenario being checked, one section per step:
  1. cut     a segment ends at the cut (turns or positions, under the 131k context),
             writes its notes, saves its world and closes its session
  2. resume  the next segment of the lineage restores exactly that world
             (game step, achievements, inventory, position)
  3. respawn a death inside a segment respawns the player and the segment keeps going
  4. judge   every finished segment is scored by the LLM judge (no fallback)
  5. overlap sampling keeps going while the trainer trains; the new weights land after
  6. stale   how old the policy behind each trained segment is, and segments that
             straddled a weight update
Each section ends in PASS, WARN or FAIL. Reads episodes.jsonl, the trace stream,
the trainer and orchestrator logs and metrics.jsonl from the run directory.
"""

from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

VITALS = {"health", "food", "drink", "energy"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
results: list[tuple[str, str, str]] = []


def verdict(section: str, status: str, why: str) -> None:
    results.append((section, status, why))
    print(f"  -> {status}: {why}")


def q(values, fmt="{:.0f}"):
    if not values:
        return "-"
    v = sorted(values)
    pick = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return " / ".join(f"{name} {fmt.format(pick(p))}" for name, p in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))) + f" / max {fmt.format(v[-1])}"


def duration_s(text: str) -> float:
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(h|ms|m|s)", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def clock_to_epoch(clock: str, ref: float) -> float:
    """HH:MM:SS from a log line, on the day of `ref` (a nearby epoch), wrapped to the closest day."""
    day = dt.datetime.fromtimestamp(ref)
    h, m, s = (int(x) for x in clock.split(":"))
    t = day.replace(hour=h, minute=m, second=s, microsecond=0).timestamp()
    for shift in (-86400, 86400):
        if abs(t + shift - ref) < abs(t - ref):
            t += shift
    return t


def load_stream(run: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(run, "monitors/file/traces/stream/*.jsonl.zst")))
    if not files:
        return []
    import zstandard as zstd

    episodes = []
    for f in files:
        with open(f, "rb") as fh:
            text = zstd.ZstdDecompressor().stream_reader(fh).read().decode()
        episodes += [json.loads(l) for l in text.splitlines() if l.strip()]
    return episodes


def main(out: str) -> None:
    lines = [json.loads(l) for l in open(os.path.join(out, "episodes.jsonl")) if l.strip()]
    segs = [l for l in lines if l.get("task") == "crafter"]
    judged = {l["node_id"]: l for l in lines if l.get("task") == "crafter-judge"}
    by_id = {s["node_id"]: s for s in segs}
    run = next(iter(sorted(glob.glob(os.path.join(out, "run/*/logs")))), None)
    run = os.path.dirname(run) if run else None
    cfg = open(os.path.join(out, "rl.toml")).read() if os.path.exists(os.path.join(out, "rl.toml")) else ""
    get = lambda key, default=None: (re.search(rf"^{re.escape(key)}\s*=\s*([^#\n]+)", cfg, re.M) or [None, default])[1]
    seg_turns, seg_tokens = get("env.segment_turns"), get("env.segment_tokens")
    ctx = get("max_model_len", "131072")
    print(f"{out}\n{len(segs)} segments, {len(judged)} judge lines; config: segment_turns={seg_turns} segment_tokens={seg_tokens} "
          f"max_model_len={ctx} on_death={get('env.on_death')} respawn_at={get('env.respawn_at')} respawn_safe={get('env.respawn_safe')} cut_on_death={get('env.cut_on_death')}")

    # 1. cut -------------------------------------------------------------------
    print("\n1. cut: segments end at the cut, write notes, save the world, close")
    ends = collections.Counter(s["ended_by"] for s in segs)
    tokens = [s["trace_tokens"] for s in segs]
    cont = [s for s in segs if s["continues"]]
    no_state = [s for s in cont if not s.get("state_path") or not s.get("state_bytes")]
    no_notes = [s for s in cont if not s.get("summary")]
    print(f"  ended_by {dict(ends)}; turns {q([s['steps'] for s in segs])}; trace tokens {q(tokens)} (context {ctx})")
    print(f"  continuing segments {len(cont)}: without a saved state {len(no_state)}, without notes {len(no_notes)}; "
          f"state size {q([s.get('state_bytes', 0) / 1000 for s in cont], '{:.0f}')} KB; notes chars {q([s.get('summary_chars', 0) for s in cont])}")
    retried = collections.Counter(s.get("summary_retry_reason") or "unrecorded" for s in cont if s.get("summary_retried"))
    print(f"  notes retried {sum(retried.values())}/{len(cont)}: reasons {dict(retried)}")
    bad_end = {k: v for k, v in ends.items() if k not in {"segment_turns", "segment_tokens", "death_cut", "final_goal"}}
    over = [t for t in tokens if t > int(ctx)]
    if bad_end or over or no_state:
        verdict("cut", "FAIL", f"unexpected ends {bad_end}, {len(over)} over the context, {len(no_state)} unsaved")
    else:
        verdict("cut", "PASS" if not no_notes else "WARN", f"every segment ended at a cut and saved its world; {len(no_notes)} wrote no notes")
    if run:
        log = ANSI.sub("", open(os.path.join(run, "logs/attempt_1/inference.log"), errors="ignore").read())
        released = log.count("Session KV released")
        print(f"  inference log: {released} session KV releases for {len(segs)} segments (one per segment: the engine frees each segment's KV at its end)")
        if released < len(segs):
            verdict("close", "WARN", f"only {released} session releases for {len(segs)} segments")

    # 2. resume ----------------------------------------------------------------
    print("\n2. resume: a child segment starts from its parent's saved world")
    checked, mismatches = 0, collections.Counter()
    for s in segs:
        parent = by_id.get(s.get("parent"))
        if not parent or s["depth"] < 2:
            continue
        checked += 1
        start = s["game_step"] - s["steps"]
        if start != parent["game_step"]:
            mismatches["game_step"] += 1
        if sorted(s.get("achievements_before") or []) != sorted(parent.get("achievements_unlocked") or []):
            mismatches["achievements"] += 1
        inv = {k: v for k, v in (parent.get("inventory") or {}).items() if k not in VITALS and v}
        if (s.get("inventory_before") or {}) != inv:
            mismatches["inventory"] += 1
        if s.get("position_before") != parent.get("position"):
            mismatches["position"] += 1
    depths = collections.Counter(s["depth"] for s in segs)
    print(f"  segments per depth {dict(sorted(depths.items()))}; checked {checked} parent-child pairs; mismatches {dict(mismatches) or 'none'}")
    if not checked:
        verdict("resume", "WARN", "no depth >= 2 segment with its parent in the log")
    else:
        verdict("resume", "FAIL" if mismatches else "PASS", f"{checked} children restored their parent's world" if not mismatches else f"mismatches {dict(mismatches)}")

    # 3. respawn ---------------------------------------------------------------
    print("\n3. respawn: a death respawns the player and play continues")
    with_death = [s for s in segs if s.get("deaths")]
    deaths = [d for s in with_death for d in s["deaths"]]
    played_after = [s["steps"] - s["deaths"][-1]["step"] for s in with_death if s["ended_by"] != "death_cut"]
    causes = collections.Counter(d.get("cause") or ("starved: " + ",".join(k for k in ("food", "drink", "energy") if d["vitals"].get(k, 1) == 0) if any(d["vitals"].get(k, 1) == 0 for k in ("food", "drink", "energy")) else "damage") for d in deaths)  # `cause` since 2026-10-26
    moved = [max(abs(a - b) for a, b in zip(d["pos"], d["respawn"])) for d in deaths]
    print(f"  {len(deaths)} deaths in {len(with_death)} of {len(segs)} segments; causes {dict(causes)}; "
          f"respawn distance from the death site {q(moved)}; turns played after the last death {q(played_after)}")
    if ends.get("died"):
        verdict("respawn", "FAIL", f"{ends['died']} segments ended by death: respawn is off")
    elif not deaths:
        verdict("respawn", "WARN", "no death happened, respawn not exercised (expected with short segments)")
    else:
        verdict("respawn", "PASS", f"{len(deaths)} deaths respawned; the game continued in the same segment or the next")

    # 4. judge -----------------------------------------------------------------
    print("\n4. judge: every finished segment gets an LLM score")
    missing = [s for s in segs if s["node_id"] not in judged]
    failed = [j for j in judged.values() if j.get("judge_failed")]
    latency = [judged[s["node_id"]]["at"] - s["end"] for s in segs if s["node_id"] in judged]
    scores = [j["judge_score"] for j in judged.values() if not j.get("judge_failed")]
    groups = collections.defaultdict(list)
    for s in segs:
        if s["node_id"] in judged and not judged[s["node_id"]].get("judge_failed"):
            groups[s["parent"]].append(judged[s["node_id"]]["judge_score"])
    spread = [max(g) - min(g) for g in groups.values() if len(g) > 1]
    print(f"  judged {len(judged)}/{len(segs)}, fallbacks {len(failed)}; judge latency after the segment's end {q(latency, '{:.1f}')} s")
    print(f"  scores {q(scores, '{:.2f}')}; within-node spread {q(spread, '{:.2f}')}, flat groups {sum(x == 0 for x in spread)}/{len(spread)}")
    modes = collections.Counter(j.get("mode", "single") for j in judged.values())
    stat = lambda key: [float(j[key]) for j in judged.values() if j.get(key) is not None]
    def line(name, key):
        v = stat(key)
        if v:
            print(f"  {name}: mean {statistics.mean(v):.1f} s, {q(v, '{:.1f}')} s")
    print(f"  judge mode {dict(modes)}")
    cost = sum(float(j.get("cost_usd") or 0) for j in judged.values())
    if any("cost_usd" in j for j in judged.values()):
        tin = sum(float(j.get("input_tokens") or 0) for j in judged.values())
        tout = sum(float(j.get("output_tokens") or 0) for j in judged.values())
        print(f"  judge cost ${cost:.2f} (input {tin / 1e6:.2f}M, output {tout / 1e6:.2f}M tokens incl. thinking): "
              f"${1000 * cost / max(1, len(judged)):.2f} per 1000 segments")
    line("wait from segment end to score", "wait_s")
    line("waiting for the group (siblings + call)", "group_wait_s")
    line("queued behind the concurrency limit", "queue_s")
    line("judge call", "call_s")
    if stat("group_size"):
        inc, fb = stat("group_incomplete"), stat("group_fallback")
        print(f"  group size {q(stat('group_size'))}; incomplete groups (a sibling timed out) {sum(inc):.0f}; judged alone after a group failure {sum(fb):.0f}")
        held = sum(stat("group_wait_s"))
        print(f"  finished segments held open waiting for siblings: {held / 60:.1f} segment-minutes of in-flight slots (they count toward the in-flight limit)")
    if missing or failed:
        verdict("judge", "FAIL" if len(failed) + len(missing) > len(segs) // 10 else "WARN", f"{len(missing)} unjudged, {len(failed)} fell back")
    else:
        verdict("judge", "PASS", "every segment judged by the LLM; groups have spread" if spread and sum(x == 0 for x in spread) < len(spread) else "judged, but groups are flat")

    if not run:
        print("\n(no run dir: skipping overlap and staleness)")
        return summary()
    metrics = [json.loads(l) for l in open(os.path.join(run, "monitors/file/metrics.jsonl")) if l.strip()]
    trainer = {m["step"]: m for m in metrics if m.get("producer") == "trainer" and "time/forward_backward" in m}
    orch = {m["step"]: m for m in metrics if m.get("producer") == "orch" and "time/step" in m}

    # 5. overlap ---------------------------------------------------------------
    print("\n5. overlap: sampling continues while the trainer trains")
    print("  (the orchestrator collects batch k+1 while the trainer trains on batch k, and stops dispatching new")
    print("   segments if it would get two batches ahead: TARGET_LAG = 1 in orchestrator.py; in-flight segments keep running)")
    started_during, waited = [], 0.0
    for n in sorted(trainer):
        m = trainer[n]
        end_t = m["time"]
        begin = end_t - (m["time/step"] - m.get("time/wait_for_batch", 0.0))
        running = sum(1 for x in segs if x["start"] < end_t and x["end"] > begin)
        started = sum(1 for x in segs if begin <= x["start"] < end_t)
        started_during.append(started)
        print(f"  trainer step {n}: trains {dt.datetime.fromtimestamp(begin):%H:%M:%S}-{dt.datetime.fromtimestamp(end_t):%H:%M:%S} "
              f"({end_t - begin:.0f} s, forward/backward {m['time/forward_backward']:.0f} s, weight broadcast {m.get('time/broadcast_weights', 0):.0f} s): "
              f"{running} segments sampling meanwhile, {started} newly started")
    for n in sorted(orch):
        waited += orch[n].get("time/wait_for_policy", 0.0)
        print(f"  orchestrator batch {n}: collected in {orch[n]['time/step']:.0f} s, then waited {orch[n].get('time/wait_for_policy', 0):.0f} s for the trainer")
    last = max(orch) if orch else 0
    tail_note = f" (the wait before the last batch {last} is the run ending: there is no batch {last + 1} to collect)" if last else ""
    print(f"  total wait for the trainer {waited:.0f} s{tail_note}")
    # The trainer should never sit idle for want of data, and sampling should not run so far
    # ahead that the queue ages: a step is "starved" when the trainer waited for its batch,
    # "ahead" when the batch it trains was finished before the previous step even began.
    starved = [n for n in sorted(trainer) if n > 1 and trainer[n].get("time/wait_for_batch", 0.0) > 30]
    ahead = [n for n in sorted(orch) if n > 1 and orch[n].get("off_policy/mean", 0.0) >= 2]
    print(f"  trainer steps that waited > 30 s for data: {starved or 'none'}; batches sampled two or more updates before they trained: {ahead or 'none'}")
    if not trainer:
        verdict("overlap", "WARN", "no trainer step in metrics")
    elif starved:
        verdict("overlap", "WARN", f"the trainer waited for data at steps {starved}: sampling is the bottleneck")
    elif ahead:
        verdict("overlap", "WARN", f"sampling ran ahead of training (batches {ahead} were 2+ updates old): fewer segments in flight, "
                "or a bigger batch, would keep the data fresher")
    else:
        verdict("overlap", "PASS", "sampling and training overlapped with neither side starved")

    # 6. stale -----------------------------------------------------------------
    print("\n6. stale: how old the policy behind each trained segment is")
    for n in sorted(orch):
        m = orch[n]
        print(f"  batch {n}: off-policy mean {m.get('off_policy/mean', 0):.2f} max {m.get('off_policy/max', 0):.0f} "
              f"(in flight at the update {m.get('off_policy/in_flight/max', 0):.0f}, waiting in the queue {m.get('off_policy/in_queue/max', 0):.0f}); "
              f"dropped as too stale {m.get('off_policy/dropped', 0):.0f}")
    stream = load_stream(run)
    spans = collections.Counter()
    for ep in stream:
        pol = ((ep.get("run") or {}).get("work") or {}).get("policy") or {}
        if pol:
            spans["straddled an update" if pol.get("start") != pol.get("end") else f"one policy (v{pol.get('start')})"] += 1
    print(f"  segments by the policy versions they sampled from: {dict(spans)}")
    worst = max((m.get("off_policy/max", 0) for m in orch.values()), default=0)
    dropped = sum(m.get("off_policy/dropped", 0) for m in orch.values())
    limit = int(get("max_off_policy_steps", "8"))
    verdict("stale", "PASS" if worst <= limit and not dropped else "WARN",
            f"max staleness {worst:.0f} steps (limit {limit}), {spans.get('straddled an update', 0)} segments straddled an update, {dropped:.0f} dropped")
    summary()


def summary() -> None:
    print("\nsummary")
    for section, status, why in results:
        print(f"  {status:4s} {section:8s} {why}")


if __name__ == "__main__":
    main(sys.argv[1])

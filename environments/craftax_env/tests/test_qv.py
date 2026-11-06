import asyncio
import json
from types import SimpleNamespace

from craftax_env import qv

STATE = {"kind": "boundary", "hud": "Step: 1", "map": "", "legend": "", "nearest": {}, "achievements": {},
         "locked": {"collect_wood": 1.0, "make_wood_pickaxe": 1.0}, "B": 0.0, "deaths": 0}


def judge_reply(tool, p=0.5, flaws=()):
    return {"P": {"tool": tool, "ready": 0, "safe": 2, "near": 1}, "next_achievements": [{"name": "collect_wood", "p": p}],
            "position_gain": 0, "ideal_handoff": "", "diagnosis": "", "flaws": [{"type": t, "quote": "", "fact": ""} for t in flaws]}


def node(role, content, sampled=False):
    return SimpleNamespace(message=SimpleNamespace(role=role, content=content, reasoning_content=None),
                           sampled=sampled, mask=[1] if sampled else [0])


def trace(tid, inherited=None, writes=None):
    system = "RULES" + (f"\n\n{qv.SUMMARY_HEADER}\n{inherited}" if inherited else "")
    nodes = [node("system", system), node("user", "obs"), node("assistant", "<action>NOOP</action>", True)]
    if writes:
        nodes += [node("user", "obs\n\nYour context is about to be reset."), node("assistant", f"<summary>{writes}</summary>", True)]
    return SimpleNamespace(id=tid, task=SimpleNamespace(data=SimpleNamespace(system_prompt=system)),
                           branches=[SimpleNamespace(nodes=nodes)], nodes=nodes)


def test_score_segment_hands_q_to_the_next_segment_and_stops_at_the_cost_cap(tmp_path, monkeypatch):
    later = {**STATE, "hud": "Step: 9"}
    for tid, side, extra, state in (("a", "start", {}, STATE), ("a", "end", {"next": "b"}, later),
                                    ("b", "start", {}, later), ("b", "end", {"next": None}, later)):
        (tmp_path / f"{tid}.{side}.json").write_text(json.dumps({**extra, "state": state}))
    prompts = []

    async def fake_call(client, text, model):
        prompts.append(text)
        tool = 0 if "Step: 1" in text else 1  # the game start, then a better state
        return judge_reply(tool, flaws=("false_fact",) if "SUMMARY UNDER REVIEW" in text else ()), \
            {"prompt_tokens": 1_000_000, "total_tokens": 1_000_000}

    monkeypatch.setattr(qv, "call", fake_call)
    monkeypatch.setattr(qv, "_RUN", qv._Run())
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    stop = tmp_path / "STOP"
    args = dict(state_dir=str(tmp_path), cost_cap=0.6, price_in=0.25, price_out=0.0, stop_path=str(stop), wait_seconds=1)

    r_a, info_a = asyncio.run(qv.score_segment(trace("a", writes="go chop wood"), **args))
    # start: Q = V = 0 + 3 (no summary); end: V = 4, G* = 0.5, the summary keeps 1 - 0.3
    assert info_a["start"]["Q"] == 3 and abs(info_a["end"]["Q"] - (4 + 0.5 * 0.7)) < 1e-9
    assert r_a == 1.0  # (4.35 - 3) / max(G*, 1) = 1.35, clipped
    assert len(prompts) == 2 and any("go chop wood" in p for p in prompts) and not stop.exists()

    r_b, info_b = asyncio.run(qv.score_segment(trace("b", inherited="go chop wood"), **args))
    assert len(prompts) == 3  # its start Q was handed over, not judged again
    assert info_b["start"]["Q"] == info_a["end"]["Q"] and info_b["end"]["Q"] == 4  # the game ended: Q = V
    assert stop.exists() and "cap" in stop.read_text()  # 3 calls x $0.25 passed $0.60
    assert asyncio.run(qv.score_segment(trace("c"), **args))[0] is None and len(prompts) == 3

"""The segment judge: sub-scores mix into the `judge` reward; a failure falls back to the achievement delta."""

import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest
import verifiers.v1 as vf
from crafter_env.judge import (
    CRAFTER_GUIDE,
    CrafterJudge,
    CrafterJudgeConfig,
    JudgeVerdict,
    judge_fields,
    judge_segment,
    mix_score,
    tech_tier,
    with_tier,
)


class Trace:
    def __init__(self, info):
        self.rewards, self.metrics, self.info = {}, {}, {"crafter": info}

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value * weight

    def record_metrics(self, metrics):
        self.metrics.update(metrics)

    def record_judge_call(self, **kwargs):
        self.info.setdefault("judge_calls", []).append(kwargs)


INFO = {"segment_achievements": ["collect_wood", "place_table"], "achievements": ["collect_wood", "place_table"], "summary": "Table at 3E.",
        "inventory": {"health": 9, "food": 5, "drink": 4, "energy": 7, "wood": 3}, "deaths": [], "position": [30, 31], "game_step": 400, "steps": 400, "depth": 1,
        "achievements_before": [], "inventory_before": {}, "position_before": [32, 32], "events": ["step 12: unlocked collect_wood", "step 40: placed a table at (31, 31)"],
        "cells_visited": 55, "invalid_actions": 1, "malformed_replies": 0, "world_nearby": "tree 2W, water 5N 3E, stone 9S, coal 14S 2E", "next_tier": "make_wood_pickaxe (make_wood_pickaxe): needs near a table"}


def test_prompt_fields_and_verdict():
    judge = CrafterJudge(vf.JudgeConfig(base_url="http://localhost:1/v1", api_key_var="X", model="m"))
    fields = judge_fields(INFO, "collect_diamond", 24)
    assert fields["game_step_before"] == 0 and fields["achievements_before"] == "none" and fields["inventory_before"] == "empty"
    prompt = judge.build_messages(**fields)
    for needle in (CRAFTER_GUIDE, "step 40: placed a table", 'inventory: {"wood": 3}', "coal 14S 2E", "make_wood_pickaxe", "Table at 3E.", "within 24 tiles", "segment 1 of the game, game step 0"):
        assert needle in prompt, needle
    assert judge.schema is JudgeVerdict and judge.reward_name == "crafter"
    assert mix_score(JudgeVerdict(progress=1.0, plan=0.0, consistency=0.0), (0.5, 0.3, 0.2)) == 0.5

    async def evaluate(**fields):
        assert fields["summary"] == "Table at 3E." and fields["next_tier"].startswith("make_wood_pickaxe")
        return SimpleNamespace(parsed=JudgeVerdict(progress=0.6, plan=0.4, consistency=1.0, rationale="first table"))

    judge.evaluate = evaluate
    trace = Trace(dict(INFO))
    score = asyncio.run(judge_segment(judge, trace, goal="collect_diamond", weight=0.5, mix=(0.5, 0.3, 0.2), world_radius=24))
    assert abs(score - 0.62) < 1e-9 and abs(trace.rewards["judge"] - 0.31) < 1e-9 and trace.metrics["judge_failed"] == 0.0
    assert trace.metrics["progress"] == 0.6 and trace.info["crafter"]["judge"]["rationale"] == "first table"


def test_failure_falls_back_to_the_achievement_delta(tmp_path):
    judge = CrafterJudge(vf.JudgeConfig(base_url="http://localhost:1/v1", api_key_var="X", model="m"))

    async def evaluate(**fields):
        raise RuntimeError("endpoint down")

    judge.evaluate = evaluate
    trace = Trace(dict(INFO))
    log = tmp_path / "ep.jsonl"
    score = asyncio.run(judge_segment(judge, trace, goal="collect_diamond", weight=1.0, gate=asyncio.Semaphore(2), episode_log=str(log)))
    assert score == 2 / 22 and trace.metrics["judge_failed"] == 1.0 and trace.info["crafter"]["judge"]["failed"]
    assert "endpoint down" in trace.info["crafter"]["judge"]["rationale"] and '"task": "crafter-judge"' in log.read_text()


from crafter_env.judge import BranchVerdict, CrafterGroupJudge, GroupRendezvous, GroupVerdict, group_verdicts


def branch_info(i, delta):
    return {**INFO, "node_id": f"n{i}", "parent": "p0", "segment_achievements": delta, "summary": f"notes of branch {i}", "ended_at": 0.0}


def test_group_prompt_and_verdicts():
    judge = CrafterGroupJudge(vf.JudgeConfig(base_url="http://localhost:1/v1", api_key_var="X", model="m"))
    infos = [branch_info(i, ["collect_wood"] * (i % 2)) for i in range(4)]
    prompt = judge.build_messages(infos=infos, goal="collect_diamond", world_radius=24)
    assert prompt.count("== Branch ") == 4 and "notes of branch 3" in prompt and "grading 4 branches" in prompt and CRAFTER_GUIDE in prompt

    async def complete(messages, trace=None, schema=None, parse=None):
        return SimpleNamespace(parsed=GroupVerdict(verdicts=[BranchVerdict(branch=i, progress=i / 4, plan=0.5, consistency=1.0, rationale=f"b{i}") for i in range(4)]))

    judge.complete = complete
    out, call_s, err = asyncio.run(group_verdicts(judge, infos, goal="collect_diamond", mix=(0.5, 0.3, 0.2)))
    assert err is None and [round(v["score"], 3) for v in out] == [0.35, 0.475, 0.6, 0.725] and out[2]["rationale"] == "b2"

    async def partial(messages, trace=None, schema=None, parse=None):
        return SimpleNamespace(parsed=GroupVerdict(verdicts=[BranchVerdict(branch=0, progress=1, plan=1, consistency=1)]))

    judge.complete = partial
    out, _, err = asyncio.run(group_verdicts(judge, infos, goal="collect_diamond", mix=(0.5, 0.3, 0.2)))
    assert out is None and "covers branches [0]" in err


def test_rendezvous_one_call_for_the_group_and_timeouts(tmp_path):
    calls = []

    async def call(infos):
        calls.append([i["node_id"] for i in infos])
        await asyncio.sleep(0.05)
        return [{"score": 0.1 * k, "failed": False, "rationale": i["node_id"]} for k, i in enumerate(infos)], 0.05, None, {"cost_usd": 0.04, "input_tokens": 400.0}

    async def members(rv, n, delays):
        async def one(i, delay):
            await asyncio.sleep(delay)
            return await rv.meet("p0", f"n{i}", branch_info(i, []), call)
        return await asyncio.gather(*(one(i, d) for i, d in zip(range(n), delays)))

    rv = GroupRendezvous(str(tmp_path / "g"), size=4, timeout_s=5, poll_s=0.01)
    res = asyncio.run(members(rv, 4, [0, 0.05, 0.1, 0.2]))
    assert len(calls) == 1 and sorted(calls[0]) == ["n0", "n1", "n2", "n3"]
    assert [v["rationale"] for v, _ in res] == ["n0", "n1", "n2", "n3"]
    waits = [t["group_wait_s"] for _, t in res]
    assert waits[0] > waits[3] >= 0.05 and all(t["group_size"] == 4 and t["group_incomplete"] == 0 for _, t in res)
    assert all(abs(t["_cost"]["cost_usd"] - 0.01) < 1e-12 for _, t in res)  # one call, its cost split across the four branches

    # A sibling that never arrives: the others give up after the timeout and are judged alone.
    rv = GroupRendezvous(str(tmp_path / "h"), size=4, timeout_s=0.2, poll_s=0.01)
    res = asyncio.run(members(rv, 3, [0, 0, 0]))
    assert all(v is None and t["group_incomplete"] == 1.0 for v, t in res) and len(calls) == 1

    # A failed group call: every member falls back.
    async def failing(infos):
        return None, 0.01, "boom", None

    rv = GroupRendezvous(str(tmp_path / "i"), size=2, timeout_s=5, poll_s=0.01)

    async def two():
        return await asyncio.gather(*(rv.meet("p1", f"n{i}", branch_info(i, []), failing) for i in range(2)))

    assert all(v is None for v, _ in asyncio.run(two()))


def test_thinking_tokens_are_billed_as_output():
    from crafter_env.judge import usage_cost
    from openai.types import CompletionUsage
    from verifiers.v1.types import Usage

    u = Usage.from_openai(CompletionUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1600))  # 500 hidden thinking tokens
    assert u.completion_tokens == 600 and u.reasoning_tokens == 500
    c = usage_cost(u, "gemini-3.8-flash")
    assert c["output_tokens"] == 600 and abs(c["cost_usd"] - (1000 * 0.75 + 600 * 3.75) / 1e6) < 1e-12
    plain = Usage.from_openai(CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    assert plain.completion_tokens == 5 and plain.reasoning_tokens is None
    assert usage_cost(None, "gemini-3.8-flash")["cost_usd"] == 0.0


@pytest.fixture
def capi_http(monkeypatch):
    from crafter_env import capi

    requests, replies = [], []
    capi._session.cache_clear()
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github-oauth-test-token")

    def handle(request):
        requests.append(request)
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "gpt-4.1"}]},
            )
        status, payload = replies.pop(0)
        return httpx.Response(status, json=payload)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(capi, "build_async_httpx", lambda config: httpx.AsyncClient(transport=transport))
    yield requests, replies
    capi._session.cache_clear()


def completion(verdict):
    return {
        "id": "judge-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(verdict)}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def test_capi_config_and_single_judge(capi_http):
    from crafter_env.taskset import CrafterEnvConfig

    requests, replies = capi_http
    config = CrafterEnvConfig(
        judge={
            "backend": "capi",
            "api_key_var": "COPILOT_GITHUB_TOKEN",
            "model": "gpt-4.1",
            "headers": {"Authorization": "unrelated-header-test-token"},
        }
    )
    assert config.judge.backend == "capi"
    assert CrafterEnvConfig(judge=vf.JudgeConfig(model="m")).judge.backend == "openai"
    judge = CrafterJudge(config.judge)
    replies.extend(
        [(200, completion({"progress": 0.6, "plan": 0.4, "consistency": 1.0, "rationale": "first table"}))] * 2
    )
    trace = Trace(dict(INFO))

    async def evaluate_twice():
        score = await judge_segment(judge, trace, goal="collect_diamond", weight=1.0)
        await judge.evaluate(**judge_fields(INFO, "collect_diamond"))
        return score

    assert abs(asyncio.run(evaluate_twice()) - 0.62) < 1e-9
    assert len(requests) == 3  # one authentication probe, reused by both calls
    assert requests[0].headers["Authorization"] == "Bearer github-oauth-test-token"
    assert str(requests[1].url) == "https://api.githubcopilot.com/chat/completions"
    assert requests[1].headers["Authorization"] == "Bearer github-oauth-test-token"
    assert requests[1].headers["Copilot-Integration-Id"] == "copilot-developer-cli"
    assert requests[1].headers["X-Initiator"] == "agent"
    payload = json.loads(requests[1].content)
    assert "response_format" not in payload and "Return only a JSON object" in payload["messages"][-1]["content"]
    assert "temperature" not in payload
    assert trace.metrics["judge_input_tokens"] == 100 and trace.metrics["judge_output_tokens"] == 20
    assert trace.info["judge_calls"][0]["response"].parsed.progress == 0.6
    assert "github-oauth-test-token" not in str(trace.info["judge_calls"])


def test_capi_group_judge_and_invalid_verdict_fallback(capi_http):
    requests, replies = capi_http
    config = CrafterJudgeConfig(backend="capi", api_key_var="COPILOT_GITHUB_TOKEN", model="gpt-4.1")
    verdicts = [{"branch": i, "progress": i / 2, "plan": 0.5, "consistency": 1.0} for i in range(2)]
    replies.extend(
        [
            (200, completion({"verdicts": verdicts, "comparison": "second branch progressed further"})),
            (200, completion({"progress": 5, "plan": 0.5, "consistency": 1.0})),
        ]
    )
    trace = Trace(dict(INFO))

    async def evaluate():
        out, _, error = await group_verdicts(
            CrafterGroupJudge(config),
            [branch_info(i, []) for i in range(2)],
            goal="collect_diamond",
            mix=(0.5, 0.3, 0.2),
        )
        assert error is None and out[1]["score"] > out[0]["score"]
        await judge_segment(CrafterJudge(config), trace, goal="collect_diamond", weight=1.0)

    asyncio.run(evaluate())
    assert trace.metrics["judge_failed"] == 1 and trace.rewards["judge"] == 2 / 22
    assert trace.info["judge_calls"][0]["response"].usage.completion_tokens == 20
    assert len(requests) == 3


def test_capi_refreshes_after_401_but_does_not_retry_quota(capi_http):
    requests, replies = capi_http
    config = CrafterJudgeConfig(backend="capi", api_key_var="COPILOT_GITHUB_TOKEN", model="gpt-4.1")
    replies.extend(
        [
            (401, {"error": {"message": "expired token"}}),
            (200, completion({"progress": 0.5, "plan": 0.5, "consistency": 0.5})),
            (429, {"error": {"message": "quota reached"}}),
        ]
    )
    judge = CrafterJudge(config)
    trace = Trace(dict(INFO))

    async def evaluate():
        assert (await judge.evaluate(**judge_fields(INFO, "collect_diamond"))).parsed.progress == 0.5
        await judge_segment(judge, trace, goal="collect_diamond", weight=1.0)

    asyncio.run(evaluate())
    assert len(requests) == 5 and trace.metrics["judge_failed"] == 1
    assert sum(r.url.path == "/models" for r in requests) == 2


def test_capi_does_not_use_gemini_prices():
    import math

    from crafter_env.judge import usage_cost
    from verifiers.v1.types import Usage

    usage = Usage(prompt_tokens=100, completion_tokens=20)
    assert math.isnan(usage_cost(usage, "gemini-3.8-flash", backend="capi")["cost_usd"])
    assert usage_cost(usage, "gpt-4.1", price=(1, 2, 0.1), backend="capi")["cost_usd"] == 140 / 1e6


def test_copilot_environment_token_precedes_saved_login(monkeypatch):
    from crafter_env import copilot_auth

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "explicit-test-token")
    monkeypatch.setenv("OTHER_COPILOT_TOKEN", "other-test-token")
    monkeypatch.setattr(copilot_auth, "_read_config", lambda path: pytest.fail("must not read saved credentials"))
    assert copilot_auth.github_token() == "explicit-test-token"
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", " ")
    with pytest.raises(copilot_auth.CopilotCredentialError, match="empty"):
        copilot_auth.github_token()
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "test-token\ninjected-header")
    with pytest.raises(copilot_auth.CopilotCredentialError, match="invalid"):
        copilot_auth.github_token()


@pytest.mark.parametrize("keyring_value", [None, "keyring-test-token"])
def test_saved_copilot_jsonc_and_keyring_precedence(tmp_path, monkeypatch, keyring_value):
    from crafter_env import copilot_auth

    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "unrelated-test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "unrelated-gemini-test-token")
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text("""// Copilot CLI configuration
{
    "lastLoggedInUser": {"host": "https://github.com", "login": "selected",},
    "authTokens": {
        "https://github.com:other": {"token": "other-account-test-token"},
        "https://github.com:selected": {"token": "saved-test-token"}, /* private */
    },
}
""")
    path.chmod(0o600)
    accounts = []
    monkeypatch.setattr(copilot_auth, "_keyring_token", lambda account: accounts.append(account) or keyring_value)
    assert copilot_auth.github_token() == (keyring_value or "saved-test-token")
    assert accounts == ["https://github.com:selected"]


def test_copilot_keyring_command_captures_secret(monkeypatch):
    from crafter_env import copilot_auth

    monkeypatch.setitem(sys.modules, "keyring", None)
    monkeypatch.setattr(copilot_auth.sys, "platform", "linux")
    monkeypatch.setattr(copilot_auth.shutil, "which", lambda name: "/usr/bin/secret-tool")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"keyring-test-token\n")

    monkeypatch.setattr(copilot_auth.subprocess, "run", run)
    assert copilot_auth._keyring_token("https://github.com:selected") == "keyring-test-token"
    command, kwargs = calls[0]
    assert command == [
        "/usr/bin/secret-tool",
        "lookup",
        "service",
        "copilot-cli",
        "account",
        "https://github.com:selected",
    ]
    assert kwargs["stdout"] == copilot_auth.subprocess.PIPE
    assert kwargs["stderr"] == copilot_auth.subprocess.DEVNULL
    assert "keyring-test-token" not in str(command)


def test_saved_copilot_errors_do_not_expose_config(tmp_path, monkeypatch):
    import os

    from crafter_env import copilot_auth

    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path))
    monkeypatch.setattr(copilot_auth, "_keyring_token", lambda account: None)
    path = tmp_path / "config.json"
    path.write_text('{"authTokens": "secret-config-test-token", invalid}')
    with pytest.raises(copilot_auth.CopilotCredentialError) as error:
        copilot_auth.github_token()
    assert "secret-config-test-token" not in str(error.value)
    path.write_text(
        json.dumps(
            {
                "loggedInUsers": [
                    {"host": "https://github.com", "login": "a"},
                    {"host": "https://github.com", "login": "b"},
                ]
            }
        )
    )
    with pytest.raises(copilot_auth.CopilotCredentialError, match="active"):
        copilot_auth.github_token()
    if os.name == "posix":
        path.write_text(
            json.dumps(
                {
                    "lastLoggedInUser": {"host": "https://github.com", "login": "a"},
                    "authTokens": {"https://github.com:a": {"token": "private-test-token"}},
                }
            )
        )
        path.chmod(0o644)
        with pytest.raises(copilot_auth.CopilotCredentialError, match="600"):
            copilot_auth.github_token()


def test_capi_errors_and_reflected_tokens_are_redacted(capi_http, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    requests, replies = capi_http
    config = CrafterJudgeConfig(backend="capi", api_key_var="COPILOT_GITHUB_TOKEN", model="gpt-4.1")
    replies.extend(
        [
            (403, {"error": {"message": "github-oauth-test-token"}}),
            (
                200,
                completion(
                    {
                        "progress": 0.5,
                        "plan": 0.5,
                        "consistency": 0.5,
                        "rationale": "github-oauth-test-token",
                    }
                ),
            ),
        ]
    )
    traces = [Trace(dict(INFO)), Trace(dict(INFO))]

    async def evaluate():
        for trace in traces:
            await judge_segment(CrafterJudge(config), trace, goal="collect_diamond", weight=1.0)

    asyncio.run(evaluate())
    assert len(requests) == 3
    assert all(t.metrics["judge_failed"] == 1 and t.rewards["judge"] == 2 / 22 for t in traces)
    assert "HTTP 403" in traces[0].info["crafter"]["judge"]["rationale"]
    recorded = str([t.info for t in traces]) + caplog.text
    assert "github-oauth-test-token" not in recorded
    assert "[REDACTED]" in traces[1].info["judge_calls"][0]["response"].text


def test_tier_blend():
    assert tech_tier([]) == 0.0 and tech_tier(["collect_drink", "eat_cow"]) == 0.0
    assert tech_tier(["collect_wood", "place_table"]) == 0.25 and tech_tier(["collect_diamond"]) == 1.0
    assert tech_tier(["place_furnace", "collect_coal"]) == 0.625
    v = {"score": 0.4, "failed": False}
    assert with_tier(v, {"achievements": ["place_furnace"]}, 0.0) is v
    out = with_tier(v, {"achievements": ["place_furnace"]}, 0.5)
    assert out["llm_score"] == 0.4 and out["tier"] == 0.625 and abs(out["score"] - 0.5125) < 1e-9
    # the plateaued advanced lineage now outranks the beginner's first table (job 11420870, depth 7)
    advanced = with_tier({"score": 0.38}, {"achievements": ["place_furnace", "collect_coal"]}, 0.5)["score"]
    beginner = with_tier({"score": 0.55}, {"achievements": ["collect_wood", "place_table"]}, 0.5)["score"]
    assert advanced > beginner

    judge = CrafterJudge(vf.JudgeConfig(base_url="http://localhost:1/v1", api_key_var="X", model="m"))

    async def evaluate(**fields):
        return SimpleNamespace(parsed=JudgeVerdict(progress=0.6, plan=0.4, consistency=1.0, rationale="first table"))

    judge.evaluate = evaluate
    trace = Trace({**INFO, "achievements": ["collect_wood", "place_table"]})
    score = asyncio.run(judge_segment(judge, trace, goal="collect_diamond", weight=1.0, mix=(0.5, 0.3, 0.2), tier_weight=0.5))
    assert abs(score - (0.5 * 0.62 + 0.5 * 0.25)) < 1e-9
    assert trace.metrics["judge_tier"] == 0.25 and abs(trace.metrics["judge_llm_score"] - 0.62) < 1e-9
    assert trace.info["crafter"]["judge"]["tier"] == 0.25

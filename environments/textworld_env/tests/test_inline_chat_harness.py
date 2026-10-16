import asyncio
import json
from types import SimpleNamespace

import httpx

import textworld_env.harness as harness_module
import verifiers.v1 as vf
from textworld_env.harness import InlineChatHarness, InlineChatHarnessConfig
from verifiers.v1.errors import MODEL_TRANSPORT_ERROR_EXIT_CODE


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict, finish: str | None) -> dict:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _launch(handler, prompt="What do you do?"):
    requests = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    harness_module._HTTP = httpx.AsyncClient(transport=httpx.MockTransport(record))
    harness = InlineChatHarness(InlineChatHarnessConfig(id="textworld-env"))
    data = vf.TaskData(prompt=prompt, system_prompt="You play a game.")
    result = asyncio.run(
        harness.launch(SimpleNamespace(model="m"), None, None, "http://interception/v1", "s3cret", {}, data)
    )
    return result, requests


def test_one_streamed_call_through_the_interception_endpoint():
    def ok(request):
        body = _sse(_chunk({"role": "assistant", "content": "<action>look</action>"}, None), _chunk({}, "stop"))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    result, requests = _launch(ok)
    assert result.exit_code == 0
    (request,) = requests
    assert str(request.url) == "http://interception/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer s3cret"
    body = json.loads(request.content)
    assert body["model"] == "m" and body["stream"] is True
    assert body["messages"] == [
        {"role": "system", "content": "You play a game."},
        {"role": "user", "content": "What do you do?"},
    ]


def test_context_overflow_ends_the_run_cleanly_and_other_errors_fail():
    overflow = _launch(lambda r: httpx.Response(400, json={"error": {"message": "maximum context length exceeded"}}))
    assert overflow[0].exit_code == 0
    failure = _launch(lambda r: httpx.Response(422, json={"error": {"message": "bad request"}}))
    assert failure[0].exit_code == 1


def test_stream_without_a_finish_is_a_transport_error():
    def cut(request):
        body = _sse(_chunk({"role": "assistant", "content": "<act"}, None))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    assert _launch(cut)[0].exit_code == MODEL_TRANSPORT_ERROR_EXIT_CODE

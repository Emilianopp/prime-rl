"""prime-rl's inference endpoints served from an SGLang engine.

The orchestrator and the renderer client talk to inference only through
prime-rl's own routes (``/inference/v1/generate``, ``/pause``, ``/resume``,
``/update_weights``, ...). This module serves those same routes on top of
SGLang's HTTP app, so switching ``inference.backend`` to ``sglang`` changes
nothing above the engine.

It is self-contained (SGLang, FastAPI and numpy only; no ``prime_rl``
imports) so it can run from a separate SGLang environment::

    python server.py --server-args-json args.json

All token generation goes through ``generate_sglang`` - the one place that
speaks SGLang's request format.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Same clamp as prime-rl's vLLM server: renderers read it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0

# Same GC settings as prime-rl's vLLM API server (see ``_tune_gc``).
API_SERVER_GC_THRESHOLDS = (50_000, 20, 100)

# vLLM-style sampling keys, renamed to SGLang's.
_RENAMED = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
    "seed": "sampling_seed",
    "include_stop_str_in_output": "no_stop_trim",
}
_SAME = {
    "temperature",
    "top_p",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
}
# Keys with no SGLang sampling meaning: logprobs and token ids are always
# returned, and the rest are client-side or vLLM response-shaping knobs.
_IGNORED = {
    "logprobs",
    "return_token_ids",
    "routed_experts_prompt_start",
    "detokenize",
    "flat_logprobs",
    "reasoning_effort",
}


class GenerateRequestError(ValueError):
    """A request the SGLang backend cannot serve faithfully."""


def translate_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
    """vLLM ``SamplingParams`` keys -> SGLang ``SamplingParams`` keys.

    Anything not translated is refused rather than dropped: a silently ignored
    sampling knob would make rollout logprobs disagree with the trainer.
    """
    out: dict[str, Any] = {}
    unsupported = []
    for key, value in params.items():
        if value is None or key in _IGNORED:
            continue
        if key in _SAME:
            out[key] = value
        elif key in _RENAMED:
            out[_RENAMED[key]] = value
        elif key == "top_k":
            # vLLM disables top-k with -1 or 0; SGLang with -1.
            out["top_k"] = -1 if value in (-1, 0) else int(value)
        elif key == "n":
            if value != 1:
                unsupported.append("n != 1")
        else:
            unsupported.append(key)
    if unsupported:
        raise GenerateRequestError(f"unsupported sampling parameters for the SGLang backend: {sorted(unsupported)}")
    return out


def encode_array(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {
        "data": base64.b64encode(memoryview(array)).decode("ascii"),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def pack_logprobs(output_token_logprobs: list) -> np.ndarray:
    """SGLang's ``[(logprob, token_id, text), ...]`` -> float32, clamped like vLLM."""
    values = np.array(
        [LOGPROB_SENTINEL if entry[0] is None else entry[0] for entry in output_token_logprobs],
        dtype=np.float32,
    )
    return np.maximum(values, LOGPROB_SENTINEL)


def build_generate_response(ret: dict[str, Any], prompt_token_ids: list[int]) -> dict[str, Any]:
    """An SGLang ``/generate`` result -> prime-rl's ``/inference/v1/generate`` response."""
    meta = ret["meta_info"]
    finish = meta.get("finish_reason") or {}
    completion_ids = list(ret.get("output_ids") or [])
    logprobs = pack_logprobs(meta.get("output_token_logprobs") or [])
    if len(logprobs) != len(completion_ids):
        raise RuntimeError(
            f"SGLang returned {len(logprobs)} logprobs for {len(completion_ids)} completion tokens"
        )
    prompt_tokens = int(meta.get("prompt_tokens", len(prompt_token_ids)))
    completion_tokens = int(meta.get("completion_tokens", len(completion_ids)))
    return {
        "request_id": meta.get("id", ""),
        "prompt_token_ids": list(prompt_token_ids),
        "choices": [
            {
                "index": 0,
                "token_ids": completion_ids,
                "completion_logprobs": encode_array(logprobs),
                "finish_reason": finish.get("type"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": int(meta.get("cached_tokens", 0))},
        },
    }


async def generate_sglang(
    tokenizer_manager: Any,
    token_ids: list[int],
    sampling_params: dict[str, Any],
    *,
    cache_salt: str | None = None,
    priority: int | None = None,
) -> dict[str, Any]:
    """Token-in / token-out generation on SGLang; the single SGLang call site.

    Always returns prime-rl's generate response: the prompt token ids the engine
    ran, the completion token ids, and one sampled-token logprob per completion
    token. Later steps add an optional session and ``kv_eviction`` here, so every
    mode keeps this one path and one response shape.
    """
    from sglang.srt.managers.io_struct import GenerateReqInput

    obj = GenerateReqInput(
        input_ids=list(token_ids),
        sampling_params=translate_sampling_params(sampling_params),
        return_logprob=True,
        logprob_start_len=-1,
        top_logprobs_num=0,
        stream=False,
        cache_salt=cache_salt,
        priority=priority,
    )
    # No client handle: prime-rl's dispatcher owns cancellation, not HTTP disconnects.
    ret = await tokenizer_manager.generate_request(obj, None).__anext__()
    finish = ret["meta_info"].get("finish_reason") or {}
    if finish.get("type") == "abort":
        raise GenerateRequestError(finish.get("message") or "request aborted by the SGLang scheduler")
    return build_generate_response(ret, token_ids)


def render_load_metrics(loads: list[Any]) -> str:
    """SGLang per-scheduler load snapshots -> the ``vllm:*`` Prometheus gauges that
    prime-rl's concurrency controller reads (KV usage and capacity, running and
    waiting requests). Without them the controller never grows past one in-flight
    episode."""
    families = {
        "vllm:kv_cache_usage_perc": lambda load: float(load.token_usage),
        "vllm:num_requests_running": lambda load: float(load.num_running_reqs),
        "vllm:num_requests_waiting": lambda load: float(load.num_waiting_reqs),
    }
    lines = []
    for name, value in families.items():
        lines.append(f"# TYPE {name} gauge")
        lines += [f'{name}{{engine="{load.dp_rank}"}} {value(load)}' for load in loads]
    lines.append("# TYPE vllm:cache_config_info gauge")
    lines += [
        f'vllm:cache_config_info{{engine="{load.dp_rank}",kv_cache_size_tokens="{int(load.max_total_num_tokens)}"}} 1.0'
        for load in loads
    ]
    return "\n".join(lines) + "\n"


def _tokenizer_manager():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.tokenizer_manager


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": status_code}}, status_code=status_code)


@router.post("/inference/v1/generate")
async def inference_generate(request: Request):
    body = await request.json()
    if body.get("stream"):
        return _error("streaming is not supported on the SGLang backend", 400)
    if body.get("content_parts") or body.get("features"):
        return _error("multimodal inputs are not supported on the SGLang backend", 400)
    try:
        return await generate_sglang(
            _tokenizer_manager(),
            body["token_ids"],
            body.get("sampling_params") or {},
            cache_salt=body.get("cache_salt"),
            priority=body.get("priority"),
        )
    except (GenerateRequestError, ValueError) as e:
        # The tokenizer manager raises ValueError for requests the scheduler rejects (400).
        return _error(str(e), 400)


@router.get("/metrics")
async def metrics(request: Request):
    """Engine load in vLLM's metric names (SGLang's own ``--enable-metrics`` uses ``sglang:*``)."""
    loads = await _tokenizer_manager().get_loads()
    return PlainTextResponse(render_load_metrics(loads), media_type="text/plain; version=0.0.4")


@router.post("/pause")
async def pause(request: Request):
    """Freeze generation in place: in-flight requests keep their KV and continue after /resume."""
    from sglang.srt.managers.io_struct import PauseGenerationReqInput

    await _tokenizer_manager().pause_generation(PauseGenerationReqInput(mode="in_place"))
    return {"status": "paused"}


@router.post("/resume")
async def resume(request: Request):
    from sglang.srt.managers.io_struct import ContinueGenerationReqInput

    await _tokenizer_manager().continue_generation(ContinueGenerationReqInput())
    return {"status": "resumed"}


@router.post("/update_weights")
async def update_weights(request: Request):
    """Load a filesystem checkpoint in place, keeping the KV cache (no flush)."""
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

    data = await request.json()
    weight_dir = data.get("weight_dir")
    if not weight_dir:
        return _error("update_weights requires weight_dir (filesystem weight broadcast)", 400)
    success, message, _ = await _tokenizer_manager().update_weights_from_disk(
        UpdateWeightFromDiskReqInput(model_path=weight_dir, flush_cache=False)
    )
    if not success:
        return _error(f"weight update failed: {message}", 500)
    return {"status": "ok"}


@router.get("/liveness")
async def liveness(request: Request):
    """Up and not shutting down. Unlike ``is_ready()`` this stays OK while paused for a weight update."""
    from sglang.srt.managers.tokenizer_manager import ServerStatus

    tokenizer_manager = _tokenizer_manager()
    if tokenizer_manager.gracefully_exit or tokenizer_manager.server_status != ServerStatus.Up:
        return JSONResponse({"status": "engine_unresponsive"}, status_code=503)
    return {"status": "ok"}


@router.post("/finish_session")
async def finish_session(request: Request):
    """The router releases sticky sessions; with no router there is nothing to release."""
    return {"status": "ok"}


def _tune_gc() -> None:
    """Under RL load this process holds a few objects per generated token (SGLang's
    logprob tuples), and default thresholds rerun full collections over them every
    few seconds. Move everything alive after startup out of GC and collect less often."""
    gc.collect()
    gc.freeze()
    gc.set_threshold(*API_SERVER_GC_THRESHOLDS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="prime-rl endpoints on an SGLang engine")
    parser.add_argument("--server-args-json", required=True, help="JSON file of SGLang ServerArgs fields")
    args = parser.parse_args(argv)
    with open(args.server_args_json) as f:
        server_args_dict = json.load(f)

    from sglang.srt.entrypoints.http_server import app, launch_server
    from sglang.srt.server_args import ServerArgs

    app.include_router(router)
    launch_server(ServerArgs(**server_args_dict), launch_callback=_tune_gc)  # after warmup


if __name__ == "__main__":
    main()

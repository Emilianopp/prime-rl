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
import asyncio
import base64
import gc
import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

try:
    from . import model_evictions
    from .eviction import CallView, EvictionPlan, TurnWindow, merge, subtract, survivor
except ImportError:  # run as a script from the SGLang environment: siblings are on sys.path
    import model_evictions
    from eviction import CallView, EvictionPlan, TurnWindow, merge, subtract, survivor

logger = logging.getLogger(__name__)

router = APIRouter()

# SGLang session and kv_eviction call ids allow only these characters
# (``^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$``).
_SESSION_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]")

# verifiers sends the trace id in this header (verifiers.v1.clients.client.SESSION_ID_HEADER).
SESSION_ID_HEADER = "X-Session-ID"

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


def build_generate_response(
    ret: dict[str, Any], prompt_token_ids: list[int] | None, prompt_len: int | None = None
) -> dict[str, Any]:
    """An SGLang ``/generate`` result -> prime-rl's ``/inference/v1/generate`` response.

    ``prompt_token_ids=None`` (a session delta request, whose client holds the prompt)
    leaves them out of the response; ``prompt_len`` is then the prompt's length."""
    meta = ret["meta_info"]
    finish = meta.get("finish_reason") or {}
    completion_ids = list(ret.get("output_ids") or [])
    logprobs = pack_logprobs(meta.get("output_token_logprobs") or [])
    if len(logprobs) != len(completion_ids):
        raise RuntimeError(
            f"SGLang returned {len(logprobs)} logprobs for {len(completion_ids)} completion tokens"
        )
    if prompt_len is None:
        prompt_len = len(prompt_token_ids or ())
    prompt_tokens = int(meta.get("prompt_tokens", prompt_len))
    completion_tokens = int(meta.get("completion_tokens", len(completion_ids)))
    return {
        "request_id": meta.get("id", ""),
        **({"prompt_token_ids": list(prompt_token_ids)} if prompt_token_ids is not None else {}),
        "choices": [
            {
                "index": 0,
                "token_ids": completion_ids,
                "completion_logprobs": encode_array(logprobs),
                "finish_reason": finish.get("type"),
                **({"eviction": ret["eviction"]} if ret.get("eviction") else {}),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": int(meta.get("cached_tokens", 0))},
        },
    }


class SglangEngine:
    """The few SGLang calls this module makes, over the tokenizer manager.

    Kept behind one small class so the session logic can be tested with a fake.
    """

    def __init__(self, tokenizer_manager: Any):
        self.tokenizer_manager = tokenizer_manager

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        request: Request | None = None,
        session_id: str | None = None,
        cache_salt: str | None = None,
        priority: int | None = None,
        routed_dp_rank: int | None = None,
        kv_eviction: dict[str, Any] | None = None,
        rid: str | None = None,
    ) -> dict[str, Any]:
        from sglang.srt.managers.io_struct import GenerateReqInput

        obj = GenerateReqInput(
            rid=rid,
            input_ids=list(input_ids),
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
            stream=False,
            cache_salt=cache_salt,
            priority=priority,
            session_params={"id": session_id} if session_id is not None else None,
            routed_dp_rank=routed_dp_rank,
            kv_eviction=kv_eviction,
        )
        # The HTTP request lets SGLang abort the generation when its client disconnects
        # (a cancelled episode); without it the request runs on to max_tokens.
        return await self.tokenizer_manager.generate_request(obj, request).__anext__()

    async def open_session(self, session_id: str, timeout: float) -> bool:
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        opened = await self.tokenizer_manager.open_session(
            OpenSessionReqInput(capacity_of_str_len=1 << 30, session_id=session_id, streaming=True, timeout=timeout)
        )
        return opened is not None

    async def close_session(self, session_id: str) -> None:
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        await self.tokenizer_manager.close_session(CloseSessionReqInput(session_id=session_id))

    def abort_request(self, rid: str) -> None:
        """Abort a running or queued request; its caller gets an ``abort`` finish."""
        self.tokenizer_manager.abort_request(rid)


def _aborted(ret: dict[str, Any]) -> bool:
    return (ret["meta_info"].get("finish_reason") or {}).get("type") == "abort"


@dataclass
class _CallEviction:
    request: dict[str, Any]
    """The call's ``kv_eviction`` object."""
    dead: list[list[int]]
    """The session's dead spans after the call, chronological."""
    spans: list[list[int]]
    """What the call evicts, chronological: the record for the trainer's mask."""


@dataclass
class _TraceSession:
    engine_session: str | None = None
    # Prompt + completion of the last call, chronological: what the engine
    # session holds, apart from the evicted tokens.
    resident: list[int] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # KV-Streams (eviction on): the session's call chain and what it evicted.
    finish_reason: Any = None  # of the last call, to explain a later reopen
    inflight_rid: str | None = None  # the session request in flight, for abort on finish
    finished: bool = False
    call_index: int = 0
    state_id: str | None = None
    protected: int = 0
    # Dead spans of the session (chronological ``[start, end)``, sorted, disjoint).
    dead: list[list[int]] = field(default_factory=list)
    # Positions just after each message end in ``resident`` (eviction on), kept
    # incrementally so a turn scans only its new tokens.
    ends: list[int] = field(default_factory=list)
    # Each eviction plan's own state for this session.
    plan_states: list[dict[str, Any]] = field(default_factory=list)


class SessionDesyncError(Exception):
    """A delta request (``prefix_len``) that does not match its session: the client
    resends the full prompt."""


class SessionTable:
    """One SGLang streaming session per trace (keyed by ``X-Session-ID``).

    A request whose prompt extends the session's resident tokens sends only the
    new tail; the session already holds the rest as KV, so it is never
    re-prefilled or evicted by the radix cache. Any other prompt (a rewritten
    history, e.g. a re-prefill compaction branch) reopens the session on the full
    prompt. A request arriving while its trace's session is busy (a concurrent
    branch) runs stateless. The response is identical either way.

    With an ``eviction`` schedule (KV-Streams), each appended turn also evicts
    the turns the schedule has retired from the session's KV, in place: the
    survivors keep their positions and are never recomputed. The response still
    carries the full chronological prompt; the model saw only the live turns.

    A client that tracks its session can send just the new tokens with
    ``prefix_len`` = the session's resident length (``generate(..., prefix_len=)``):
    then nothing per call scales with the trace's length - no full prompt to
    receive, compare or echo back. A ``prefix_len`` the session does not hold
    raises ``SessionDesyncError`` (HTTP 409) and the client resends in full. A
    response that ran on the session is marked ``"session": true``, and a client
    sends deltas only after one, so a server without sessions (vLLM) never gets one.
    """

    def __init__(
        self,
        engine: SglangEngine | None,
        timeout: float,
        eviction: TurnWindow | None = None,
        overlap_schedule: bool = False,
        plans: list[EvictionPlan] = (),
    ):
        self.engine = engine
        self.timeout = timeout
        # The turn window gives the session's message structure (the protected first
        # message) and is the first eviction plan; ``plans`` add to it.
        self.eviction = eviction
        self.plans: list[EvictionPlan] = [eviction, *plans] if eviction is not None else []
        # With overlap scheduling SGLang computes a call's last sampled token's KV before
        # the next call (while the evicted tokens are still resident); without it, that
        # token's KV is computed by the next call, after the eviction.
        self.overlap_schedule = overlap_schedule
        self._traces: dict[str, _TraceSession] = {}
        self._incarnation = itertools.count()
        self._request = itertools.count()
        self.stats = {
            "opened": 0,
            "reopened": 0,
            "evicting_calls": 0,
            "evicted_tokens": 0,
            "failed_calls": 0,
            "resurrected": 0,
        }

    async def generate(
        self,
        trace_id: str,
        token_ids: list[int],
        sampling_params: dict[str, Any],
        prefix_len: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate on the trace's session. ``token_ids`` is the full prompt, or with
        ``prefix_len > 0`` only what follows the session's ``prefix_len`` resident tokens."""
        trace = self._traces.setdefault(trace_id, _TraceSession())
        if trace.lock.locked():
            if prefix_len:
                raise SessionDesyncError(f"trace {trace_id} has a call in flight")
            return await self.engine.generate(token_ids, sampling_params, **kwargs)
        async with trace.lock:
            if prefix_len and (trace.engine_session is None or len(trace.resident) != prefix_len or not token_ids):
                raise SessionDesyncError(
                    f"trace {trace_id}: delta on {prefix_len} tokens, session holds {len(trace.resident)}"
                )
            for attempt in range(2):
                resident = trace.resident
                held = len(resident)
                if prefix_len and trace.engine_session is not None:
                    extends, tail = True, token_ids
                else:
                    if prefix_len:  # a delta whose session was closed by a failed attempt
                        token_ids, prefix_len = prior + token_ids, 0
                    extends = (
                        trace.engine_session is not None
                        and len(token_ids) > held
                        and token_ids[:held] == resident
                    )
                    tail = token_ids[held:] if extends else token_ids
                prior = resident  # _close() swaps in a fresh list: this one stays intact
                if not extends:
                    self._note_resurrection(trace_id, trace)
                    reopen = trace.engine_session is not None or attempt > 0
                    if trace.engine_session is not None and attempt == 0:
                        self._log_rewrite(trace_id, trace, token_ids)
                    await self._close(trace)
                    engine_session = f"{_SESSION_UNSAFE.sub('-', trace_id)[:200]}.{next(self._incarnation)}"
                    if not await self.engine.open_session(engine_session, self.timeout):
                        logger.warning("could not open an SGLang session for trace %s; running stateless", trace_id)
                        return await self.engine.generate(token_ids, sampling_params, **kwargs)
                    trace.engine_session = engine_session
                    self.stats["reopened" if reopen else "opened"] += 1
                    if trace.finished:
                        # finish() ran while the session opened, so it had nothing to abort or
                        # close: close it here and generate nothing.
                        await self._close(trace)
                        raise GenerateRequestError(f"trace {trace_id} finished while its session opened")
                    held = 0
                prompt_ends: list[int] = []
                call_kwargs, plan = dict(kwargs), None
                if self.eviction is not None:
                    prompt_ends = (trace.ends if extends else []) + self.eviction.message_ends(tail, held)
                    plan = self._eviction_call(trace, prompt_ends, tail)
                    call_kwargs["kv_eviction"] = plan.request
                    # SGLang requires a kv_eviction request's rid to be its call id.
                    call_kwargs["rid"] = call_kwargs["kv_eviction"]["call_id"]
                else:
                    call_kwargs["rid"] = f"{trace.engine_session}.r{next(self._request)}"
                trace.inflight_rid = call_kwargs["rid"]
                try:
                    ret = await self.engine.generate(
                        tail, sampling_params, session_id=trace.engine_session, **call_kwargs
                    )
                except BaseException:  # e.g. SGLang aborted it on a client disconnect
                    self._note_resurrection(trace_id, trace)
                    await self._close(trace)
                    raise
                finally:
                    trace.inflight_rid = None
                if trace.finished:
                    # Aborted by finish(): the trace is gone, so never reopen it.
                    return ret
                if _aborted(ret):
                    # The engine session's contents are no longer known: start over.
                    # An eviction call is retried once on a fresh session.
                    self._note_resurrection(trace_id, trace)
                    await self._close(trace)
                    if self.eviction is not None:
                        self.stats["failed_calls"] += 1
                        message = (ret["meta_info"].get("finish_reason") or {}).get("message")
                        logger.warning("KV-Streams call on trace %s aborted: %s", trace_id, message)
                        if attempt == 0:
                            continue
                    return ret
                output = list(ret.get("output_ids") or [])
                if extends:
                    trace.resident.extend(tail)  # in place: no per-turn copy of the trace
                else:
                    trace.resident = list(tail)
                prompt_len = len(trace.resident)
                trace.resident.extend(output)
                trace.finish_reason = ret["meta_info"].get("finish_reason")
                if plan is not None:
                    trace.ends = prompt_ends + self.eviction.message_ends(output, prompt_len)
                    info = ret["meta_info"].get("kv_eviction") or {}
                    trace.state_id = (info.get("cache_state") or {}).get("state_id")
                    trace.call_index += 1
                    if plan.spans:
                        self.stats["evicting_calls"] += 1
                        self.stats["evicted_tokens"] += sum(end - start for start, end in plan.spans)
                        # For the trainer, in chronological positions: the evicted tokens
                        # are unseen from the first token whose KV is computed after this
                        # eviction - the previous call's last sampled token, or (with
                        # overlap scheduling, which computes that one early) the new tail.
                        ret["eviction"] = {"spans": plan.spans, "at": held - (0 if self.overlap_schedule else 1)}
                    trace.dead = plan.dead
                ret["prompt_len"] = prompt_len
                return ret

    def _note_resurrection(self, trace_id: str, trace: _TraceSession) -> None:
        """A fresh session holds the whole prompt again, evicted turns included, so
        this trace's earlier evictions no longer describe what later calls attend
        to and its training mask will be off. Expected to be rare; counted."""
        evicted = sum(end - start for start, end in trace.dead)
        if evicted:
            self.stats["resurrected"] += 1
            logger.warning("trace %s reopened after evicting %d tokens: they are attended again", trace_id, evicted)

    @staticmethod
    def _log_rewrite(trace_id: str, trace: _TraceSession, token_ids: list[int]) -> None:
        """Why a trace's prompt stopped extending its session: where it diverges."""
        resident = trace.resident
        at = next((i for i, (a, b) in enumerate(zip(resident, token_ids)) if a != b), min(len(resident), len(token_ids)))
        logger.warning(
            "session reopen: trace %s prompt %d tokens vs session %d, diverges at %d: session %s prompt %s "
            "(last finish %s)",
            trace_id,
            len(token_ids),
            len(resident),
            at,
            resident[max(0, at - 4) : at + 4],
            token_ids[max(0, at - 4) : at + 4],
            trace.finish_reason,
        )

    def _eviction_call(self, trace: _TraceSession, prompt_ends: list[int], tail: list[int]) -> _CallEviction:
        """This call's ``kv_eviction`` object and the dead spans it leaves.

        Every plan names the chronological spans that should be dead from this call
        on; what is not dead yet dies now. The engine takes spans in the session's
        resident (survivor) coordinates, where every dead token before a span shifts
        it down, so adjacent survivor spans (a retired turn around a span that died
        earlier) join into one.
        """
        cache_id = trace.engine_session
        if trace.call_index == 0:
            trace.protected = self.eviction.layout_ends(prompt_ends)[0]
            trace.plan_states = [{} for _ in self.plans]
            # The first call claims the fresh session and cannot evict.
            request = {
                "version": 1,
                "cache_id": cache_id,
                "call_id": f"{cache_id}.c0",
                "call_index": 0,
                "expected_state_id": None,
                "protected_prefix_len": trace.protected,
                "evict_spans": [],
            }
            return _CallEviction(request, dead=[], spans=[])
        # The last sampled token has no KV yet, so only the resident sequence
        # before it can be evicted; a span that ends later waits for the next call.
        limit = len(trace.resident) - 1
        wanted = []
        for plan, state in zip(self.plans, trace.plan_states):
            view = CallView(trace.resident + tail, prompt_ends, trace.protected, limit, trace.dead, state)
            wanted += [[max(start, trace.protected), min(end, limit)] for start, end in plan.evict(view)]
        spans = subtract(merge([span for span in wanted if span[0] < span[1]]), trace.dead)
        evict_spans: list[list[int]] = []
        for chrono_start, chrono_end in spans:
            span = [survivor(chrono_start, trace.dead), survivor(chrono_end, trace.dead)]
            if evict_spans and evict_spans[-1][1] == span[0]:
                evict_spans[-1][1] = span[1]
            else:
                evict_spans.append(span)
        request = {
            "version": 1,
            "cache_id": cache_id,
            "call_id": f"{cache_id}.c{trace.call_index}",
            "call_index": trace.call_index,
            "expected_state_id": trace.state_id,
            "protected_prefix_len": trace.protected,
            "evict_spans": evict_spans,
        }
        return _CallEviction(request, dead=merge(trace.dead + spans), spans=spans)

    async def finish(self, trace_id: str) -> None:
        """Release a finished or cancelled trace: abort its in-flight request, then
        close its session. SGLang defers a close until the session's in-flight
        request ends, and a request queued on a full engine may never end; a
        cancelled episode's partial trace is discarded, so aborting loses nothing."""
        trace = self._traces.pop(trace_id, None)
        if trace is None:
            return
        trace.finished = True
        if trace.inflight_rid is not None:
            self.engine.abort_request(trace.inflight_rid)
        await self._close(trace)

    async def _close(self, trace: _TraceSession) -> None:
        engine_session, trace.engine_session = trace.engine_session, None
        if engine_session is not None:
            # SGLang defers the close until an in-flight request on it finishes.
            await self.engine.close_session(engine_session)
        trace.resident, trace.ends, trace.dead, trace.plan_states = [], [], [], []
        trace.call_index, trace.state_id, trace.protected = 0, None, 0


class TraceAffinity:
    """Pin each trace to one data-parallel rank, the least loaded (fewest live
    traces) at its first request: the in-server analogue of vllm-router's
    ``sticky_least_loaded``. A trace's prefix - radix cache or streaming session -
    stays on the rank that holds it. SGLang sends session open/close to every
    rank, so pinning the generate requests is enough."""

    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self._ranks: dict[str, int] = {}
        self._live = [0] * dp_size

    def rank(self, trace_id: str | None) -> int | None:
        if self.dp_size <= 1 or trace_id is None:
            return None  # SGLang's own load balancing
        rank = self._ranks.get(trace_id)
        if rank is None:
            rank = min(range(self.dp_size), key=self._live.__getitem__)
            self._ranks[trace_id] = rank
            self._live[rank] += 1
        return rank

    def release(self, trace_id: str) -> None:
        rank = self._ranks.pop(trace_id, None)
        if rank is not None:
            self._live[rank] -= 1


async def generate_sglang(
    engine: SglangEngine,
    token_ids: list[int],
    sampling_params: dict[str, Any],
    *,
    request: Request | None = None,
    sessions: SessionTable | None = None,
    affinity: TraceAffinity | None = None,
    trace_id: str | None = None,
    cache_salt: str | None = None,
    priority: int | None = None,
    prefix_len: int = 0,
) -> dict[str, Any]:
    """Token-in / token-out generation on SGLang; the single SGLang call site.

    Returns prime-rl's generate response: the full prompt token ids, the completion
    token ids, and one sampled-token logprob per completion token - the same with
    or without a session. A session delta request (``prefix_len > 0``: ``token_ids``
    is only what follows the session's resident tokens) gets no prompt token ids
    back; its client already holds them.
    """
    params = translate_sampling_params(sampling_params)
    kwargs = {"cache_salt": cache_salt, "priority": priority, "request": request}
    if affinity is not None:
        kwargs["routed_dp_rank"] = affinity.rank(trace_id)
    if prefix_len and (sessions is None or trace_id is None):
        raise SessionDesyncError("a delta request (prefix_len) needs a session")
    if sessions is not None and trace_id is not None:
        ret = await sessions.generate(trace_id, list(token_ids), params, prefix_len=prefix_len, **kwargs)
    else:
        ret = await engine.generate(token_ids, params, **kwargs)
    if _aborted(ret):
        finish = ret["meta_info"].get("finish_reason") or {}
        raise GenerateRequestError(finish.get("message") or "request aborted by the SGLang scheduler")
    if prefix_len:
        response = build_generate_response(ret, None, prompt_len=ret.get("prompt_len"))
    else:
        response = build_generate_response(ret, token_ids)
    if "prompt_len" in ret:
        # Ran on the trace's session: the client may send the next turn as a delta.
        response["session"] = True
    return response


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


# Set in main(): the engine adapter, the per-trace session table (sessions on)
# and the trace -> DP rank pinning (dp_size > 1).
_STATE: dict[str, Any] = {"engine": None, "sessions": None, "affinity": None}


def _tokenizer_manager():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.tokenizer_manager


def _engine() -> SglangEngine:
    if _STATE["engine"] is None:
        _STATE["engine"] = SglangEngine(_tokenizer_manager())
    return _STATE["engine"]


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": status_code}}, status_code=status_code)


@router.post("/inference/v1/generate")
async def inference_generate(request: Request):
    body = await request.json()
    if body.get("stream"):
        return _error("streaming is not supported on the SGLang backend", 400)
    if body.get("content_parts") or body.get("features"):
        return _error("multimodal inputs are not supported on the SGLang backend", 400)
    sessions = _STATE["sessions"]
    if sessions is not None and sessions.engine is None:
        sessions.engine = _engine()
    try:
        return await generate_sglang(
            _engine(),
            body["token_ids"],
            body.get("sampling_params") or {},
            sessions=sessions,
            affinity=_STATE["affinity"],
            trace_id=request.headers.get(SESSION_ID_HEADER),
            cache_salt=body.get("cache_salt"),
            priority=body.get("priority"),
            prefix_len=int(body.get("prefix_len") or 0),
            request=request,
        )
    except SessionDesyncError as e:
        # The client resends in full right away; tell OpenAI-style clients not to retry.
        response = JSONResponse(
            {
                "error": {
                    "message": str(e),
                    "code": 409,
                    "type": "session_desync",
                }
            },
            status_code=409,
            headers={"x-should-retry": "false"},
        )
        return response
    except (GenerateRequestError, ValueError) as e:
        # The tokenizer manager raises ValueError for requests the scheduler rejects (400).
        return _error(str(e), 400)


@router.get("/metrics")
async def metrics(request: Request):
    """Engine load in vLLM's metric names (SGLang's own ``--enable-metrics`` uses ``sglang:*``)."""
    loads = await _tokenizer_manager().get_loads()
    body = render_load_metrics(loads)
    sessions = _STATE["sessions"]
    if sessions is not None:
        body += "".join(f"# TYPE kvs:{name}_total counter\nkvs:{name}_total {value}\n" for name, value in sessions.stats.items())
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


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
    """The orchestrator releases a finished trace: close its SGLang session and DP pin, if any."""
    sessions, affinity = _STATE["sessions"], _STATE["affinity"]
    session_id = request.query_params.get("session_id")
    if sessions is not None and session_id:
        await sessions.finish(session_id)
    if affinity is not None and session_id:
        affinity.release(session_id)
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
    parser.add_argument(
        "--server-args-json",
        required=True,
        help='JSON file: {"server_args": {SGLang ServerArgs fields}, "sessions": bool, "session_timeout": seconds, '
        '"eviction": {"max_turns", "stride", "message_end_id", "model_eviction"?: {"name", "ids"}} or null}',
    )
    args = parser.parse_args(argv)
    with open(args.server_args_json) as f:
        launch = json.load(f)

    from sglang.srt.entrypoints.http_server import app, launch_server
    from sglang.srt.server_args import ServerArgs

    if launch.get("sessions"):
        # The engine binds lazily: the tokenizer manager exists only once the server starts.
        schedule = dict(launch.get("eviction") or {})
        model = schedule.pop("model_eviction", None)
        _STATE["sessions"] = SessionTable(
            engine=None,
            timeout=float(launch.get("session_timeout", 3600.0)),
            eviction=TurnWindow(**schedule) if schedule else None,
            overlap_schedule=not launch["server_args"].get("disable_overlap_schedule", False),
            plans=[model_evictions.build(model["name"], model["ids"])] if model else [],
        )
    dp_size = int(launch["server_args"].get("dp_size") or 1)
    if dp_size > 1:
        _STATE["affinity"] = TraceAffinity(dp_size)
    app.include_router(router)

    def started() -> None:  # after warmup, once SGLang has configured logging
        if _STATE.get("sessions") is not None:
            plans = [f"{type(p).__module__.rsplit('.', 1)[-1]}.{type(p).__name__}" for p in _STATE["sessions"].plans]
            logger.info("KV-Streams eviction plans: %s", ", ".join(plans) or "none")
        _tune_gc()

    launch_server(ServerArgs(**launch["server_args"]), launch_callback=started)


if __name__ == "__main__":
    main()

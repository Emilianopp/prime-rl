"""Copilot CAPI transport for Crafter's judges.

Set env.judge.backend = "capi" and env.judge.model to an available model.
COPILOT_GITHUB_TOKEN takes precedence over the active CLI login saved in the OS
keyring or private config.json. Authentication is noninteractive in workers.

CAPI is undocumented. Copilot entitlement, organization policies and quotas
apply; Copilot usage is not a Gemini-style per-token USD price.
"""

from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from typing import Any, Literal

import httpx
import verifiers.v1 as vf
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import BaseModel, ValidationError
from verifiers.v1.clients.base import MAX_RETRIES, build_async_httpx
from verifiers.v1.configs.client import resolve_headers
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.judge import JudgeResponse
from verifiers.v1.types import SamplingConfig, Usage
from verifiers.v1.utils.generic import merge_defaults

from crafter_env.copilot_auth import github_token

API_URL = "https://api.githubcopilot.com"
INTEGRATION_ID = "copilot-developer-cli"


class CAPIError(RuntimeError):
    """Sanitized provider failure safe for judge logs and fallback traces."""


def _http_failure(stage: str, status: int) -> CAPIError:
    hints = {
        401: "Run copilot login again, or unset an expired COPILOT_GITHUB_TOKEN.",
        403: "Check Copilot entitlement, organization policy and OAuth client eligibility.",
        404: "Check that the selected model is available through this CAPI endpoint.",
        429: "Copilot quota or rate limit reached; retry later.",
    }
    return CAPIError(
        f"CAPI {stage} failed (HTTP {status}). {hints.get(status, 'Retry later; the endpoint may have changed.')}"
    )


class CrafterJudgeConfig(vf.JudgeConfig):
    backend: Literal["openai", "capi"] = "openai"
    """openai includes Gemini's compatible endpoint; capi uses
    COPILOT_GITHUB_TOKEN or the saved CLI login directly against the Copilot API."""


@lru_cache(maxsize=8)
def _session(github_token: str) -> dict[str, Any]:
    return {}


async def _credentials(config: vf.JudgeConfig, *, refresh: bool = False) -> dict[str, Any]:
    token_source = github_token()
    session = _session(token_source)
    if refresh or time.time() >= session.get("validated_until", 0):
        try:
            async with build_async_httpx(config) as http:
                response = await http.get(
                    f"{API_URL}/models",
                    headers={
                        "Authorization": f"Bearer {token_source}",
                        "Accept": "application/json",
                        "Copilot-Integration-Id": INTEGRATION_ID,
                        "User-Agent": "crafter-capi-judge",
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
            if not isinstance(data.get("data"), list):
                raise ValueError
        except httpx.HTTPStatusError as exc:
            raise _http_failure("authentication", exc.response.status_code) from None
        except httpx.RequestError:
            raise CAPIError(
                "CAPI authentication network request failed; check connectivity and TLS settings."
            ) from None
        except (KeyError, TypeError, ValueError, AttributeError):
            raise CAPIError("Invalid CAPI model-list response; the undocumented endpoint may have changed.") from None
        session.update(
            token=token_source, github_token=token_source, validated_until=time.time() + 300, base_url=API_URL
        )
    return session


async def capi_complete(
    config: vf.JudgeConfig,
    name: str,
    messages: Any,
    *,
    trace: Any = None,
    schema: type[BaseModel] | None = None,
    parse: Any = None,
    **sampling: Any,
) -> JudgeResponse:
    wire = (
        [{"role": "user", "content": messages}] if isinstance(messages, str) else [message_to_wire(m) for m in messages]
    )
    if schema is not None:
        wire = [
            *wire,
            {
                "role": "user",
                "content": "Return only a JSON object matching this schema, without Markdown:\n"
                + json.dumps(schema.model_json_schema()),
            },
        ]
    options = {key: sampling.pop(key) for key in ("extra_headers", "extra_query", "timeout") if key in sampling}
    if options.get("extra_headers") is not None:
        options["extra_headers"] = {
            key: value
            for key, value in options["extra_headers"].items()
            if key.lower() not in ("authorization", "proxy-authorization")
        }
    params = SamplingConfig.model_validate(merge_defaults(config.sampling, sampling)).model_dump(exclude_none=True)
    sdk_fields = CompletionCreateParamsNonStreaming.__annotations__
    kwargs = {
        "model": config.model,
        "messages": wire,
        **{key: value for key, value in params.items() if key in sdk_fields},
        "extra_body": {key: value for key, value in params.items() if key not in sdk_fields},
        **options,
    }
    response = None
    try:
        for attempt in range(2):
            session = await _credentials(config, refresh=bool(attempt))
            headers = {
                **{
                    key: value
                    for key, value in resolve_headers(config.model_copy(update={"base_url": API_URL})).items()
                    if key.lower() not in ("authorization", "proxy-authorization")
                },
                "Copilot-Integration-Id": INTEGRATION_ID,
                "Openai-Intent": "conversation-panel",
                "X-Initiator": "agent",
                "X-Request-Id": str(uuid.uuid4()),
            }
            try:
                async with AsyncOpenAI(
                    base_url=session["base_url"],
                    api_key=session["token"],
                    default_headers=headers,
                    max_retries=MAX_RETRIES,
                    http_client=build_async_httpx(config),
                ) as client:
                    completion = await client.chat.completions.create(**kwargs)
                break
            except AuthenticationError:
                if attempt:
                    raise _http_failure("judging", 401) from None
            except APIStatusError as exc:
                raise _http_failure("judging", exc.status_code) from None
            except APIConnectionError:
                raise CAPIError("CAPI judging network request failed; check connectivity and TLS settings.") from None
        choice = completion.choices[0]
        text = choice.message.content or ""
        safe_text = text
        for secret in (session["token"], session["github_token"]):
            safe_text = safe_text.replace(secret, "[REDACTED]")
        response = JudgeResponse(text=safe_text, usage=Usage.from_openai(completion.usage))
        if safe_text != text:
            raise CAPIError("CAPI returned credential material; it was redacted and the verdict rejected.")
        if choice.message.refusal:
            raise CAPIError("CAPI judge refused the verdict.")
        if choice.finish_reason == "length":
            raise RuntimeError("judge output was truncated")
        if schema is not None:
            try:
                response.parsed = schema.model_validate_json(response.text)
            except ValidationError:
                raise CAPIError("CAPI judge returned an invalid JSON verdict.") from None
        if parse is not None:
            response.parsed = parse(response)
        return response
    finally:
        if trace is not None and response is not None:
            trace.record_judge_call(name=name, request={"model": config.model, "messages": wire}, response=response)

import math

import numpy as np
import pytest
from renderers.client import _parse_completion_logprobs

from prime_rl.inference.sglang.server import (
    LOGPROB_SENTINEL,
    GenerateRequestError,
    build_generate_response,
    pack_logprobs,
    translate_sampling_params,
)


def test_translate_renames_and_keeps_vllm_keys():
    out = translate_sampling_params(
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "min_p": 0.0,
            "max_tokens": 128,
            "min_tokens": 1,
            "seed": 7,
            "stop_token_ids": [2],
            "skip_special_tokens": False,
            "repetition_penalty": 1.1,
        }
    )
    assert out == {
        "temperature": 0.7,
        "top_p": 0.9,
        "min_p": 0.0,
        "max_new_tokens": 128,
        "min_new_tokens": 1,
        "sampling_seed": 7,
        "stop_token_ids": [2],
        "skip_special_tokens": False,
        "repetition_penalty": 1.1,
    }


@pytest.mark.parametrize("top_k, expected", [(-1, -1), (0, -1), (20, 20)])
def test_translate_top_k_disable_sentinels(top_k, expected):
    assert translate_sampling_params({"top_k": top_k}) == {"top_k": expected}


def test_translate_drops_response_shaping_keys_and_nones():
    params = {"logprobs": 1, "return_token_ids": True, "reasoning_effort": "low", "temperature": None}
    assert translate_sampling_params(params) == {}


@pytest.mark.parametrize("params", [{"logit_bias": {"1": 2.0}}, {"n": 2}, {"guided_json": "{}"}])
def test_translate_refuses_unsupported_keys(params):
    with pytest.raises(GenerateRequestError):
        translate_sampling_params(params)


def test_pack_logprobs_clamps_missing_and_infinite():
    packed = pack_logprobs([(-0.5, 1, None), (None, 2, None), (-math.inf, 3, None)])
    assert packed.dtype == np.float32
    assert packed.tolist() == [-0.5, LOGPROB_SENTINEL, LOGPROB_SENTINEL]


def _sglang_result(output_ids, logprobs, finish="stop"):
    return {
        "output_ids": output_ids,
        "meta_info": {
            "id": "rid-1",
            "finish_reason": {"type": finish},
            "prompt_tokens": 3,
            "completion_tokens": len(output_ids),
            "cached_tokens": 2,
            "output_token_logprobs": [(lp, t, None) for lp, t in zip(logprobs, output_ids)],
        },
    }


def test_response_matches_the_renderers_parser():
    response = build_generate_response(_sglang_result([5, 6], [-0.25, -1.5]), [1, 2, 3])
    assert response["prompt_token_ids"] == [1, 2, 3]
    choice = response["choices"][0]
    assert choice["token_ids"] == [5, 6]
    assert choice["finish_reason"] == "stop"
    # The renderer client decodes exactly this packed payload on the vLLM path.
    assert _parse_completion_logprobs(choice, choice["token_ids"]) == pytest.approx([-0.25, -1.5])
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }


def test_response_rejects_logprob_count_mismatch():
    result = _sglang_result([5, 6], [-0.25, -1.5])
    result["meta_info"]["output_token_logprobs"] = result["meta_info"]["output_token_logprobs"][:1]
    with pytest.raises(RuntimeError):
        build_generate_response(result, [1, 2, 3])

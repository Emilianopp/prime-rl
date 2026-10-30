import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.sglang.launcher import eviction_launch_args, sglang_server_args


def test_sglang_backend_drops_the_default_router():
    config = InferenceConfig(backend="sglang")
    assert config.router is None
    assert InferenceConfig().router is not None  # vLLM default unchanged


def test_server_args_translate_engine_fields_and_apply_sglang_extras():
    config = InferenceConfig(
        backend="sglang",
        vllm={"model": "Qwen/Qwen3-4B", "max_model_len": 4096, "enable_prefix_caching": False},
        sglang={"page-size": 1, "tp_size": 1, "enable_streaming_session": "true"},
    )
    args = sglang_server_args(config)
    assert args["model_path"] == "Qwen/Qwen3-4B"
    assert args["context_length"] == 4096
    assert args["disable_radix_cache"] is True
    assert args["enable_fp32_lm_head"] is True
    assert args["page_size"] == 1  # kebab-case normalized
    assert args["enable_streaming_session"] is True  # CLI-style string JSON-parsed
    assert "mem_fraction_static" not in args  # only carried over when set explicitly
    assert "python" not in args


def test_detokenization_is_off_unless_turned_back_on():
    assert sglang_server_args(InferenceConfig(backend="sglang"))["skip_tokenizer_init"] is True
    config = InferenceConfig(backend="sglang", sglang={"skip_tokenizer_init": False})
    assert sglang_server_args(config)["skip_tokenizer_init"] is False


def test_explicit_memory_fraction_is_carried_over():
    config = InferenceConfig(backend="sglang", vllm={"gpu_memory_utilization": 0.5})
    assert sglang_server_args(config)["mem_fraction_static"] == 0.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"router": {"type": "vllm-router"}},
        {"weight_broadcast": {"type": "nccl"}},
        {"vllm": {"enable_lora": True}},
        {"enable_return_sampling_mask": True},
    ],
)
def test_unsupported_features_are_refused(overrides):
    with pytest.raises(ValidationError, match="does not support"):
        InferenceConfig(backend="sglang", **overrides)


def test_sessions_enable_streaming_sessions_and_stay_out_of_server_args():
    config = InferenceConfig(backend="sglang", sglang={"sessions": True, "session_timeout": 120})
    args = sglang_server_args(config)
    assert args["enable_streaming_session"] is True
    assert "sessions" not in args and "session_timeout" not in args


def test_data_parallel_size_maps_to_dp_size():
    config = InferenceConfig(backend="sglang", vllm={"data_parallel_size": 2})
    assert sglang_server_args(config)["dp_size"] == 2


def test_eviction_enables_kv_eviction_and_resolves_the_message_end_token():
    config = InferenceConfig(backend="sglang", sglang={"sessions": True, "eviction": {"max_turns": 8, "stride": 1}})
    args = sglang_server_args(config)
    assert args["enable_kv_eviction"] is True and args["page_size"] == 1
    assert "eviction" not in args
    ids = {"<|im_end|>": 151645}
    assert eviction_launch_args(config, token_id=ids.get, model_type="qwen3") == {
        "max_turns": 8,
        "stride": 1,
        "message_end_id": 151645,
    }
    assert eviction_launch_args(InferenceConfig(backend="sglang")) is None


def test_eviction_requires_sessions():
    with pytest.raises(ValidationError, match="requires inference.sglang.sessions"):
        InferenceConfig(backend="sglang", sglang={"eviction": {"max_turns": 8}})


def test_the_models_eviction_plan_comes_with_eviction_by_model_type():
    config = InferenceConfig(backend="sglang", sglang={"sessions": True, "eviction": {"max_turns": 8, "stride": 1}})
    vocab = {"<think>\n\n</think>\n\n": [248068, 271, 248069, 271], "<|im_end|>": [248046]}
    vocab |= {"<|im_start|>user\n": [248045, 846, 198], "<tool_response>": [248066]}
    end = {"<|im_end|>": 248046}.get
    args = eviction_launch_args(config, token_id=end, encode=vocab.__getitem__, model_type="qwen3_5")
    assert args["model_eviction"] == {
        "name": "qwen3.5",
        "ids": {
            "empty_think_ids": [248068, 271, 248069, 271],
            "user_ids": [248045, 846, 198],
            "tool_response_ids": [248066],
        },
    }
    # A model without a plan gets the turn window alone.
    assert "model_eviction" not in eviction_launch_args(config, token_id=end, model_type="llama")
    assert "model_eviction" not in sglang_server_args(config)


def test_model_plan_false_keeps_the_turn_window_alone():
    config = InferenceConfig(
        backend="sglang", sglang={"sessions": True, "eviction": {"max_turns": 8, "stride": 1, "model_plan": False}}
    )
    args = eviction_launch_args(config, token_id={"<|im_end|>": 248046}.get, model_type="qwen3_5")
    assert args == {"max_turns": 8, "stride": 1, "message_end_id": 248046}

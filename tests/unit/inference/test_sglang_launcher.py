import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.sglang.launcher import sglang_server_args


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
        {"vllm": {"data_parallel_size": 2}},
    ],
)
def test_unsupported_features_are_refused(overrides):
    with pytest.raises(ValidationError, match="does not support"):
        InferenceConfig(backend="sglang", **overrides)

"""Start prime-rl's inference endpoints on an SGLang engine (``inference.backend = "sglang"``)."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig

SERVER_SCRIPT = Path(__file__).with_name("server.py")

# [inference.sglang] fields prime-rl reads itself; everything else is a ServerArgs field.
PRIME_RL_FIELDS = {"python", "sessions", "session_timeout"}


def sglang_server_args(config: InferenceConfig) -> dict[str, Any]:
    """Translate the engine-agnostic ``[inference.vllm]`` fields to SGLang ``ServerArgs``
    names, then apply ``[inference.sglang]`` verbatim (it wins on conflicts)."""
    vllm = config.vllm
    args: dict[str, Any] = {
        "model_path": vllm.model,
        "host": config.server.host or "0.0.0.0",
        "port": config.server.port,
        "tp_size": vllm.tensor_parallel_size,
        "log_level": config.log.level.lower(),
        # Token-in/token-out: like the vLLM server's detokenize=False, but engine-wide
        # in SGLang, so stop strings and min_tokens need skip_tokenizer_init = false.
        "skip_tokenizer_init": True,
    }
    if vllm.max_model_len is not None:
        args["context_length"] = vllm.max_model_len
    # vLLM's and SGLang's memory fractions are not the same quantity, so only an
    # explicit setting is carried over; otherwise SGLang picks its own default.
    if "gpu_memory_utilization" in vllm.model_fields_set:
        args["mem_fraction_static"] = vllm.gpu_memory_utilization
    if vllm.enable_prefix_caching is False:
        args["disable_radix_cache"] = True
    if config.enable_fp32_lm_head:
        args["enable_fp32_lm_head"] = True
    if config.sglang.sessions:
        args["enable_streaming_session"] = True
    args.update(config.sglang.model_dump(exclude=PRIME_RL_FIELDS))
    return args


def server(config: InferenceConfig) -> None:
    """Replace this process with the SGLang server (so signals reach SGLang directly)."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    args_path = config.output_dir / "sglang_server_args.json"
    launch = {
        "server_args": sglang_server_args(config),
        "sessions": config.sglang.sessions,
        "session_timeout": config.sglang.session_timeout,
    }
    args_path.write_text(json.dumps(launch, indent=2))
    python = config.sglang.python or sys.executable
    os.execv(python, [python, str(SERVER_SCRIPT), "--server-args-json", str(args_path)])

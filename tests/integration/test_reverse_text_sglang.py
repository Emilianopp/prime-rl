"""reverse-text RL served by the SGLang backend.

Same run and thresholds as ``test_reverse_text.py`` (the vLLM baseline), so a
pass means the backend swap is behavior-preserving: rewards rise to the same
range and the trainer/inference mismatch KL stays in the vLLM band.
Requires the KV-Streams SGLang fork in the environment, or set
``PRIME_SGLANG_PYTHON`` to an SGLang venv's interpreter (forwarded as
``--inference.sglang.python``). ``PRIME_RL_TEST_EXTRA_ARGS`` appends further
``rl`` arguments.
"""

import os
import shlex
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_avg_mismatch_kl_in_range,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "reverse-text-sglang"
TIMEOUT = 1200


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse-text-sglang/start.toml",
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"test-reverse-text-sglang:{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    if sglang_python := os.environ.get("PRIME_SGLANG_PYTHON"):
        cmd += ["--inference.sglang.python", sglang_python]
    # Cluster-specific overrides, e.g. a free ZMQ port where a node service holds 5555.
    cmd += shlex.split(os.environ.get("PRIME_RL_TEST_EXTRA_ARGS", ""))
    return run_process(cmd, timeout=TIMEOUT)


def _log(run_dir: Path, name: str) -> list[str]:
    with open(run_dir / "logs" / "latest" / f"{name}.log") as f:
        return strip_escape_codes(f.read()).splitlines()


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, run_dir: Path):
    check_no_error(rl_process, run_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, run_dir: Path):
    check_reward_goes_up(_log(run_dir, "orchestrator"))


def test_reward_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    check_reward_in_range(_log(run_dir, "orchestrator"), min_threshold=0.6)


def test_mismatch_kl_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    check_avg_mismatch_kl_in_range(_log(run_dir, "trainer"), last_n_steps=3, max_threshold=0.01)


def test_weight_updates_keep_the_kv_cache(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """In-place weight updates must not flush the cache (in-flight requests keep their KV)."""
    inference = "\n".join(_log(run_dir, "inference"))
    assert "Cache flushed successfully" not in inference

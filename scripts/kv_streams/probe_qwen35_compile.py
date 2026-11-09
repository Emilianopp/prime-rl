"""Does a Qwen3.5 decoder stack compile under the trainer's activation checkpointing?

Builds a few real-size Qwen3.5 layers (3 linear-attention + 1 full-attention by
default), wraps them exactly as the trainer does (apply_ac -> apply_compile), and
runs forward + backward on a packed row with an optional kv_death mask. Reports
dynamo graph breaks, peak memory, step time, and the gradients' agreement with an
uncompiled run of the same weights and inputs.

    python probe_qwen35_compile.py --config <hf config.json> --tokens 16384 [--mask]
"""

import argparse
import copy
from contextlib import nullcontext
import json
import time

import torch
from torch import nn

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig
from prime_rl.trainer.model import apply_ac, apply_compile
from prime_rl.trainer.models.layers import kv_death
from prime_rl.trainer.models.layers.kv_death import build_kv_death_block_mask
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from prime_rl.utils.act_offloading import OffloadActivations


class Wrapper(nn.Module):
    """``get_language_model`` finds the layers at ``.model``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


def run(model, inputs, seq_lens, block_mask, probe, steps, offload=False):
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with OffloadActivations() if offload else nullcontext():
            hidden = model(inputs_embeds=inputs, seq_lens=seq_lens, block_mask=block_mask).last_hidden_state
        loss = (hidden.float() * probe).sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    grads = {name: p.grad.float().clone() for name, p in model.named_parameters() if p.grad is not None}
    return loss.item(), grads, times, torch.cuda.max_memory_allocated() / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--mask", action="store_true", help="kv_death block mask on the full-attention layer")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--offload", action="store_true", help="run under activation offloading, as the trainer does")
    ap.add_argument("--no-fp32", action="store_true", help="skip the fp32 reference")
    ap.add_argument("--save-grads", help="torch.save the eager gradients here (compare across code versions)")
    ap.add_argument("--compare-grads", help="a --save-grads file: report the compiled gradients' difference to it")
    ap.add_argument("--dynamic-output-shapes", action="store_true", help="let dynamo trace data-dependent shapes")
    args = ap.parse_args()
    if args.dynamic_output_shapes:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

    raw = json.load(open(args.config))
    config = Qwen3_5TextConfig(**raw.get("text_config", raw))
    config.num_hidden_layers = args.layers
    config.layer_types = config.layer_types[: args.layers]
    torch.manual_seed(0)
    with torch.device("cuda"):
        eager = Qwen3_5Model(config).to(torch.bfloat16)
        for p in eager.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
    compiled = copy.deepcopy(eager)
    wrapper = Wrapper(compiled)
    apply_ac(wrapper, ActivationCheckpointConfig())
    apply_compile(wrapper, CompileConfig())
    eager_wrapper = Wrapper(eager)
    apply_ac(eager_wrapper, ActivationCheckpointConfig())

    total = args.tokens
    seq_lens = torch.tensor([total // 3, total - total // 3], device="cuda")
    inputs = torch.randn(1, total, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    probe = torch.randn(1, total, config.hidden_size, device="cuda")
    block_mask = None
    if args.mask:
        # Evict every token older than 4k positions, like a turn window.
        positions = torch.cat([torch.arange(int(n), device="cuda") for n in seq_lens])
        death = torch.where(positions + 4096 < seq_lens.repeat_interleave(seq_lens), positions + 4096, -1)
        block_mask = build_kv_death_block_mask(death, seq_lens)

    torch._dynamo.utils.counters.clear()
    loss_c, grads_c, times_c, mem_c = run(compiled, inputs, seq_lens, block_mask, probe, args.steps, args.offload)
    breaks = dict(torch._dynamo.utils.counters["graph_break"])
    loss_e, grads_e, times_e, mem_e = run(eager, inputs, seq_lens, block_mask, probe, args.steps, args.offload)

    worst = max(
        ((g - grads_e[n]).norm() / grads_e[n].norm().clamp(min=1e-12)).item() for n, g in grads_c.items()
    )
    # Rounding check: both bf16 runs against an fp32 copy of the same weights and inputs.
    # Compiled is as accurate as eager if its error to fp32 is no larger than eager's.
    reference = {}
    if args.no_fp32:
        grads_x = None
    exact = None if args.no_fp32 else copy.deepcopy(eager).float()
    if exact is not None:
        kv_death._KERNEL_OPTIONS.clear()  # its flex tiles are sized for bf16 and overflow shared memory in fp32
        _, grads_x, _, _ = run(exact, inputs.float(), seq_lens, block_mask, probe, 1)

    def errors(grads):
        rel = {n: ((g - grads_x[n]).norm() / grads_x[n].norm().clamp(min=1e-12)).item() for n, g in grads.items()}
        ordered = sorted(rel.values())
        return {"median": ordered[len(ordered) // 2], "max": ordered[-1], "worst": max(rel, key=rel.get)}

    if grads_x is not None:
        reference["grad_error_vs_fp32"] = {"compiled": errors(grads_c), "eager": errors(grads_e)}
    if args.save_grads:
        torch.save(grads_e, args.save_grads)
    if args.compare_grads:
        saved = torch.load(args.compare_grads)
        reference["worst_grad_rel_diff_vs_saved"] = max(
            ((g - saved[n].cuda()).norm() / saved[n].cuda().norm().clamp(min=1e-12)).item() for n, g in grads_c.items()
        )
    print(json.dumps(reference | {
        "tokens": total,
        "mask": args.mask,
        "graph_breaks": sum(breaks.values()),
        "break_reasons": list(breaks)[:5],
        "loss": {"compiled": loss_c, "eager": loss_e},
        "worst_grad_rel_diff": worst,
        "step_s": {"compiled": min(times_c), "eager": min(times_e)},
        "peak_gib": {"compiled": round(mem_c, 2), "eager": round(mem_e, 2)},
    }, indent=1))


if __name__ == "__main__":
    main()

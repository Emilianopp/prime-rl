"""FLA kernels as torch custom ops, so the layers around them compile.

FLA wraps its causal convolution and gated delta rule in ``torch.compiler.disable``
(and their varlen setup reads data-dependent shapes); its gated RMS norm's backward
queries the device in a way dynamo cannot trace. A graph break inside an
activation-checkpointed block makes dynamo give up on the whole block, so every
decoder layer then runs eagerly, forward and recompute alike. As custom ops the
FLA calls are opaque nodes in the graph and the rest of the block compiles.

Each op calls FLA's own autograd.Function forward and backward. The backward op
re-runs the forward first to rebuild what FLA saves for backward instead of
returning it from the forward op: the forward is cheap next to the backward, and
under activation checkpointing the forward runs again anyway.
"""

import torch
from fla.modules.conv.triton import CausalConv1dFunction
from fla.modules.fused_norm_gate import LayerNormGatedFunction
from fla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction
from torch import Tensor


class _Ctx:
    """Stands in for autograd's ctx to call an autograd.Function's forward and backward directly."""

    def save_for_backward(self, *tensors: Tensor | None) -> None:
        self.saved_tensors = tensors


def _conv_forward(ctx: _Ctx, x: Tensor, weight: Tensor, cu_seqlens: Tensor, activation: str) -> Tensor:
    y, _ = CausalConv1dFunction.forward(
        ctx, x, weight, None, None, None, False, activation, cu_seqlens
    )
    return y


@torch.library.custom_op("prime_rl::fla_causal_conv1d", mutates_args=())
def causal_conv1d(x: Tensor, weight: Tensor, cu_seqlens: Tensor, activation: str) -> Tensor:
    """FLA's varlen causal depthwise convolution, ``x`` [1, T, D], ``weight`` [D, W], no bias."""
    return _conv_forward(_Ctx(), x, weight, cu_seqlens, activation)


@causal_conv1d.register_fake
def _(x, weight, cu_seqlens, activation):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::fla_causal_conv1d_backward", mutates_args=())
def causal_conv1d_backward(
    dy: Tensor, x: Tensor, weight: Tensor, cu_seqlens: Tensor, activation: str
) -> tuple[Tensor, Tensor]:
    ctx = _Ctx()
    _conv_forward(ctx, x, weight, cu_seqlens, activation)
    dx, dweight = CausalConv1dFunction.backward(ctx, dy)[:2]
    return dx, dweight


@causal_conv1d_backward.register_fake
def _(dy, x, weight, cu_seqlens, activation):
    return torch.empty_like(x), torch.empty_like(weight)


def _conv_setup(ctx, inputs, output):
    x, weight, cu_seqlens, activation = inputs
    ctx.save_for_backward(x, weight, cu_seqlens)
    ctx.activation = activation


def _conv_backward(ctx, dy):
    x, weight, cu_seqlens = ctx.saved_tensors
    dx, dweight = causal_conv1d_backward(dy, x, weight, cu_seqlens, ctx.activation)
    return dx, dweight, None, None


causal_conv1d.register_autograd(_conv_backward, setup_context=_conv_setup)


def _gdn_forward(ctx: _Ctx, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor) -> Tensor:
    o, _ = ChunkGatedDeltaRuleFunction.forward(
        ctx, q, k, v, g, beta, k.shape[-1] ** -0.5, None, False,
        cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
    )
    return o


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule", mutates_args=())
def chunk_gated_delta_rule(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor) -> Tensor:
    """FLA's varlen chunked gated delta rule with in-kernel q/k L2 norm and the default scale:
    q/k [1, T, H, K], v [1, T, HV, V], g (log decay) and beta [1, T, HV]. Returns o like v."""
    return _gdn_forward(_Ctx(), q, k, v, g, beta, cu_seqlens)


@chunk_gated_delta_rule.register_fake
def _(q, k, v, g, beta, cu_seqlens):
    return v.new_empty(v.shape, dtype=q.dtype)


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule_backward", mutates_args=())
def chunk_gated_delta_rule_backward(
    do: Tensor, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ctx = _Ctx()
    _gdn_forward(ctx, q, k, v, g, beta, cu_seqlens)
    dq, dk, dv, dg, dbeta = ChunkGatedDeltaRuleFunction.backward(ctx, do, None)[:5]
    return dq, dk, dv, dg, dbeta


@chunk_gated_delta_rule_backward.register_fake
def _(do, q, k, v, g, beta, cu_seqlens):
    return tuple(torch.empty_like(t) for t in (q, k, v, g, beta))


def _gdn_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _gdn_backward(ctx, do):
    return *chunk_gated_delta_rule_backward(do, *ctx.saved_tensors), None


chunk_gated_delta_rule.register_autograd(_gdn_backward, setup_context=_gdn_setup)


def _norm_forward(ctx: _Ctx, x: Tensor, g: Tensor, weight: Tensor, activation: str, eps: float) -> Tensor:
    return LayerNormGatedFunction.forward(ctx, x, g, weight, None, activation, None, eps, False, False, True)


@torch.library.custom_op("prime_rl::fla_rms_norm_gated", mutates_args=())
def rms_norm_gated(x: Tensor, g: Tensor, weight: Tensor, activation: str, eps: float) -> Tensor:
    """FLA's ``FusedRMSNormGated``: RMS norm of ``x`` over its last dim, scaled by ``weight``,
    times ``activation(g)``."""
    return _norm_forward(_Ctx(), x, g, weight, activation, eps)


@rms_norm_gated.register_fake
def _(x, g, weight, activation, eps):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::fla_rms_norm_gated_backward", mutates_args=())
def rms_norm_gated_backward(
    dy: Tensor, x: Tensor, g: Tensor, weight: Tensor, activation: str, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    ctx = _Ctx()
    _norm_forward(ctx, x, g, weight, activation, eps)
    dx, dg, dweight = LayerNormGatedFunction.backward(ctx, dy)[:3]
    return dx, dg, dweight


@rms_norm_gated_backward.register_fake
def _(dy, x, g, weight, activation, eps):
    return torch.empty_like(x), torch.empty_like(g), torch.empty_like(weight)


def _norm_setup(ctx, inputs, output):
    x, g, weight, activation, eps = inputs
    ctx.save_for_backward(x, g, weight)
    ctx.activation, ctx.eps = activation, eps


def _norm_backward(ctx, dy):
    return *rms_norm_gated_backward(dy, *ctx.saved_tensors, ctx.activation, ctx.eps), None, None


rms_norm_gated.register_autograd(_norm_backward, setup_context=_norm_setup)

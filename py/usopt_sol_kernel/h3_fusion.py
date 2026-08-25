"""Bit-exact MiniMax H3 modulation kernels for ComfyUI's eager BF16 path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_TILE = 1024


@triton.jit
def _round_bf16_fp32(value):
    """Round FP32 to BF16 precision while retaining an FP32 Triton value.

    A cast down and immediately back up can be optimized away. Performing the
    round-to-nearest-even operation on the bits creates the same precision
    barrier as Comfy's separate eager BF16 kernels.
    """
    bits = value.to(tl.int32, bitcast=True)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _modulate_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    row_index_ptr,
    tokens,
    width,
    sx_t,
    sx_d,
    ss_t,
    ss_d,
    sh_t,
    sh_d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < tokens) & (cols < width)
    table_row = tl.load(row_index_ptr + row)

    x = tl.load(x_ptr + row * sx_t + cols * sx_d, mask=valid).to(tl.float32)
    scale = tl.load(
        scale_ptr + table_row * ss_t + cols * ss_d,
        mask=valid,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + table_row * sh_t + cols * sh_d,
        mask=valid,
    ).to(tl.float32)

    # Match: x.mul_(1.0 + scale.to(bf16)).add_(shift.to(bf16)).
    scale = _round_bf16_fp32(scale)
    shift = _round_bf16_fp32(shift)
    one_plus_scale = _round_bf16_fp32(1.0 + scale)
    multiplied = _round_bf16_fp32(x * one_plus_scale)
    result = _round_bf16_fp32(multiplied + shift)
    tl.store(x_ptr + row * sx_t + cols * sx_d, result.to(tl.bfloat16), mask=valid)


@triton.jit
def _gate_add_kernel(
    residual_ptr,
    gate_ptr,
    branch_ptr,
    row_index_ptr,
    tokens,
    width,
    sr_t,
    sr_d,
    sg_t,
    sg_d,
    sb_t,
    sb_d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < tokens) & (cols < width)
    table_row = tl.load(row_index_ptr + row)

    residual = tl.load(
        residual_ptr + row * sr_t + cols * sr_d,
        mask=valid,
    ).to(tl.float32)
    branch = tl.load(
        branch_ptr + row * sb_t + cols * sb_d,
        mask=valid,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + table_row * sg_t + cols * sg_d,
        mask=valid,
    ).to(tl.float32)

    # addcmul_ performs the multiply/add in opmath precision and rounds once
    # when its BF16 output is stored.
    gate = _round_bf16_fp32(gate)
    result = residual + branch * gate
    tl.store(
        residual_ptr + row * sr_t + cols * sr_d,
        result.to(tl.bfloat16),
        mask=valid,
    )


def _validate_activation(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [tokens, hidden]")
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA bfloat16 tensor")
    if x.stride(1) != 1:
        raise ValueError(f"{name}'s hidden dimension must be contiguous")


def _validate_table(table: torch.Tensor, x: torch.Tensor, name: str) -> None:
    if table.ndim != 2 or table.shape[1] != x.shape[1]:
        raise ValueError(f"{name} must have shape [rows, {x.shape[1]}]")
    if table.device != x.device or table.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be CUDA bfloat16/float32 on x.device")
    if table.stride(1) != 1:
        raise ValueError(f"{name}'s hidden dimension must be contiguous")


def _validate_index(row_index: torch.Tensor, x: torch.Tensor, rows: int) -> None:
    if (
        row_index.ndim != 1
        or row_index.shape[0] != x.shape[0]
        or row_index.dtype != torch.int32
        or row_index.device != x.device
        or not row_index.is_contiguous()
    ):
        raise TypeError("row_index must be contiguous CUDA int32 with one entry per token")
    if rows <= 0:
        raise ValueError("modulation table must contain at least one row")


def fused_modulate_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_index: torch.Tensor,
) -> torch.Tensor:
    """In-place equivalent of Comfy H3's segmented mul_ then add_."""
    _validate_activation(x, "x")
    _validate_table(scale, x, "scale")
    _validate_table(shift, x, "shift")
    _validate_index(row_index, x, min(scale.shape[0], shift.shape[0]))
    if scale.shape[0] != shift.shape[0]:
        raise ValueError("scale and shift must have the same row count")

    grid = (x.shape[0], triton.cdiv(x.shape[1], _TILE))
    _modulate_kernel[grid](
        x,
        scale,
        shift,
        row_index,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        BLOCK=_TILE,
        num_warps=4,
    )
    return x


def fused_gate_add_(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    row_index: torch.Tensor,
) -> torch.Tensor:
    """In-place equivalent of Comfy H3's segmented addcmul_."""
    _validate_activation(residual, "residual")
    _validate_activation(branch, "branch")
    if residual.shape != branch.shape or residual.device != branch.device:
        raise ValueError("residual and branch must share shape and device")
    _validate_table(gate, residual, "gate")
    _validate_index(row_index, residual, gate.shape[0])

    grid = (residual.shape[0], triton.cdiv(residual.shape[1], _TILE))
    _gate_add_kernel[grid](
        residual,
        gate,
        branch,
        row_index,
        residual.shape[0],
        residual.shape[1],
        residual.stride(0),
        residual.stride(1),
        gate.stride(0),
        gate.stride(1),
        branch.stride(0),
        branch.stride(1),
        BLOCK=_TILE,
        num_warps=4,
    )
    return residual


def make_segment_index(tokens: int, segments, device: torch.device) -> torch.Tensor:
    """Build the tiny, request-static row lookup reused by every H3 block."""
    tokens = int(tokens)
    host = torch.empty(tokens, dtype=torch.int32)
    cursor = 0
    for start, stop, row in segments:
        start, stop, row = int(start), int(stop), int(row)
        if start != cursor or stop < start or stop > tokens or row < 0:
            raise ValueError("modulation segments must be ordered, contiguous, and in range")
        host[start:stop] = row
        cursor = stop
    if cursor != tokens:
        raise ValueError("modulation segments must cover every packed token")
    return host.to(device=device)


__all__ = ["fused_modulate_", "fused_gate_add_", "make_segment_index"]

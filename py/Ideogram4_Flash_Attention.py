import logging
import weakref

import torch

from comfy.patcher_extension import CallbacksMP


IDEOGRAM4_BLOCK_COUNT = 34


def _normalize_mask(mask, batch_size, sequence_length):
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim != 3:
        raise ValueError("Expected an attention mask shaped Bx1xLxL or BxLxL.")

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)

    if mask.shape != (batch_size, sequence_length, sequence_length):
        raise ValueError("Attention mask shape does not match the QKV sequence.")

    if mask.dtype == torch.bool:
        return mask

    return mask >= 0


def _build_varlen_plan(mask, batch_size, sequence_length):
    allowed = _normalize_mask(mask, batch_size, sequence_length)
    groups = []

    for batch_index in range(batch_size):
        valid_tokens = allowed[batch_index, -1]
        expected = valid_tokens.unsqueeze(0) == valid_tokens.unsqueeze(1)
        if not torch.equal(allowed[batch_index], expected):
            raise ValueError("Mask is not an Ideogram-style two-segment block mask.")

        valid_indices = torch.nonzero(valid_tokens, as_tuple=False).flatten()
        padded_indices = torch.nonzero(~valid_tokens, as_tuple=False).flatten()

        if valid_indices.numel():
            groups.append((batch_index, valid_indices))
        if padded_indices.numel():
            groups.append((batch_index, padded_indices))

    lengths = [indices.numel() for _, indices in groups]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=mask.device,
        dtype=torch.int32,
    )
    return groups, cumulative_lengths, max(lengths)


class CRTIdeogram4FlashAttention:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "CRT/Model Patches"
    DESCRIPTION = (
        "Patches Ideogram 4 attention to use FlashAttention 2, including its "
        "padding mask through the variable-length kernel. Optionally swaps the "
        "last transformer blocks between CPU and GPU to reduce VRAM use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": IDEOGRAM4_BLOCK_COUNT,
                        "step": 1,
                        "tooltip": (
                            "Number of trailing Ideogram transformer blocks to "
                            "keep on CPU and load one at a time during inference."
                        ),
                    },
                ),
            }
        }

    def patch(self, model, blocks_to_swap):
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError as error:
            raise ImportError(
                "FlashAttention 2 is required for this node. Install the "
                "'flash-attn' package in ComfyUI's Python environment."
            ) from error

        model_clone = model.clone()
        cached_mask_ref = None
        cached_plan = None
        fallback_messages = set()
        block_swap_handles = []
        block_swap_logged = False

        def fallback(func, reason, *args, **kwargs):
            if reason not in fallback_messages:
                logging.warning(
                    "[CRT Ideogram 4 FlashAttention] Falling back to ComfyUI "
                    "attention: %s",
                    reason,
                )
                fallback_messages.add(reason)
            return func(*args, **kwargs)

        def attention_override(
            func,
            q,
            k,
            v,
            heads,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            **kwargs,
        ):
            nonlocal cached_mask_ref, cached_plan

            original_args = (q, k, v, heads)
            original_kwargs = {
                "mask": mask,
                "attn_precision": attn_precision,
                "skip_reshape": skip_reshape,
                "skip_output_reshape": skip_output_reshape,
                **kwargs,
            }

            if q.device.type != "cuda":
                return fallback(
                    func, "QKV tensors are not on CUDA.", *original_args, **original_kwargs
                )

            input_dtype = v.dtype
            if q.dtype not in (torch.float16, torch.bfloat16):
                q = q.to(torch.float16)
            if k.dtype not in (torch.float16, torch.bfloat16):
                k = k.to(torch.float16)
            if v.dtype not in (torch.float16, torch.bfloat16):
                v = v.to(torch.float16)

            if skip_reshape:
                batch_size, actual_heads, sequence_length, head_dim = q.shape
                if actual_heads != heads:
                    return fallback(
                        func,
                        "QKV head count does not match the requested head count.",
                        *original_args,
                        **original_kwargs,
                    )
                q_nhd, k_nhd, v_nhd = (
                    tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)
                )
            else:
                batch_size, sequence_length, inner_dim = q.shape
                if inner_dim % heads:
                    return fallback(
                        func,
                        "QKV inner dimension is not divisible by the head count.",
                        *original_args,
                        **original_kwargs,
                    )
                head_dim = inner_dim // heads
                q_nhd, k_nhd, v_nhd = (
                    tensor.view(batch_size, sequence_length, heads, head_dim)
                    for tensor in (q, k, v)
                )

            if head_dim > 256:
                return fallback(
                    func,
                    f"FlashAttention 2 supports head dimensions up to 256, got {head_dim}.",
                    *original_args,
                    **original_kwargs,
                )

            try:
                if mask is None:
                    output = flash_attn_func(
                        q_nhd,
                        k_nhd,
                        v_nhd,
                        dropout_p=0.0,
                        causal=False,
                    )
                else:
                    if cached_mask_ref is None or cached_mask_ref() is not mask:
                        cached_plan = _build_varlen_plan(
                            mask, batch_size, sequence_length
                        )
                        cached_mask_ref = weakref.ref(mask)

                    groups, cumulative_lengths, max_length = cached_plan
                    packed_q = torch.cat(
                        [
                            q_nhd[batch_index].index_select(0, indices)
                            for batch_index, indices in groups
                        ],
                        dim=0,
                    )
                    packed_k = torch.cat(
                        [
                            k_nhd[batch_index].index_select(0, indices)
                            for batch_index, indices in groups
                        ],
                        dim=0,
                    )
                    packed_v = torch.cat(
                        [
                            v_nhd[batch_index].index_select(0, indices)
                            for batch_index, indices in groups
                        ],
                        dim=0,
                    )

                    packed_output = flash_attn_varlen_func(
                        packed_q,
                        packed_k,
                        packed_v,
                        cumulative_lengths,
                        cumulative_lengths,
                        max_length,
                        max_length,
                        dropout_p=0.0,
                        causal=False,
                    )

                    output = torch.empty_like(q_nhd)
                    offset = 0
                    for batch_index, indices in groups:
                        group_length = indices.numel()
                        output[batch_index].index_copy_(
                            0,
                            indices,
                            packed_output[offset : offset + group_length],
                        )
                        offset += group_length
            except Exception as error:
                return fallback(
                    func,
                    str(error),
                    *original_args,
                    **original_kwargs,
                )

            output = output.to(input_dtype)
            if skip_output_reshape:
                return output.transpose(1, 2)
            return output.reshape(batch_size, sequence_length, heads * head_dim)

        def remove_block_swap_hooks():
            while block_swap_handles:
                block_swap_handles.pop().remove()

        def setup_block_swap(patcher):
            nonlocal block_swap_logged

            remove_block_swap_hooks()
            if blocks_to_swap == 0:
                return

            diffusion_model = patcher.model.diffusion_model
            layers = getattr(diffusion_model, "layers", None)
            if layers is None:
                raise ValueError(
                    "Block swap requires an Ideogram 4 model with a 'layers' "
                    "transformer stack."
                )

            layer_count = len(layers)
            if layer_count != IDEOGRAM4_BLOCK_COUNT:
                raise ValueError(
                    "This node expects the 34-block Ideogram 4 architecture, "
                    f"but the connected model has {layer_count} blocks."
                )
            if blocks_to_swap > layer_count:
                raise ValueError(
                    f"Cannot swap {blocks_to_swap} blocks from a {layer_count}-block model."
                )

            offload_device = patcher.offload_device
            swapped_layers = layers[layer_count - blocks_to_swap :]

            def move_to_execution_device(module, args):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError(
                        "Could not determine the execution device for an "
                        "Ideogram transformer block."
                    )
                module.to(args[0].device)

            def move_to_offload_device(module, args, output):
                module.to(offload_device)
                return output

            for layer in swapped_layers:
                layer.to(offload_device)
                block_swap_handles.append(
                    layer.register_forward_pre_hook(move_to_execution_device)
                )
                try:
                    handle = layer.register_forward_hook(
                        move_to_offload_device,
                        always_call=True,
                    )
                except TypeError:
                    handle = layer.register_forward_hook(move_to_offload_device)
                block_swap_handles.append(handle)

            if not block_swap_logged:
                logging.info(
                    "[CRT Ideogram 4 FlashAttention] Block swap enabled for "
                    "the last %d of %d transformer blocks using %s.",
                    blocks_to_swap,
                    layer_count,
                    offload_device,
                )
                block_swap_logged = True

        def setup_block_swap_after_load(patcher, *args):
            setup_block_swap(patcher)

        def cleanup_block_swap(patcher):
            remove_block_swap_hooks()

        model_clone.model_options.setdefault("transformer_options", {})[
            "optimized_attention_override"
        ] = attention_override
        if blocks_to_swap:
            model_clone.add_callback(CallbacksMP.ON_LOAD, setup_block_swap_after_load)
            model_clone.add_callback(CallbacksMP.ON_PRE_RUN, setup_block_swap)
            model_clone.add_callback(CallbacksMP.ON_CLEANUP, cleanup_block_swap)

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "CRTIdeogram4FlashAttention": CRTIdeogram4FlashAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTIdeogram4FlashAttention": "Ideogram 4 FlashAttention (CRT)",
}

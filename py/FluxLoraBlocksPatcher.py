import torch
import comfy.model_management
import comfy.model_patcher
import comfy.weight_adapter
import re

MAX_SINGLE_BLOCKS_COUNT = 38
MAX_DOUBLE_BLOCKS_COUNT = 19


class FluxLoraBlocksPatcher:
    CATEGORY = "CRT/LoRA"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flux_model": ("MODEL",),
            }
        }
        for i in range(MAX_SINGLE_BLOCKS_COUNT):
            inputs["required"][f"lora_block_{i}_weight"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                },
            )
        for i in range(MAX_DOUBLE_BLOCKS_COUNT):
            inputs["required"][f"lora_block_{i}_double_weight"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                },
            )
        return inputs

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"

    def patch(self, flux_model, **kwargs):
        m = flux_model.clone()

        single_block_scales = {i: kwargs.get(f"lora_block_{i}_weight", 1.0) for i in range(MAX_SINGLE_BLOCKS_COUNT)}
        double_block_scales = {
            i: kwargs.get(f"lora_block_{i}_double_weight", 1.0) for i in range(MAX_DOUBLE_BLOCKS_COUNT)
        }

        active_lora_processing_needed = (
            any(abs(s - 1.0) > 1e-5 for s in single_block_scales.values())
            or any(abs(s) < 1e-5 for s in single_block_scales.values())
            or any(abs(s - 1.0) > 1e-5 for s in double_block_scales.values())
            or any(abs(s) < 1e-5 for s in double_block_scales.values())
        )

        if not active_lora_processing_needed:
            return (m,)

        block_type_idx_extract_pattern = r"(single_blocks|double_blocks)\.(\d+)\."

        try:
            compiled_block_type_idx_extract_pattern = re.compile(block_type_idx_extract_pattern)
        except re.error as e:
            print(f"[FluxLoraBlocksPatcher ERROR] Regex compilation error: {e}")
            return (m,)

        lora_patches_were_actually_modified_in_loop = False
        if hasattr(m, 'patches') and m.patches:
            keys_to_delete_from_patches_dict = []

            for target_key, original_patch_list_for_key in list(m.patches.items()):
                match_obj = compiled_block_type_idx_extract_pattern.search(target_key)
                if match_obj:
                    block_type_str = match_obj.group(1)
                    captured_block_idx_str = match_obj.group(2)

                    if captured_block_idx_str.isdigit():
                        block_idx_from_key = int(captured_block_idx_str)
                        lora_scale_for_this_block = 1.0

                        if block_type_str == "single_blocks":
                            if block_idx_from_key < MAX_SINGLE_BLOCKS_COUNT:
                                lora_scale_for_this_block = single_block_scales.get(block_idx_from_key, 1.0)
                            else:
                                continue
                        elif block_type_str == "double_blocks":
                            if block_idx_from_key < MAX_DOUBLE_BLOCKS_COUNT:
                                lora_scale_for_this_block = double_block_scales.get(block_idx_from_key, 1.0)
                            else:
                                continue
                        else:
                            continue

                        needs_modification = (
                            abs(lora_scale_for_this_block - 1.0) > 1e-5 or abs(lora_scale_for_this_block) < 1e-5
                        )

                        if not needs_modification:
                            continue

                        lora_patches_were_actually_modified_in_loop = True

                        new_patch_ops_for_this_key = []
                        for patch_op in original_patch_list_for_key:
                            processed_op = patch_op

                            if (
                                isinstance(patch_op, tuple)
                                and len(patch_op) == 5
                                and isinstance(patch_op[1], comfy.weight_adapter.LoRAAdapter)
                            ):
                                original_lora_strength_mult = float(patch_op[0])
                                adapter_obj = patch_op[1]
                                preserved_elements = patch_op[2:]

                                if abs(lora_scale_for_this_block) < 1e-7:
                                    processed_op = None
                                else:
                                    effective_new_strength_mult = (
                                        original_lora_strength_mult * lora_scale_for_this_block
                                    )
                                    if abs(effective_new_strength_mult) > 1e-7:
                                        processed_op = tuple(
                                            [effective_new_strength_mult, adapter_obj] + list(preserved_elements)
                                        )
                                    else:
                                        processed_op = None

                            elif isinstance(patch_op, tuple) and len(patch_op) == 2:
                                op_strength, op_tensor = patch_op
                                if isinstance(op_strength, (float, int)) and isinstance(op_tensor, torch.Tensor):
                                    original_lora_strength = float(op_strength)
                                    if abs(lora_scale_for_this_block) < 1e-7:
                                        processed_op = None
                                    else:
                                        effective_new_strength = original_lora_strength * lora_scale_for_this_block
                                        if abs(effective_new_strength) > 1e-7:
                                            processed_op = (effective_new_strength, op_tensor)
                                        else:
                                            processed_op = None

                            if processed_op is not None:
                                new_patch_ops_for_this_key.append(processed_op)

                        if new_patch_ops_for_this_key:
                            m.patches[target_key] = new_patch_ops_for_this_key
                        elif target_key in m.patches:
                            keys_to_delete_from_patches_dict.append(target_key)

            for key_to_del in keys_to_delete_from_patches_dict:
                if key_to_del in m.patches:
                    del m.patches[key_to_del]

        if lora_patches_were_actually_modified_in_loop:
            try:
                m.add_patches({}, 1.0, 1.0)
            except Exception as e:
                print(f"[FluxLoraBlocksPatcher WARNING] Nudge attempt m.add_patches failed: {e}")

        return (m,)


NODE_CLASS_MAPPINGS = {"FluxLoraBlocksPatcher": FluxLoraBlocksPatcher}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxLoraBlocksPatcher": "Flux LoRA Blocks Patcher"}

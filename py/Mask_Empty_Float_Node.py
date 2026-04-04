import torch
import numpy as np
import comfy


class MaskEmptyFloatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "value_if_non_empty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_output",)
    FUNCTION = "check_mask"
    CATEGORY = "CRT/Utils/Logic & Values"

    def check_mask(self, mask, value_if_non_empty):
        mask_np = mask.cpu().numpy()
        is_empty = np.all(mask_np == 0)
        return (0.0 if is_empty else float(value_if_non_empty),)

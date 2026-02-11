import torch
import comfy


class MaskPassOrPlaceholder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_output",)
    FUNCTION = "process_mask"
    CATEGORY = "CRT/Utils/Logic & Values"

    def process_mask(self, mask=None):
        if mask is not None:
            return (mask,)
        else:
            placeholder_mask = torch.ones((1, 2, 2), dtype=torch.float32)
            return (placeholder_mask,)

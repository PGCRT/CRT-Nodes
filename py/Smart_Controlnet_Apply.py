import torch
import comfy.utils
import comfy.model_management as mm
from nodes import ControlNetApplyAdvanced
import time


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_print(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")


class SmartControlNetApply:
    """
    Smart ControlNet node that applies ControlNet with intelligent bypassing.
    Skips ControlNet application when strength is 0 to avoid unnecessary computation.
    Expects preprocessed images as input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "union_type": (
                    ["auto"]
                    + list(
                        getattr(
                            __import__('comfy.cldm.control_types', fromlist=['UNION_CONTROLNET_TYPES']),
                            'UNION_CONTROLNET_TYPES',
                            {},
                        ).keys()
                    ),
                    {"default": "auto"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "processed_image")
    FUNCTION = "apply_smart_controlnet"
    CATEGORY = "CRT/Conditioning"

    def __init__(self):
        self.controlnet_applier = ControlNetApplyAdvanced()

    def _set_union_controlnet_type(self, control_net, union_type):
        """Set union ControlNet type if specified."""
        if union_type == "auto":
            return control_net

        try:
            from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

            control_net = control_net.copy()
            type_number = UNION_CONTROLNET_TYPES.get(union_type, -1)
            if type_number >= 0:
                control_net.set_extra_arg("control_type", [type_number])
            else:
                control_net.set_extra_arg("control_type", [])
        except ImportError:
            colored_print("   ⚠️ Union ControlNet types not available", Colors.YELLOW)

        return control_net

    def _handle_inpainting_controlnet(self, control_net, image, mask, vae):
        """Handle inpainting ControlNet if mask is provided."""
        if mask is None or not hasattr(control_net, 'concat_mask') or not control_net.concat_mask:
            return image, []

        mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
        masked_image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])

        colored_print("   🎭 Applied inpainting mask", Colors.CYAN)
        return masked_image, [mask]

    def apply_smart_controlnet(
        self, positive, negative, image, vae, control_net, strength, end_percent, union_type, mask=None
    ):
        """
        Smart ControlNet application with intelligent bypassing.
        Expects preprocessed images as input.
        """
        colored_print("\n🎨 Smart ControlNet Processing", Colors.HEADER)
        colored_print(f"   📊 Strength: {strength:.3f} | End: {end_percent:.3f}", Colors.BLUE)

        start_time = time.time()

        # Early exit if strength is 0
        if strength == 0.0:
            colored_print("   ⚡ BYPASSED - Strength is 0, skipping all processing", Colors.YELLOW)
            return (positive, negative, image)

        # Set union type
        control_net = self._set_union_controlnet_type(control_net, union_type)
        if union_type != "auto":
            colored_print(f"   🔗 Union type: {union_type}", Colors.CYAN)

        # Use input image as-is (assumes it's already preprocessed)
        processed_image = image

        # Handle inpainting
        extra_concat = []
        if mask is not None:
            processed_image, extra_concat = self._handle_inpainting_controlnet(control_net, processed_image, mask, vae)

        # Apply ControlNet
        colored_print("   🎮 Applying ControlNet...", Colors.GREEN)
        try:
            if extra_concat:
                result = self.controlnet_applier.apply_controlnet(
                    positive,
                    negative,
                    control_net,
                    processed_image,
                    strength,
                    0.0,
                    end_percent,
                    vae=vae,
                    extra_concat=extra_concat,
                )
            else:
                result = self.controlnet_applier.apply_controlnet(
                    positive, negative, control_net, processed_image, strength, 0.0, end_percent, vae=vae
                )

            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart ControlNet completed in {total_time:.2f}s", Colors.GREEN)

            return (result[0], result[1], processed_image)

        except Exception as e:
            colored_print(f"   ❌ ControlNet application failed: {str(e)}", Colors.RED)
            return (positive, negative, processed_image)


# Register the node mappings
NODE_CLASS_MAPPINGS = {
    "SmartControlNetApply": SmartControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartControlNetApply": "🎨 Smart ControlNet Apply",
}

import math
import comfy.utils


class ImageScaleRangeFromMp:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (s.upscale_methods,),
                "min_megapixels": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 64.0, "step": 0.01}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "CRT/Image"

    def execute(self, image, upscale_method, min_megapixels, max_megapixels, resolution_steps):
        samples = image.movedim(-1, 1)
        h, w = samples.shape[2], samples.shape[3]
        current_mp = (h * w) / (1024 * 1024)

        if current_mp < min_megapixels:
            target_mp = min_megapixels
        elif current_mp > max_megapixels:
            target_mp = max_megapixels
        else:
            target_mp = None

        # Resize both axes by one scale factor. Quantizing each target axis before
        # this resize would change the aspect ratio and stretch the image.
        if target_mp is not None:
            total = target_mp * 1024 * 1024
            scale_by = math.sqrt(total / (w * h))
            resized_w = max(1, round(w * scale_by))
            resized_h = max(1, round(h * scale_by))
            samples = comfy.utils.common_upscale(
                samples, resized_w, resized_h, upscale_method, "disabled"
            )
        else:
            resized_w = int(w)
            resized_h = int(h)

        # Quantize with a centered crop, never with a second resize. Cropping
        # downward means the final MP can be slightly below the selected bound.
        step = max(1, int(resolution_steps))
        cropped_w = (resized_w // step) * step
        cropped_h = (resized_h // step) * step

        # A dimension smaller than the requested step has no positive,
        # step-aligned crop. Keep that dimension instead of returning an empty
        # image; normal image sizes are expected to be at least one full step.
        if cropped_w == 0:
            cropped_w = resized_w
        if cropped_h == 0:
            cropped_h = resized_h

        left = (resized_w - cropped_w) // 2
        top = (resized_h - cropped_h) // 2
        samples = samples[:, :, top : top + cropped_h, left : left + cropped_w]

        result = samples.movedim(1, -1)
        return (result, int(cropped_w), int(cropped_h))

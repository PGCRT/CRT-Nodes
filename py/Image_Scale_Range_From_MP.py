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
            return (image, int(w), int(h))

        total = target_mp * 1024 * 1024
        scale_by = math.sqrt(total / (w * h))
        new_w = round(w * scale_by / resolution_steps) * resolution_steps
        new_h = round(h * scale_by / resolution_steps) * resolution_steps

        s = comfy.utils.common_upscale(samples, int(new_w), int(new_h), upscale_method, "disabled")
        s = s.movedim(1, -1)
        return (s, int(new_w), int(new_h))

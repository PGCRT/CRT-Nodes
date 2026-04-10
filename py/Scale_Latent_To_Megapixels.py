import math

import comfy.utils


class ScaleLatentToMegapixels:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (cls.upscale_methods,),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 200.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "CRT/Latent"

    def upscale(self, samples, upscale_method, megapixels):
        s = samples.copy()

        current_latent_w = samples["samples"].shape[-1]
        current_latent_h = samples["samples"].shape[-2]
        current_pixels = (current_latent_w * 8) * (current_latent_h * 8)
        target_pixels = max(1.0, megapixels * 1000000.0)

        scale_by = math.sqrt(target_pixels / current_pixels)
        width = max(1, round(current_latent_w * scale_by))
        height = max(1, round(current_latent_h * scale_by))

        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], width, height, upscale_method, "disabled"
        )
        return (s,)


NODE_CLASS_MAPPINGS = {"Scale Latent to Megapixels": ScaleLatentToMegapixels}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Scale Latent to Megapixels": "Scale Latent to Megapixels (CRT)"
}

import torch
import torch.nn.functional as F
import numpy as np


class CRT_QuantizeAndCropImage:
    BASE_BUCKETS = [
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
        (1088, 1088),
        (1152, 960),
        (960, 1152),
        (1280, 896),
        (896, 1280),
        (1344, 832),
        (832, 1344),
        (1408, 768),
        (768, 1408),
        (1472, 704),
        (704, 1472),
        (1600, 640),
        (640, 1600),
        (1664, 576),
        (576, 1664),
        (1728, 576),
        (576, 1728),
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_side_length": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "The absolute maximum size (in pixels) for either the width or height of the final image.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "CRT/Image"

    def _quantize_dimensions(self, original_width, original_height, max_side_length):
        if original_width == 0 or original_height == 0:
            return max_side_length, max_side_length

        original_aspect_ratio = original_width / original_height

        best_bucket = None
        min_aspect_diff = float('inf')

        for bucket_w, bucket_h in self.BASE_BUCKETS:
            bucket_aspect_ratio = bucket_w / bucket_h
            aspect_diff = abs(original_aspect_ratio - bucket_aspect_ratio)
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                best_bucket = (bucket_w, bucket_h)

        target_w, target_h = best_bucket

        max_dim = max(target_w, target_h)
        if max_dim > max_side_length:
            downscale_ratio = max_side_length / max_dim
            target_w = int(round(target_w * downscale_ratio))
            target_h = int(round(target_h * downscale_ratio))

        target_w = int(round(target_w / 64.0) * 64)
        target_h = int(round(target_h / 64.0) * 64)

        if target_w == 0:
            target_w = 64
        if target_h == 0:
            target_h = 64

        return target_w, target_h

    def execute(self, image, max_side_length):
        _, original_h, original_w, _ = image.shape

        target_w, target_h = self._quantize_dimensions(original_w, original_h, max_side_length)

        img_aspect = original_w / original_h
        target_aspect = target_w / target_h

        if img_aspect > target_aspect:
            scale_h = target_h
            scale_w = int(round(scale_h * img_aspect))
        else:
            scale_w = target_w
            scale_h = int(round(scale_w / img_aspect))

        interpolation_mode = "bicubic"

        img_permuted = image.permute(0, 3, 1, 2)
        resized_img = F.interpolate(img_permuted, size=(scale_h, scale_w), mode=interpolation_mode, antialias=True)
        resized_img = resized_img.permute(0, 2, 3, 1)

        y_offset = max(0, resized_img.shape[1] - target_h)
        x_offset = max(0, resized_img.shape[2] - target_w)

        top = y_offset // 2
        left = x_offset // 2

        cropped_image = resized_img[:, top : top + target_h, left : left + target_w, :]

        return (cropped_image, target_w, target_h)

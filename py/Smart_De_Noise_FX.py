import torch
import torch.nn.functional as F


class SmartDeNoiseFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": ("FLOAT", {"default": 1.25, "min": 0.001, "max": 8.0, "step": 0.001, "display": "slider"}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.001, "max": 0.25, "step": 0.001, "display": "slider"},
                ),
                "radius_multiplier": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.001, "display": "slider"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_smart_denoise"
    CATEGORY = "CRT/FX"

    def _gaussian_kernel(self, sigma, radius, device, dtype):
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma + 1e-6))
        return kernel / kernel.sum()

    def _separable_blur(self, image_bchw, sigma, radius):
        if radius <= 0:
            return image_bchw

        batch, channels, _, _ = image_bchw.shape
        kernel = self._gaussian_kernel(sigma, radius, image_bchw.device, image_bchw.dtype)
        kernel_h = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        kernel_v = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

        blurred = F.conv2d(
            F.pad(image_bchw, (radius, radius, 0, 0), mode='replicate'),
            kernel_h,
            groups=channels,
        )
        blurred = F.conv2d(
            F.pad(blurred, (0, 0, radius, radius), mode='replicate'),
            kernel_v,
            groups=channels,
        )
        return blurred

    def apply_smart_denoise(self, image: torch.Tensor, sigma, threshold, radius_multiplier):
        image_bchw = image.permute(0, 3, 1, 2)
        radius = int(round(max(0.0, radius_multiplier) * sigma * 2.0))
        if radius <= 0:
            return (image.clamp(0.0, 1.0),)

        max_radius = max(1, min(image_bchw.shape[-2], image_bchw.shape[-1]) // 2)
        radius = min(radius, max_radius)

        blurred = self._separable_blur(image_bchw, max(float(sigma), 1e-3), radius)
        detail = (image_bchw - blurred).abs().mean(dim=1, keepdim=True)

        # Preserve edges by reducing blur where the local detail exceeds the threshold.
        threshold = max(float(threshold), 1e-6)
        edge_preserve = torch.exp(-(detail * detail) / (2.0 * threshold * threshold))
        final_image = torch.lerp(image_bchw, blurred, edge_preserve)
        return (final_image.permute(0, 2, 3, 1).clamp(0.0, 1.0),)

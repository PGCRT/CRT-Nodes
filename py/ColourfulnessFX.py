import torch


class ColourfulnessFX:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colourfulness": (
                    "FLOAT",
                    {"default": 0.4, "min": -1.0, "max": 2.0, "step": 0.01, "display": "slider"},
                ),
                "luma_limit": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_colourfulness"
    CATEGORY = "CRT/FX"

    def _soft_lim(self, v, s):
        # Sigmoid-like function: (v*s) / sqrt(s*s + v*v)
        return (v * s) * torch.rsqrt(s * s + v * v)

    def _wpmean(self, a, b, w):
        # Weighted power mean, p = 0.5
        return (w.abs() * torch.sqrt(a.abs()) + (1 - w).abs() * torch.sqrt(b.abs())) ** 2

    def apply_colourfulness(self, image: torch.Tensor, colourfulness, luma_limit):
        if image.is_cuda:
            device = image.device
        else:
            device = torch.device("cpu")

        batch_size, height, width, _ = image.shape
        result_images = []

        luma_coeff = torch.tensor([0.2558, 0.6511, 0.0931], device=device).view(1, 1, 3)

        for i in range(batch_size):
            c0 = image[i].clamp(0.0, 1.0)

            # --- Luma Calculation (fast method from shader) ---
            luma = torch.sqrt(torch.sum(c0 * c0.abs() * luma_coeff, dim=-1, keepdim=True))

            # --- Calculate saturation change ---
            diff_luma = c0 - luma
            c_diff = diff_luma * colourfulness

            # --- Smart Limiting to Prevent Clipping ---
            if colourfulness > 0.0:
                # 1. Reference limited change
                rlc_diff = torch.clamp((c_diff * 1.2) + c0, -0.0001, 1.0001) - c0

                # 2. Calculate max possible saturation increase
                max_rgb = torch.max(diff_luma, dim=-1, keepdim=True)[0]
                min_rgb = torch.min(diff_luma, dim=-1, keepdim=True)[0]

                poslim = (1.0002 - luma) / (max_rgb.abs() + 1e-7)
                neglim = (luma + 0.0002) / (min_rgb.abs() + 1e-7)

                diffmax_scale = torch.min(poslim, neglim).clamp(max=32.0)
                diffmax = diff_luma * diffmax_scale - diff_luma

                # 3. Blend the limiting methods and apply smoothly
                limit = self._wpmean(diffmax, rlc_diff, torch.tensor(luma_limit, device=device))
                c_diff = self._soft_lim(c_diff, limit.clamp(min=1e-7))

            # --- Final Combination ---
            final_rgb = (c0 + c_diff).clamp(0.0, 1.0)
            result_images.append(final_rgb)

        return (torch.stack(result_images, dim=0),)

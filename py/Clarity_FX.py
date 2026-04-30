import torch
import torch.nn.functional as F


class ClarityFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ClarityRadius": ("INT", {"default": 3, "min": 0, "max": 4, "step": 1, "display": "slider"}),
                "ClarityOffset": (
                    "FLOAT",
                    {"default": 2.00, "min": 1.00, "max": 5.00, "step": 0.01, "display": "slider"},
                ),
                "ClarityStrength": (
                    "FLOAT",
                    {"default": 0.400, "min": 0.00, "max": 2.00, "step": 0.01, "display": "slider"},
                ),
                "ClarityBlendMode": (
                    ["Hard Light", "Soft Light", "Overlay", "Multiply", "Vivid Light", "Linear Light", "Addition"],
                ),
                "ClarityBlendIfDark": ("INT", {"default": 50, "min": 0, "max": 255, "step": 5, "display": "slider"}),
                "ClarityBlendIfLight": ("INT", {"default": 205, "min": 0, "max": 255, "step": 5, "display": "slider"}),
                "ClarityDarkIntensity": (
                    "FLOAT",
                    {"default": 0.400, "min": 0.00, "max": 1.00, "step": 0.01, "display": "slider"},
                ),
                "ClarityLightIntensity": (
                    "FLOAT",
                    {"default": 0.000, "min": 0.00, "max": 1.00, "step": 0.01, "display": "slider"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_clarity"
    CATEGORY = "CRT/FX"

    def _smoothstep(self, edge0, edge1, x):
        t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def apply_clarity(
        self,
        image: torch.Tensor,
        ClarityRadius,
        ClarityOffset,
        ClarityStrength,
        ClarityBlendMode,
        ClarityBlendIfDark,
        ClarityBlendIfLight,
        ClarityDarkIntensity,
        ClarityLightIntensity,
    ):
        device = image.device
        dtype = image.dtype

        luma = (
            image[..., 0] * 0.32786885
            + image[..., 1] * 0.655737705
            + image[..., 2] * 0.0163934436
        ).unsqueeze(-1)
        chroma = image / (luma + 1e-6)
        luma_for_conv = luma.permute(0, 3, 1, 2)

        sigma = (ClarityRadius + 1) * ClarityOffset
        if sigma > 0:
            radius = max(1, int(sigma * 2))
            coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
            kernel = torch.exp(-(coords * coords) / (2 * sigma**2))
            kernel /= kernel.sum()
            kernel_h = kernel.view(1, 1, 1, 2 * radius + 1)
            kernel_v = kernel.view(1, 1, 2 * radius + 1, 1)

            blurred_luma_h = F.conv2d(F.pad(luma_for_conv, (radius, radius, 0, 0), mode='replicate'), kernel_h)
            blurred_luma_tensor = F.conv2d(F.pad(blurred_luma_h, (0, 0, radius, radius), mode='replicate'), kernel_v)
        else:
            blurred_luma_tensor = luma_for_conv

        blurred_luma = blurred_luma_tensor.permute(0, 2, 3, 1)

        sharp = 0.5 * (luma + 1.0 - blurred_luma)
        clamped_sharp = sharp.clamp(0.0, 1.0)
        sharp_min = torch.lerp(sharp, clamped_sharp, ClarityDarkIntensity)
        sharp_max = torch.lerp(sharp, clamped_sharp, ClarityLightIntensity)
        sharp = torch.where(sharp > 0.5, sharp_max, sharp_min)

        if ClarityBlendMode == "Soft Light":
            sharp = torch.where(
                sharp < 0.5,
                2 * luma * sharp + luma**2 * (1 - 2 * sharp),
                torch.sqrt(luma.clamp(min=0)) * (2 * sharp - 1) + 2 * luma * (1 - sharp),
            )
        elif ClarityBlendMode == "Overlay":
            sharp = torch.where(luma < 0.5, 2 * luma * sharp, 1 - 2 * (1 - luma) * (1 - sharp))
        elif ClarityBlendMode == "Hard Light":
            sharp = torch.where(sharp < 0.5, 2 * luma * sharp, 1 - 2 * (1 - luma) * (1 - sharp))
        elif ClarityBlendMode == "Multiply":
            sharp = torch.clamp(2 * luma * sharp, 0, 1)
        elif ClarityBlendMode == "Vivid Light":
            sharp = torch.where(
                sharp < 0.5,
                1 - (1 - luma) / (2 * sharp + 1e-6),
                luma / (2 * (1 - sharp) + 1e-6),
            )
        elif ClarityBlendMode == "Linear Light":
            sharp = torch.clamp(luma + 2.0 * sharp - 1.0, 0, 1)
        elif ClarityBlendMode == "Addition":
            sharp = torch.clamp(luma + sharp - 0.5, 0, 1)

        if ClarityBlendIfDark > 0 or ClarityBlendIfLight < 255:
            mix_val = torch.mean(image, dim=-1, keepdim=True)
            mask = torch.ones_like(mix_val)

            if ClarityBlendIfDark > 0:
                blend_if_d = ClarityBlendIfDark / 255.0
                edge0 = blend_if_d - (blend_if_d * 0.2)
                edge1 = blend_if_d + (blend_if_d * 0.2)
                mask = self._smoothstep(edge0, edge1, mix_val)

            if ClarityBlendIfLight < 255:
                blend_if_l = ClarityBlendIfLight / 255.0
                edge0 = blend_if_l - (blend_if_l * 0.2)
                edge1 = blend_if_l + (blend_if_l * 0.2)
                mask = mask * (1.0 - self._smoothstep(edge0, edge1, mix_val))

            sharp = torch.lerp(luma, sharp, mask)

        final_luma = torch.lerp(luma, sharp, ClarityStrength)
        return ((final_luma * chroma).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {"ClarityFX": ClarityFX}
NODE_DISPLAY_NAME_MAPPINGS = {"ClarityFX": "Clarity FX"}

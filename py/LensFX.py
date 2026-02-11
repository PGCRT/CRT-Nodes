import torch


class LensFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "chromatic_aberration": (
                    "FLOAT",
                    {"default": 1.65, "min": 0.0, "max": 20.0, "step": 0.01, "display": "slider"},
                ),
                "vignette": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "grain_amount": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 20.0, "step": 0.01, "display": "slider"},
                ),
                "grain_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 20.0, "step": 0.1, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lens_effects"
    CATEGORY = "CRT/FX"

    def apply_lens_effects(self, image: torch.Tensor, chromatic_aberration, vignette, grain_amount, grain_scale, seed):

        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        for i in range(batch_size):
            img_tensor = image[i : i + 1]
            processed_image_bchw = img_tensor.permute(0, 3, 1, 2)

            y_base, x_base = torch.meshgrid(
                torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing='ij'
            )

            if chromatic_aberration > 0:
                x_aspect = x_base.clone()
                y_aspect = y_base.clone()
                aspect_ratio = width / height
                if aspect_ratio > 1.0:
                    x_aspect *= aspect_ratio
                else:
                    y_aspect /= aspect_ratio

                radius_sq = x_aspect * x_aspect + y_aspect * y_aspect
                shift_amount = chromatic_aberration * 0.005

                r_scale = 1.0 - radius_sq * shift_amount
                b_scale = 1.0 + radius_sq * shift_amount

                center_vec = torch.stack((x_base, y_base), dim=-1)

                grid_r = (center_vec * r_scale.unsqueeze(-1)).unsqueeze(0)
                grid_b = (center_vec * b_scale.unsqueeze(-1)).unsqueeze(0)

                channel_r = torch.nn.functional.grid_sample(
                    processed_image_bchw[:, 0:1, :, :],
                    grid_r,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )
                channel_g = processed_image_bchw[:, 1:2, :, :]
                channel_b = torch.nn.functional.grid_sample(
                    processed_image_bchw[:, 2:3, :, :],
                    grid_b,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )

                processed_image_bchw = torch.cat([channel_r, channel_g, channel_b], dim=1)

            processed_image_bhwc = processed_image_bchw.permute(0, 2, 3, 1)

            if vignette > 0:
                x = x_base.clone()
                y = y_base.clone()
                aspect_ratio = width / height
                if aspect_ratio > 1.0:
                    x *= aspect_ratio
                else:
                    y /= aspect_ratio

                radius_sq = x * x + y * y
                vignette_mask = 1.0 - torch.pow(radius_sq, 0.5) * vignette
                processed_image_bhwc *= vignette_mask.clamp(0.0, 1.0).unsqueeze(-1)

            if grain_amount > 0:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed + i)

                grain_h = int(height / grain_scale) if grain_scale > 0 else height
                grain_w = int(width / grain_scale) if grain_scale > 0 else width

                grain = torch.rand(1, grain_h, grain_w, 1, generator=generator, device=device) * 2.0 - 1.0
                grain = torch.nn.functional.interpolate(
                    grain.permute(0, 3, 1, 2), size=(height, width), mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1)

                processed_image_bhwc += grain * grain_amount * 0.1

            result_images.append(processed_image_bhwc.clamp(0.0, 1.0))

        return (torch.cat(result_images, dim=0),)

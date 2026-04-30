import torch
import torch.nn.functional as F


class LensDistortFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "distortion": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "anamorphic_squeeze": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "display": "slider"},
                ),
                "chromatic_radius": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lens_distortion"
    CATEGORY = "CRT/FX"

    def _base_grid(self, height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing='ij',
        )
        return x, y

    def _distorted_grid(self, x, y, width, height, distortion, anamorphic_squeeze):
        aspect_ratio = width / height
        x_norm, y_norm = x.clone(), y.clone()
        if aspect_ratio > 1.0:
            x_norm *= aspect_ratio
        else:
            y_norm /= aspect_ratio

        coord = torch.stack([x_norm, y_norm], dim=-1)
        coord[..., 1] /= anamorphic_squeeze

        r2 = torch.sum(coord * coord, dim=-1, keepdim=True)
        k = distortion * 0.18
        scale = 1.0 + k * r2 + (k * 0.35) * r2 * r2
        distorted = coord * scale
        distorted[..., 1] *= anamorphic_squeeze

        if aspect_ratio > 1.0:
            distorted[..., 0] /= aspect_ratio
        else:
            distorted[..., 1] *= aspect_ratio
        return distorted.clamp(-1.5, 1.5)

    def apply_lens_distortion(self, image: torch.Tensor, distortion, anamorphic_squeeze, chromatic_radius):
        device = image.device
        dtype = image.dtype
        batch_size, height, width, _ = image.shape
        image_bchw = image.permute(0, 3, 1, 2)

        x, y = self._base_grid(height, width, device, dtype)
        base_grid = torch.stack([x, y], dim=-1)
        distorted_grid = self._distorted_grid(x, y, width, height, distortion, anamorphic_squeeze)
        grid = distorted_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if abs(chromatic_radius) > 1e-6:
            aberration = (distorted_grid - base_grid) * chromatic_radius
            r_grid = (distorted_grid + aberration).unsqueeze(0).expand(batch_size, -1, -1, -1)
            b_grid = (distorted_grid - aberration).unsqueeze(0).expand(batch_size, -1, -1, -1)

            red = F.grid_sample(image_bchw[:, 0:1], r_grid, mode='bilinear', padding_mode='border', align_corners=False)
            green = F.grid_sample(image_bchw[:, 1:2], grid, mode='bilinear', padding_mode='border', align_corners=False)
            blue = F.grid_sample(image_bchw[:, 2:3], b_grid, mode='bilinear', padding_mode='border', align_corners=False)
            output = torch.cat([red, green, blue], dim=1)
        else:
            output = F.grid_sample(image_bchw, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return (output.permute(0, 2, 3, 1).clamp(0.0, 1.0),)

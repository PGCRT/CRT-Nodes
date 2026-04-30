import torch
import torch.nn.functional as F


class ContourFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "sensitivity": ("FLOAT", {"default": 0.77, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "line_color": ("COLOR", {"default": "#FFFFFF"}),
                "line_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "background_color": ("COLOR", {"default": "#000000"}),
                "background_opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "output_mode": (["Final Image", "Edge Mask"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_contour"
    CATEGORY = "CRT/FX"

    def apply_contour(
        self,
        image: torch.Tensor,
        threshold,
        softness,
        sensitivity,
        line_color,
        line_opacity,
        background_color,
        background_opacity,
        output_mode,
    ):

        device = image.device
        dtype = image.dtype
        batch_size, height, width, _ = image.shape

        line_color_tensor = torch.tensor(
            [int(line_color[1:3], 16) / 255.0, int(line_color[3:5], 16) / 255.0, int(line_color[5:7], 16) / 255.0],
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, 3)
        background_color_tensor = torch.tensor(
            [
                int(background_color[1:3], 16) / 255.0,
                int(background_color[3:5], 16) / 255.0,
                int(background_color[5:7], 16) / 255.0,
            ],
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, 3)

        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        luma_weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype).view(1, 3, 1, 1)

        img_tensor_bchw = image.permute(0, 3, 1, 2)
        grayscale_bchw = torch.sum(img_tensor_bchw * luma_weights, dim=1, keepdim=True)

        grad_x = F.conv2d(grayscale_bchw, sobel_x, padding=1)
        grad_y = F.conv2d(grayscale_bchw, sobel_y, padding=1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = gradient_magnitude * (1.0 + sensitivity * 9.0)

        inverse_range = 1.0 / (softness + 1e-6)
        mask = torch.clamp((gradient_magnitude - threshold) * inverse_range, 0.0, 1.0)
        mask = mask.permute(0, 2, 3, 1).expand(batch_size, height, width, 3)

        if output_mode == "Edge Mask":
            final_image = mask
        else:
            background = torch.lerp(image, background_color_tensor, background_opacity)
            final_image = torch.lerp(background, line_color_tensor, mask * line_opacity)

        return (final_image.clamp(0.0, 1.0),)

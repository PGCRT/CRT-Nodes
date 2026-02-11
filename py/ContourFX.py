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
        batch_size, height, width, _ = image.shape
        result_images = []

        line_color_tensor = torch.tensor(
            [int(line_color[1:3], 16) / 255.0, int(line_color[3:5], 16) / 255.0, int(line_color[5:7], 16) / 255.0],
            device=device,
        )
        background_color_tensor = torch.tensor(
            [
                int(background_color[1:3], 16) / 255.0,
                int(background_color[3:5], 16) / 255.0,
                int(background_color[5:7], 16) / 255.0,
            ],
            device=device,
        )

        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        luma_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)

        for i in range(batch_size):
            img_tensor_bhwc = image[i : i + 1]
            img_tensor_bchw = img_tensor_bhwc.permute(0, 3, 1, 2)

            grayscale_bchw = torch.sum(img_tensor_bchw * luma_weights, dim=1, keepdim=True)

            grad_x = F.conv2d(grayscale_bchw, sobel_x, padding=1)
            grad_y = F.conv2d(grayscale_bchw, sobel_y, padding=1)

            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

            gradient_magnitude = gradient_magnitude * (1.0 + sensitivity * 9.0)

            inverse_range = 1.0 / (softness + 1e-6)
            mask = torch.clamp((gradient_magnitude - threshold) * inverse_range, 0.0, 1.0)

            mask = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)

            if output_mode == "Edge Mask":
                final_image = mask
            else:
                background = torch.lerp(img_tensor_bhwc, background_color_tensor.view(1, 1, 1, 3), background_opacity)
                line_with_opacity = line_color_tensor * line_opacity
                final_image = torch.lerp(background, line_with_opacity.view(1, 1, 1, 3), mask)

            result_images.append(final_image.clamp(0.0, 1.0))

        return (torch.cat(result_images, dim=0),)

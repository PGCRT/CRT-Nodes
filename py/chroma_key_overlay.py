import torch
import numpy as np
from PIL import Image
import scipy.ndimage


class CRTChromaKeyOverlay:
    """
    A node that keys an image based on a color, or uses the image's existing
    alpha channel if it's present (RGBA). The result can be optionally blended
    onto a background image.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "ClickOnMe": ("COLOR", {"default": "#00FF00"}),
                "tolerance": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "expand_pixels": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
            },
            "optional": {
                "background_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_with_alpha", "blended_image")
    FUNCTION = "process"
    CATEGORY = "CRT/Image"

    def process(self, images, ClickOnMe, tolerance, expand_pixels, blur_radius, background_images=None):
        batch_size, height, width, channels = images.shape
        device = images.device

        rgb_image = images[..., :3]
        final_alpha_mask = None

        # Check if the input image already has an alpha channel
        if channels == 4:
            print("[CRTChromaKeyOverlay] Input has alpha channel. Using it directly.")
            # Use the existing alpha channel from the source image
            final_alpha_mask = images[..., 3:4]
        else:
            # If input is RGB, perform the chroma key operation
            print("[CRTChromaKeyOverlay] Input is RGB. Performing chroma key operation.")

            # Convert hex color string to a normalized RGB tensor
            hex_color = ClickOnMe.lstrip('#')
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            target_rgb = (r / 255.0, g / 255.0, b / 255.0)
            target_color = torch.tensor(target_rgb, dtype=torch.float32, device=device).view(1, 1, 1, 3)

            # Calculate color distance on the RGB image
            color_distance = torch.sqrt(torch.sum((rgb_image - target_color) ** 2, dim=3))
            threshold = (tolerance / 100.0) * (3**0.5)
            alpha_mask = (color_distance > threshold).float()

            # Process the mask (Expand and Blur) using SciPy on CPU
            alpha_mask_np = alpha_mask.cpu().numpy()
            processed_alpha_list = []
            for i in range(batch_size):
                mask_slice = alpha_mask_np[i]

                if expand_pixels > 0:
                    inverted_mask = 1 - mask_slice
                    dilated_inverted_mask = scipy.ndimage.binary_dilation(inverted_mask, iterations=expand_pixels)
                    mask_slice = 1 - dilated_inverted_mask

                if blur_radius > 0:
                    mask_slice = scipy.ndimage.gaussian_filter(mask_slice, sigma=blur_radius)

                processed_alpha_list.append(torch.from_numpy(mask_slice))

            final_alpha_mask = torch.stack(processed_alpha_list).unsqueeze(-1).to(device)

        # Combine the original RGB with the determined alpha mask
        image_with_alpha = torch.cat((rgb_image, final_alpha_mask), dim=-1)

        # Default the blended result to the image with its new alpha
        blended_image_result = image_with_alpha.clone()

        # If a background is provided, perform the blend
        if background_images is not None:
            # Resize background to match foreground if necessary
            if background_images.shape[1] != height or background_images.shape[2] != width:
                background_images = torch.nn.functional.interpolate(
                    background_images.permute(0, 3, 1, 2), size=(height, width), mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1)

            # Repeat background images if its batch size is smaller than the foreground's
            if background_images.shape[0] < batch_size:
                repeat_factor = (batch_size + background_images.shape[0] - 1) // background_images.shape[0]
                background_images = background_images.repeat(repeat_factor, 1, 1, 1)[:batch_size]

            # Define foreground, background, and alpha
            fg = rgb_image
            bg = background_images[..., :3]
            alpha = final_alpha_mask

            # Perform the alpha blend calculation
            blended_rgb = fg * alpha + bg * (1.0 - alpha)

            # Attach a full alpha channel to the blended result
            blended_image_result = torch.cat((blended_rgb, torch.ones_like(alpha)), dim=-1)

        print(f"[CRTChromaKeyOverlay] Processed {batch_size} image(s).")

        return (image_with_alpha, blended_image_result)

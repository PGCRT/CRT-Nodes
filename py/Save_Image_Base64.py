"""
save_image_base64.py  –  ComfyUI node: Save Image (Base64)
===========================================================
Encodes an IMAGE tensor to a base64 string, bypassing disk writes entirely.
The inverse of LoadImageBase64.
"""

import base64
import io
import numpy as np
from PIL import Image


class SaveImageBase64:
    """
    Encode an IMAGE tensor to a base64 string.
    Useful for returning images inline in workflow JSON or passing
    to external systems without writing to disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_data",)
    FUNCTION = "encode"
    CATEGORY = "CRT/Save"

    def encode(self, images, format, quality):
        # images: [B, H, W, C] float32 [0, 1]
        img = images[0]  # take first frame/image
        arr = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

        if arr.shape[2] == 3:
            pil_img = Image.fromarray(arr, "RGB")
        else:
            pil_img = Image.fromarray(arr, "RGBA")

        buf = io.BytesIO()
        if format == "JPEG":
            pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=quality)
        elif format == "WEBP":
            pil_img.save(buf, format="WEBP", quality=quality)
        else:
            pil_img.save(buf, format="PNG")

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return (b64,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SaveImageBase64 (CRT)": SaveImageBase64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageBase64 (CRT)": "Save Image Base64 (CRT)",
}

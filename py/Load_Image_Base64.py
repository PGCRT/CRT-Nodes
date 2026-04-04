"""
load_image_base64.py  –  ComfyUI node: Load Image (Base64)
===========================================================
Accepts a base64-encoded image string (JPEG, PNG, etc.) directly as input,
bypassing ComfyUI's file-upload endpoint entirely.

Used by DFFS_addon.cpp to embed the captured frame inline in the workflow
JSON, avoiding the /upload/image round-trip and the resulting disk write.
"""

import base64
import io
import numpy as np
import torch
from PIL import Image


class LoadImageBase64:
    """
    Decode a base64-encoded image directly to IMAGE + MASK tensors.
    Drop-in replacement for the built-in LoadImage node when the image
    data is available in memory (e.g. passed inline in the workflow JSON).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_data": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("image", "mask")
    FUNCTION      = "load"
    CATEGORY      = "CRT/Load"

    def load(self, base64_data: str):
        try:
            raw = base64.b64decode(base64_data) if base64_data.strip() else b""
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        arr  = np.array(img, dtype=np.float32) / 255.0          # H W 4

        rgb  = torch.from_numpy(arr[:, :, :3]).unsqueeze(0)     # 1 H W 3
        mask = torch.from_numpy(1.0 - arr[:, :, 3]).unsqueeze(0)  # 1 H W  (inverted alpha)

        return (rgb, mask)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "LoadImageBase64": LoadImageBase64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageBase64": "Load Image (Base64)",
}

import os
import re
from PIL import Image
import numpy as np
import torch


def natural_sort_key(filename):
    # Split filename into parts, converting numeric parts to integers for natural sorting
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


class CRTLoadLastMedia:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "X://insert/path/here"}),
                "sort_by": (["alphabetical", "date"], {"default": "date"}),
                "invert_order": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "CRT/Load"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, folder_path, sort_by, invert_order):
        # Validate folder path
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Define supported image extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        # Get all image files
        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)
        ]

        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")

        # Sort files
        if sort_by == "date":
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=invert_order)
        else:  # alphabetical with natural sorting
            image_files.sort(key=natural_sort_key, reverse=invert_order)

        # Get the last image
        last_file = os.path.join(folder_path, image_files[0])

        # Load the image
        image = Image.open(last_file).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, folder_path, sort_by, invert_order):
        # Return a unique value based on inputs and folder content
        if not os.path.exists(folder_path):
            return 0
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)
        ]
        if not image_files:
            return 0
        if sort_by == "date":
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        else:
            image_files.sort(key=natural_sort_key)
        if invert_order:
            image_files.reverse()
        return os.path.getmtime(os.path.join(folder_path, image_files[0]))

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return f"Invalid folder path: {folder_path}"
        return True


NODE_CLASS_MAPPINGS = {"CRTLoadLastMedia": CRTLoadLastMedia}

NODE_DISPLAY_NAME_MAPPINGS = {"CRTLoadLastMedia": "Load Last Image (CRT)"}

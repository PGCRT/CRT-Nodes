import os
import math
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers
import comfy.utils

MAX_RESOLUTION = 16384


class LoadImageResize:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 200.0, "step": 0.01},
                ),
                "multiple_of": (
                    "INT",
                    {"default": 16, "min": 1, "max": 512, "step": 1},
                ),
            }
        }

    CATEGORY = "CRT/Load"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "load_and_resize"

    def load_and_resize(self, image, megapixels, multiple_of=16):
        # Load image (from LoadImage node)
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            loaded_image = i.convert("RGB")

            if len(output_images) == 0:
                w = loaded_image.size[0]
                h = loaded_image.size[1]

            if loaded_image.size[0] != w or loaded_image.size[1] != h:
                continue

            loaded_image = np.array(loaded_image).astype(np.float32) / 255.0
            loaded_image = torch.from_numpy(loaded_image)[None,]

            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = (
                    np.array(i.convert("RGBA").getchannel("A")).astype(np.float32)
                    / 255.0
                )
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(loaded_image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Resize image using megapixels
        _, oh, ow, _ = output_image.shape
        aspect_ratio = ow / oh
        target_pixels = megapixels * 1000000

        new_h = math.sqrt(target_pixels / aspect_ratio)
        new_w = new_h * aspect_ratio

        if multiple_of > 1:
            width = round(new_w / multiple_of) * multiple_of
            height = round(new_h / multiple_of) * multiple_of
        else:
            width = round(new_w)
            height = round(new_h)

        # Resize using lanczos interpolation
        outputs = output_image.permute(0, 3, 1, 2)
        outputs = comfy.utils.lanczos(outputs, width, height)
        outputs = outputs.permute(0, 2, 3, 1)

        outputs = torch.clamp(outputs, 0, 1)

        return (outputs, output_mask, outputs.shape[2], outputs.shape[1])

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

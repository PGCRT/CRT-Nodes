import os
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
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "size": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                "multiple_of": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    CATEGORY = "CRT/Load"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "load_and_resize"

    def load_and_resize(self, image, size, multiple_of=0):
        # Load image (from LoadImage node)
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            loaded_image = i.convert("RGB")

            if len(output_images) == 0:
                w = loaded_image.size[0]
                h = loaded_image.size[1]

            if loaded_image.size[0] != w or loaded_image.size[1] != h:
                continue

            loaded_image = np.array(loaded_image).astype(np.float32) / 255.0
            loaded_image = torch.from_numpy(loaded_image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
            output_images.append(loaded_image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Resize image (adapted from ImageResize node)
        _, oh, ow, _ = output_image.shape

        # Calculate dimensions keeping aspect ratio with longer side = size
        if ow >= oh:  # Width is longer or equal
            width = size
            height = round(oh * (size / ow))
        else:  # Height is longer
            height = size
            width = round(ow * (size / oh))

        # Apply multiple_of constraint if specified
        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        # Resize using lanczos interpolation
        outputs = output_image.permute(0, 3, 1, 2)
        outputs = comfy.utils.lanczos(outputs, width, height)
        outputs = outputs.permute(0, 2, 3, 1)

        # Handle multiple_of constraint post-resize if needed
        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            actual_width = outputs.shape[2]
            actual_height = outputs.shape[1]
            x = (actual_width % multiple_of) // 2
            y = (actual_height % multiple_of) // 2
            x2 = actual_width - ((actual_width % multiple_of) - x)
            y2 = actual_height - ((actual_height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

        outputs = torch.clamp(outputs, 0, 1)

        return (outputs, output_mask, outputs.shape[2], outputs.shape[1])

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
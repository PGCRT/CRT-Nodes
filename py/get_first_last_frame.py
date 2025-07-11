# File: get_first_last_frame.py

import torch

class CRTFirstLastFrameSelector:
    """
    A node that takes a batch of images and returns the first and last image frames.
    If the batch contains only one image, it returns that image for both outputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("first_frame", "last_frame")
    FUNCTION = "get_frames"
    CATEGORY = "CRT/Video"

    def get_frames(self, images: torch.Tensor):
        # The input 'images' is a PyTorch tensor with shape (batch_size, height, width, channels)
        
        batch_size = images.shape[0]
        
        # Slicing to get the first frame. [0:1] keeps the batch dimension.
        first_frame = images[0:1]
        
        # Slicing to get the last frame. [-1:] also keeps the batch dimension.
        # This elegantly handles the case where batch_size is 1, as the first and last frame are the same.
        last_frame = images[-1:]

        print(f"[CRTFirstLastFrameSelector] Input batch size: {batch_size}. Extracted first and last frames.")

        return (first_frame, last_frame)

# --- MAPPINGS FOR ComfyUI ---
# This part is only used if the file is loaded directly by ComfyUI,
# not through an __init__.py file. Your __init__.py will handle this.
NODE_CLASS_MAPPINGS = {
    "CRTFirstLastFrameSelector": CRTFirstLastFrameSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTFirstLastFrameSelector": "Get First & Last Frame (CRT)"
}
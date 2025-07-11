import os
import torch
import numpy as np
from PIL import Image
import folder_paths

class SaveImageWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "image": ("IMAGE",),
                "folder_path": ("STRING", {"default": output_dir, "tooltip": "Base folder path. Defaults to ComfyUI's output folder."}),
                "subfolder_name": ("STRING", {"default": "images", "tooltip": "Subfolder name to create within the base folder."}),
                "filename": ("STRING", {"default": "output", "tooltip": "File name without extension."}),
                "extension": (["png", "jpg", "jpeg"], {"default": "png", "tooltip": "Image file extension."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    CATEGORY = "CRT/Save"
    DESCRIPTION = "Saves an image to a specified folder path with a subfolder and chosen extension."

    def save_image(self, image, folder_path, subfolder_name, filename, extension):
        if image is None:
            return ()
            
        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')

            if not subfolder_clean or not filename_clean:
                raise ValueError("Subfolder and Filename fields cannot be empty.")

            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, f"{filename_clean}.{extension}")

            # Convert tensor to PIL Image
            # image[0] takes the first image from the batch
            pil_img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

            # Save the image
            pil_img.save(final_filepath)
            print(f"✅ Saved image to: {final_filepath}")

            return ()
            
        except Exception as e:
            print(f"❌ ERROR in SaveImageWithPath: {str(e)}")
            raise e

# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SaveImageWithPath": SaveImageWithPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithPath": "Save Image With Path (CRT)"
}
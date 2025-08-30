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
                "filename": ("STRING", {"default": "output", "tooltip": "Base file name without extension. A suffix will be added for each image in a batch."}),
                "extension": (["png", "jpg", "jpeg"], {"default": "png", "tooltip": "Image file extension."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CATEGORY = "CRT/Save"
    DESCRIPTION = "Saves all images from a batch to a specified folder, adding a numerical suffix to each."

    def save_images(self, image, folder_path, subfolder_name, filename, extension):
        if image is None:
            return ()
            
        try:
            # --- Initial Setup and Cleaning ---
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')

            if not subfolder_clean or not filename_clean:
                raise ValueError("Subfolder and Filename fields cannot be empty.")

            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)

            # --- BATCH PROCESSING LOGIC ---
            batch_size = image.shape[0]
            
            for i in range(batch_size):
                # Determine the base filename for this specific image in the batch
                if batch_size > 1:
                    # If it's a batch, append a suffix like _1, _2, _3
                    base_filename = f"{filename_clean}_{i+1}"
                else:
                    # If it's a single image, just use the provided filename
                    base_filename = filename_clean
                
                # --- Overwrite Prevention Logic ---
                # Check if a file with this name already exists and add a counter if it does
                filepath_to_check = os.path.join(final_dir, f"{base_filename}.{extension}")
                final_filepath = filepath_to_check
                counter = 1
                while os.path.exists(final_filepath):
                    # If "output_1.png" exists, the next one will be "output_1_1.png"
                    final_filepath = os.path.join(final_dir, f"{base_filename}_{counter}.{extension}")
                    counter += 1

                # Convert the i-th tensor from the batch to a PIL Image
                # image[i] selects the current image from the batch
                pil_img = Image.fromarray((image[i].cpu().numpy() * 255).astype(np.uint8))

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
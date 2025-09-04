import os
import torch
from safetensors.torch import save_file

class SaveLatentWithPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "folder_path": ("STRING", {"default": "C:/tmp/loop"}),
                "filename": ("STRING", {"default": "latent"}),
                "subfolder_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "save_latent"
    CATEGORY = "CRT/Save"

    def save_latent(self, latent, folder_path, filename, subfolder_name):
        print(f"📌 SaveLatentWithPath: Starting execution")
        print(f"📌 Input parameters: folder_path={folder_path}, filename={filename}, subfolder_name={subfolder_name}")

        # Validate latent input
        if latent is None:
            print(f"⚠️ Received None as latent input. Skipping save and returning None.")
            return (None,)

        latent_tensor = None
        if isinstance(latent, dict) and "samples" in latent:
            latent_tensor = latent["samples"]
        elif isinstance(latent, torch.Tensor):
            latent_tensor = latent
        else:
            print(f"❌ Invalid latent input type: {type(latent)}. Expected dict with 'samples' or torch.Tensor. Skipping save.")
            return (None,)

        print(f"📌 Latent tensor shape: {latent_tensor.shape}")

        # Validate folder path
        full_output_folder = folder_path
        if subfolder_name:
            full_output_folder = os.path.join(folder_path, subfolder_name)
        
        try:
            os.makedirs(full_output_folder, exist_ok=True)
        except Exception as e:
            print(f"❌ Failed to create directory {full_output_folder}: {str(e)}")
            return (None,)

        # Ensure filename ends with .safetensors
        if not filename.endswith(".safetensors"):
            filename = f"{filename}.safetensors"
        final_filepath = os.path.join(full_output_folder, filename)
        print(f"📌 Saving to: {final_filepath}")

        # Check write access
        try:
            with open(final_filepath, 'wb') as f:
                pass  # Test write access
        except Exception as e:
            print(f"❌ Failed to access file {final_filepath}: {str(e)}. Cannot save latent.")
            return (None,)

        # Save the latent tensor
        try:
            save_file({"latent": latent_tensor}, final_filepath)
            print(f"📌 Successfully saved latent to {final_filepath}")
        except Exception as e:
            print(f"❌ Error saving latent to {final_filepath}: {str(e)}")
            return (None,)

        print("📌 SaveLatentWithPath: Execution complete")
        return (latent,)

    @classmethod
    def IS_CHANGED(cls, latent, folder_path, filename, subfolder_name, **kwargs):
        # Use a unique identifier based on inputs to detect changes
        full_output_folder = folder_path
        if subfolder_name:
            full_output_folder = os.path.join(folder_path, subfolder_name)
        final_filepath = os.path.join(full_output_folder, filename)
        if not filename.endswith(".safetensors"):
            final_filepath += ".safetensors"
        try:
            return os.path.getmtime(final_filepath)
        except:
            return 0

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, filename, subfolder_name, **kwargs):
        full_output_folder = folder_path
        if subfolder_name:
            full_output_folder = os.path.join(folder_path, subfolder_name)
        if not folder_path:
            return "Folder path cannot be empty"
        if not filename:
            return "Filename cannot be empty"
        try:
            os.makedirs(full_output_folder, exist_ok=True)
            test_filepath = os.path.join(full_output_folder, "test_access")
            with open(test_filepath, 'wb') as f:
                pass
            os.remove(test_filepath)
        except Exception as e:
            return f"Cannot access or create directory {full_output_folder}: {str(e)}"
        return True

NODE_CLASS_MAPPINGS = {
    "SaveLatentWithPath": SaveLatentWithPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentWithPath": "Save Latent With Path (CRT)"
}
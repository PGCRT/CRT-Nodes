import os
import torch
import re
from safetensors.torch import load_file

def natural_sort_key(filename):
    """Sort strings with numbers in a natural, human-friendly order."""
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

class LoadLastLatent:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "C:/path/to/latents"}),
                "sort_by": (["date", "alphabetical"], {"default": "date"}),
                "invert_order": ("BOOLEAN", {"default": False}),
                "pass_output": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "CRT/Load"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "load_last_latent"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """
        Defer all validation to the main execution method.
        This prevents the node from throwing an error and halting the queue
        if a folder doesn't exist, allowing for graceful fallback.
        """
        return True

    def load_last_latent(self, folder_path, sort_by, invert_order, pass_output):
        # --- Start of Fallback Logic ---

        # 1. Handle the 'pass_output' toggle
        if not pass_output:
            print("LoadLastLatent: pass_output is False, returning None.")
            return (None,)

        # 2. Validate the folder path
        if not folder_path or not os.path.isdir(folder_path):
            print(f"LoadLastLatent: Folder not found at '{folder_path}', returning None.")
            return (None,)

        # --- End of Fallback Logic ---

        # Supported latent extension
        latent_extension = 'safetensors'
        
        # Get all latent files from the validated folder
        try:
            latent_files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith(latent_extension)
            ]
        except Exception as e:
            print(f"LoadLastLatent: Error reading directory '{folder_path}': {e}. Returning None.")
            return (None,)

        # 3. Check if any latent files were found
        if not latent_files:
            print(f"LoadLastLatent: No '.safetensors' files found in '{folder_path}', returning None.")
            return (None,)

        # Sort files based on user preference
        if sort_by == "date":
            # Sort by modification time
            latent_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(folder_path, f)),
                reverse=not invert_order # Sort descending by default for date
            )
        else:
            # Sort alphabetically with natural sorting
            latent_files.sort(key=natural_sort_key, reverse=invert_order)

        # The desired file is the first one in the sorted list
        latent_filename = latent_files[0]
        latent_path = os.path.join(folder_path, latent_filename)

        # Load the latent using safetensors
        try:
            print(f"LoadLastLatent: Loading '{latent_filename}'...")
            latent_data = load_file(latent_path)
            latent_tensor = latent_data.get("latent")

            if latent_tensor is None:
                print(f"LoadLastLatent: ERROR - No 'latent' key found in safetensors file '{latent_filename}'.")
                return (None,)
            
            # Prepare the latent for ComfyUI
            latent_output = {"samples": latent_tensor.detach().clone()}
            
            print(f"LoadLastLatent: Successfully loaded latent from '{latent_filename}'.")
            return (latent_output,)

        except Exception as e:
            print(f"LoadLastLatent: ERROR - Failed to load file '{latent_filename}': {e}")
            return (None,)

    @classmethod
    def IS_CHANGED(cls, folder_path, **kwargs):
        """
        Check if the latest file in the directory has changed.
        This check is also designed to fail gracefully.
        """
        if not folder_path or not os.path.isdir(folder_path):
            return float("NaN") # Return a value that is always different

        # This part of the logic can remain similar, as it doesn't halt the queue
        try:
            latent_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.safetensors')]
            if not latent_files:
                return float("NaN")
            
            sort_by = kwargs.get("sort_by", "date")
            invert_order = kwargs.get("invert_order", False)

            if sort_by == "date":
                latent_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=not invert_order)
            else:
                latent_files.sort(key=natural_sort_key, reverse=invert_order)

            latest_file_path = os.path.join(folder_path, latent_files[0])
            return os.path.getmtime(latest_file_path)
        except:
            return float("NaN")
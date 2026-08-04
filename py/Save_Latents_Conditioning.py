import json
import os

from safetensors.torch import save_file

from ._latent_conditioning_codec import METADATA_KEY, encode_latent_conditioning


class SaveLatentsConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the SaveLatentsConditioning node.
        """
        return {
            "required": {
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "folder_path": ("STRING", {"default": "", "tooltip": "Base folder path to save the file"}),
                "subfolder_name": ("STRING", {"default": "", "tooltip": "Subfolder name within the base folder"}),
                "filename": (
                    "STRING",
                    {"default": "", "tooltip": "File name (without extension). Leave empty to number files incrementally starting at 0."},
                ),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix appended to filename."}),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, existing files will be overwritten. If disabled, a numbered suffix like _001 is added.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latents_conditioning"
    CATEGORY = "CRT/Save"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves a latent and a conditioning together in a single .safetensors file. Load them back with 'Load Latents Conditioning (CRT)'."

    def save_latents_conditioning(self, latent, conditioning, folder_path, subfolder_name, filename, suffix, overwrite=True):
        """
        Saves the latent samples and the conditioning tensors into one .safetensors file.
        """
        # Construct the full directory path
        full_folder_path = os.path.join(folder_path, subfolder_name.strip().lstrip("/\\"))

        # Create the directory structure if it doesn't exist
        if full_folder_path and not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path, exist_ok=True)

        filename_clean = filename.strip()
        suffix_clean = suffix.strip()

        if filename_clean:
            # Keep the stem separate so numbered copies are always inserted before .safetensors.
            filename_stem = f"{filename_clean}{suffix_clean}"
            full_path = os.path.join(full_folder_path, f"{filename_stem}.safetensors")
            if not overwrite:
                counter = 1
                while os.path.exists(full_path):
                    numbered_filename = f"{filename_stem}_{counter:03d}.safetensors"
                    full_path = os.path.join(full_folder_path, numbered_filename)
                    counter += 1
        else:
            # Empty filename: number files incrementally starting at 0.
            counter = 0
            full_path = os.path.join(full_folder_path, f"{counter}{suffix_clean}.safetensors")
            while os.path.exists(full_path):
                counter += 1
                full_path = os.path.join(full_folder_path, f"{counter}{suffix_clean}.safetensors")

        try:
            tensors, meta = encode_latent_conditioning(latent, conditioning)
            save_file(tensors, full_path, metadata={METADATA_KEY: json.dumps(meta)})
            print(f"[OK] Saved latents + conditioning to: {full_path}")
        except Exception as e:
            print(f"[ERROR] Error saving latents + conditioning to {full_path}: {e}")

        return ()


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {"SaveLatentsConditioning": SaveLatentsConditioning}

NODE_DISPLAY_NAME_MAPPINGS = {"SaveLatentsConditioning": "Save Latents Conditioning (CRT)"}

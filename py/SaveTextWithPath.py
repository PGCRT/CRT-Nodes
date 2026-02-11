import os


class SaveTextWithPath:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the SaveTextWithPath node.
        """
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "folder_path": ("STRING", {"default": "", "tooltip": "Base folder path to save the text file"}),
                "subfolder_name": ("STRING", {"default": "", "tooltip": "Subfolder name within the base folder"}),
                "filename": (
                    "STRING",
                    {"default": "output", "tooltip": "File name for the text file (without extension)"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    CATEGORY = "CRT/Save"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves a text string to a specified folder path with a subfolder as a .txt file."

    def save_text(self, text, folder_path, subfolder_name, filename):
        """
        Saves the provided text to a .txt file with UTF-8 encoding.
        """
        # Construct the full directory path
        full_folder_path = os.path.join(folder_path, subfolder_name)

        # Create the directory structure if it doesn't exist
        if full_folder_path and not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path, exist_ok=True)

        # Ensure the filename is clean and add the .txt extension
        clean_filename = f"{filename}.txt"

        # Construct the full file path
        full_path = os.path.join(full_folder_path, clean_filename)

        try:
            # Write the text to the file with UTF-8 encoding
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Saved text to: {full_path}")
        except Exception as e:
            print(f"❌ Error saving text to {full_path}: {e}")

        return ()


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {"SaveTextWithPath": SaveTextWithPath}

NODE_DISPLAY_NAME_MAPPINGS = {"SaveTextWithPath": "Save Text With Path (CRT)"}

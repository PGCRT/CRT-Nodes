import os
import secrets


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
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix appended to filename."}),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, existing files will be overwritten. If disabled, a numbered suffix like _001 is added.",
                    },
                ),
                "extension": (
                    [".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".html"],
                    {"default": ".txt", "tooltip": "File extension for the saved text file."},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    CATEGORY = "CRT/Save"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    DESCRIPTION = "Saves a text string to a specified folder path with a subfolder, with a selectable file extension. Accepts batched strings for text, filename and folder_path, paired item by item."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NaN != NaN, so the save re-runs on every queue instead of being skipped from cache
        return float("NaN")

    def save_text(self, text, folder_path, subfolder_name, filename, suffix=None, overwrite=None, extension=None):
        """
        Saves the provided text(s) to file(s) with UTF-8 encoding.
        With INPUT_IS_LIST every input arrives as a list; widget values are
        single-element lists, batched inputs keep their batch length.
        """
        texts = text if isinstance(text, list) else [text]
        filenames = filename if isinstance(filename, list) else [filename]
        folders = folder_path if isinstance(folder_path, list) else [folder_path]
        subfolder_value = subfolder_name[0] if isinstance(subfolder_name, list) else subfolder_name
        suffix_value = (suffix or [""])[0] if isinstance(suffix, list) else (suffix or "")
        overwrite_value = (overwrite if overwrite is not None else [True])[0] if isinstance(overwrite, list) else (overwrite if overwrite is not None else True)
        extension_value = (extension or [".txt"])[0] if isinstance(extension, list) else (extension or ".txt")

        for index, item in enumerate(texts):
            if index < len(filenames):
                name = filenames[index]
            else:
                name = f"{filenames[-1].strip()}_{index:03d}" if filenames else ""
            folder_value = folders[index] if index < len(folders) else folders[-1]
            self._save_one(
                item,
                folder_value,
                subfolder_value,
                name,
                suffix_value,
                overwrite_value,
                extension_value,
            )

        return ()

    @staticmethod
    def _save_one(text, folder_path, subfolder_name, filename, suffix, overwrite, extension):
        # Normalize the extension (accepts "md" or ".md")
        extension_clean = extension.strip()
        if not extension_clean:
            extension_clean = ".txt"
        if not extension_clean.startswith("."):
            extension_clean = f".{extension_clean}"
        # Empty subfolder: save directly into the base folder.
        subfolder_clean = subfolder_name.strip().lstrip("/\\")
        full_folder_path = (
            os.path.join(folder_path, subfolder_clean)
            if subfolder_clean
            else folder_path
        )

        # Create the directory structure if it doesn't exist
        if full_folder_path and not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path, exist_ok=True)

        # Empty filename: auto-generate a unique name (never block the save).
        filename_clean = filename.strip()
        if not filename_clean:
            filename_clean = f"text_{secrets.token_hex(16)}"
            print(f"[INFO] No filename given. Using auto-generated name: {filename_clean}")

        # Keep the stem separate so numbered copies are always inserted before the extension.
        filename_stem = f"{filename_clean}{suffix.strip()}"
        clean_filename = f"{filename_stem}{extension_clean}"

        # Construct the full file path
        full_path = os.path.join(full_folder_path, clean_filename)

        if not overwrite and os.path.exists(full_path):
            counter = 1
            while os.path.exists(full_path):
                numbered_filename = f"{filename_stem}_{counter:03d}{extension_clean}"
                full_path = os.path.join(full_folder_path, numbered_filename)
                counter += 1

        try:
            # Write the text to the file with UTF-8 encoding
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[OK] Saved text to: {full_path}")
        except Exception as e:
            print(f"[ERROR] Error saving text to {full_path}: {e}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {"SaveTextWithPath": SaveTextWithPath}

NODE_DISPLAY_NAME_MAPPINGS = {"SaveTextWithPath": "Save Text With Path (CRT)"}

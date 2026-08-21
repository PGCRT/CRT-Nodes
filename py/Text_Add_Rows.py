import os
import folder_paths


class TextAddRows:
    """Append the incoming text as a new row to a persistent .txt file."""

    def __init__(self):
        self._active_path = None
        self._last_base_key = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "The text to append as one row."}),
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder path where the .txt file is saved. If empty, uses ComfyUI's output directory.",
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Filename for the .txt file. If empty, uses Text_Add_Rows.txt. If a file with that name already exists, a numeric suffix is appended.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "file_content")
    OUTPUT_TOOLTIPS = (
        "Absolute path to the file that was appended to.",
        "Full current content of the file as a string.",
    )
    FUNCTION = "add_row"
    CATEGORY = "CRT/Text"
    OUTPUT_NODE = True
    DESCRIPTION = "Appends each incoming string as one row to a persistent .txt file, avoiding overwrites by adding a numeric suffix when needed. Also outputs the file's current content."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-run on every queue so the row is always appended.
        return float("NaN")

    def _resolve_base_path(self, path, filename):
        path = (path or "").strip()
        filename = (filename or "").strip()

        if not path:
            directory = folder_paths.get_output_directory()
        else:
            directory = path
            if os.path.isfile(directory):
                directory = os.path.dirname(directory)

        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not filename:
            filename = "Text_Add_Rows.txt"

        if not os.path.splitext(filename)[1]:
            filename += ".txt"

        return os.path.normpath(os.path.join(directory, filename))

    def _find_unique_path(self, base_path):
        if not os.path.exists(base_path):
            return base_path

        stem, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            candidate = f"{stem}_{counter}{ext}"
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    def add_row(self, text, path="", filename=""):
        base_path = self._resolve_base_path(path, filename)

        if (
            self._active_path is None
            or self._last_base_key != base_path
            or not os.path.exists(self._active_path)
        ):
            self._active_path = self._find_unique_path(base_path)
            self._last_base_key = base_path

        row = str(text) if text is not None else ""
        # Normalize to a single line so each execution adds exactly one row.
        row = row.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
        row += "\n"

        with open(self._active_path, "a", encoding="utf-8") as f:
            f.write(row)

        print(f"[OK] Appended row to: {self._active_path}")

        try:
            with open(self._active_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception:
            file_content = ""

        return (self._active_path, file_content)


NODE_CLASS_MAPPINGS = {"TextAddRows": TextAddRows}
NODE_DISPLAY_NAME_MAPPINGS = {"TextAddRows": "Text Add Rows (CRT)"}

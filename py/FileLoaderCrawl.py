import os
from pathlib import Path


class FileLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing text files"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for deterministic file selection",
                    },
                ),
                "file_extension": (
                    "STRING",
                    {"default": ".txt", "tooltip": "File extension to filter (e.g., .txt, .md)"},
                ),
                "max_words": (
                    "INT",
                    {"default": 0, "min": 0, "tooltip": "Maximum number of words in output (0 for no limit)"},
                ),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "If true, include files in subfolders"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_output", "file_name")
    FUNCTION = "load_text_file"
    CATEGORY = "CRT/Load"

    def limit_words(self, text, max_words):
        """Limit the text to a specified number of words."""
        if max_words <= 0:
            return text
        words = text.split()
        return ' '.join(words[:max_words])

    def load_text_file(self, folder_path, seed, file_extension, max_words, crawl_subfolders):
        # Define a safe, empty return value for error cases
        safe_return = ("", "")

        if not folder_path or not folder_path.strip():
            print("❌ Error: Folder path is empty.")
            return safe_return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"❌ Error: Folder '{folder}' not found or is not a directory.")
            return safe_return

        file_extension = file_extension.strip().lower()
        if file_extension and not file_extension.startswith('.'):
            file_extension = f".{file_extension}"

        try:
            # --- Smart Caching Logic ---
            cache_key = f"{str(folder.resolve())}_{crawl_subfolders}_{file_extension}"
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"🔎 Folder changed or not cached. Scanning '{folder}' for '{file_extension}' files...")
                if crawl_subfolders:
                    files = sorted([f for f in folder.rglob(f'*{file_extension}') if f.is_file()])
                else:
                    # Use glob for simpler filtering
                    files = sorted([f for f in folder.glob(f'*{file_extension}') if f.is_file()])

                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"✅ Cached {len(files)} files.")

            files = self.cache[cache_key]['files']
            # --- End Caching Logic ---

            if not files:
                print(f"❌ Warning: No files with extension '{file_extension}' found in '{folder}'.")
                return safe_return

            # --- Deterministic and Safe Selection ---
            num_files = len(files)
            selected_index = seed % num_files
            selected_file = files[selected_index]
            # --- End Selection ---

            print(f"✅ Seed {seed} → File {selected_index + 1}/{num_files}: '{selected_file.name}'")

            with open(selected_file, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

            limited_content = self.limit_words(content, max_words)
            return (limited_content, selected_file.name)

        except Exception as e:
            print(f"❌ An unexpected error occurred in FileLoaderCrawl: {str(e)}")
            return safe_return

import os
import random
from pathlib import Path

class FileLoaderCrawl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing text files"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic file selection"}),
                "file_extension": ("STRING", {"default": ".txt", "tooltip": "File extension to filter (e.g., .txt, .md)"}),
                "max_words": ("INT", {"default": 0, "min": 0, "tooltip": "Maximum number of words in output (0 for no limit)"}),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "If true, include files in subfolders"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_output", "file_name")
    FUNCTION = "load_text_file"
    CATEGORY = "CRT"
    DESCRIPTION = "Crawls a folder (optionally including subfolders) and loads the content of a text file selected deterministically using a seed, with optional word count limit."

    def limit_words(self, text, max_words):
        """Limit the text to a specified number of words."""
        if max_words <= 0:
            return text
        words = text.split()
        return ' '.join(words[:max_words])

    def load_text_file(self, folder_path, seed, file_extension, max_words, crawl_subfolders):
        folder = Path(folder_path.strip())
        if not folder.exists():
            return (f"Error: Folder '{folder}' does not exist", "")
        if not folder.is_dir():
            return (f"Error: '{folder}' is not a directory", "")
        file_extension = file_extension.strip().lower()
        if not file_extension.startswith('.'):
            file_extension = f".{file_extension}"
        try:
            if crawl_subfolders:
                files = sorted([f for f in folder.rglob(f'*{file_extension}') if f.is_file()])
            else:
                files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == file_extension])
        except Exception as e:
            return (f"Error accessing folder: {str(e)}", "")

        if not files:
            return (f"Error: No files with extension '{file_extension}' found in '{folder}'", "")
        random.seed(seed)
        selected_file = random.choice(files)
        try:
            with open(selected_file, 'r', encoding='utf-8') as file:
                content = file.read()
            limited_content = self.limit_words(content, max_words)
            return (limited_content, selected_file.name)
        except Exception as e:
            return (f"Error reading file '{selected_file.name}': {str(e)}", selected_file.name)
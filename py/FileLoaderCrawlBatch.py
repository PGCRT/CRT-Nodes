import os
from pathlib import Path
import random
import re # Required for natural sorting

class FileLoaderCrawlBatch:

    @staticmethod
    def natural_sort_key(s):
        """A key for sorting strings in a 'natural' order, e.g., '2.txt' before '10.txt'."""
        s = s.name
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing the text files"}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of files to load in the batch"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Acts as a batch offset. Set to 0 to start from the first file."}),
                "file_extension": ("STRING", {"default": ".txt", "tooltip": "The file extension to filter for (e.g., .txt)"}),
                "max_words": ("INT", {"default": 0, "min": 0, "tooltip": "Maximum number of words per output (0 for no limit)"}),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "Whether to include files in subfolders"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output_1",)
    FUNCTION = "load_text_files_batch"
    CATEGORY = "CRT/Load"

    def limit_words(self, text, max_words):
        if max_words <= 0: return text
        return ' '.join(text.split()[:max_words])

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run the node to respond to seed changes
        return True 

    def load_text_files_batch(self, folder_path, batch_count, seed, file_extension, max_words, crawl_subfolders):
        safe_return = tuple([""] * (batch_count * 2))
        if not folder_path or not Path(folder_path).is_dir():
            print(f"❌ Error: Folder '{folder_path}' not found or is not a directory.")
            return safe_return

        try:
            # Dynamically set return types for the UI based on batch_count
            self.__class__.RETURN_TYPES = ("STRING",) * (batch_count * 2)
            self.__class__.RETURN_NAMES = tuple(
                [f"text_output_{i+1}" for i in range(batch_count)] + 
                [f"file_name_{i+1}" for i in range(batch_count)]
            )
            
            # Scan and naturally sort the files
            folder = Path(folder_path)
            file_ext = f".{file_extension.strip().lstrip('.').lower()}"
            if crawl_subfolders:
                file_list = [f for f in folder.rglob(f'*{file_ext}') if f.is_file()]
            else:
                file_list = [f for f in folder.glob(f'*{file_ext}') if f.is_file()]
            
            all_files = sorted(file_list, key=self.natural_sort_key)

            if not all_files:
                print(f"❌ Warning: No files with extension '{file_ext}' found.")
                return safe_return
            
            # --- Batch Selection Logic (Increment Mode Only) ---
            selected_files = []
            num_available = len(all_files)
            
            start_index = (seed * batch_count) % num_available
            print(f"▶️ Loading Batch: Seed {seed} -> Start Index {start_index}")
            for i in range(batch_count):
                current_index = (start_index + i) % num_available
                selected_files.append(all_files[current_index])

            # --- Read Files and Prepare Outputs ---
            outputs, file_names = [], []
            for selected_file in selected_files:
                try:
                    with open(selected_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    outputs.append(self.limit_words(content, max_words))
                    file_names.append(selected_file.name)
                except Exception as e:
                    outputs.append(""); file_names.append(f"ERROR: {e}")
            
            # Pad outputs if necessary (e.g., if files fail to read)
            while len(outputs) < batch_count:
                outputs.append(""); file_names.append("")

            return tuple(outputs + file_names)

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return safe_return
import os
from pathlib import Path
import random
import torch
import re

class CRT_FileBatchPromptScheduler:

    @staticmethod
    def natural_sort_key(s):
        s = s.name
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing the text files"}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of prompts to load and encode per run"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Acts as a batch offset. Set to 0 to start from the first file."}),
                "file_extension": ("STRING", {"default": ".txt", "tooltip": "The file extension to look for (e.g., .txt)"}),
                "max_words": ("INT", {"default": 0, "min": 0, "tooltip": "Maximum number of words per prompt (0 for no limit)"}),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "Whether to include files in subfolders"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "batch_count")
    FUNCTION = "schedule_from_files"
    CATEGORY = "CRT/Conditioning"

    def limit_words(self, text, max_words):
        if max_words <= 0: return text
        return ' '.join(text.split()[:max_words])

    def schedule_from_files(self, clip, folder_path, batch_count, seed, file_extension, max_words, crawl_subfolders):
        prompts = [""] # Default return value
        
        if not folder_path or not Path(folder_path).is_dir():
            print(f"❌ Error: Folder '{folder_path}' not found or is not a directory. Using a single empty prompt.")
        else:
            try:
                # --- File Scanning and Natural Sorting ---
                folder = Path(folder_path)
                file_ext = f".{file_extension.strip().lstrip('.').lower()}"
                print(f"🔎 Scanning and sorting files in '{folder_path}'...")
                
                if crawl_subfolders:
                    file_list = [f for f in folder.rglob(f'*{file_ext}') if f.is_file()]
                else:
                    file_list = [f for f in folder.glob(f'*{file_ext}') if f.is_file()]
                
                all_files = sorted(file_list, key=self.natural_sort_key)

                if not all_files:
                    print(f"❌ Warning: No '{file_ext}' files found. Using a single empty prompt.")
                else:
                    # --- Batch Selection Logic (Increment Mode Only) ---
                    selected_files = []
                    num_available = len(all_files)
                    
                    start_index = (seed * batch_count) % num_available
                    print(f"▶️ Loading Batch: Seed {seed} -> Start Index {start_index}")
                    for i in range(batch_count):
                        current_index = (start_index + i) % num_available
                        selected_files.append(all_files[current_index])
                    
                    # --- Read Files into a Prompt List ---
                    loaded_prompts = []
                    for selected_file in selected_files:
                        try:
                            with open(selected_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            loaded_prompts.append(self.limit_words(content, max_words))
                        except Exception as e:
                            print(f"❌ Error reading file '{selected_file.name}': {e}. Skipping.")
                    
                    if loaded_prompts:
                        prompts = loaded_prompts

            except Exception as e:
                print(f"❌ An unexpected error occurred during file loading: {e}. Using a single empty prompt.")
        
        # --- Conditioning Logic ---
        final_batch_count = len(prompts)
        print(f"✅ Encoding {final_batch_count} prompts...")
        
        cond_list, pooled_list = [], []
        has_pooled_output = True

        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            if pooled is None: has_pooled_output = False
            pooled_list.append(pooled)

        final_cond = torch.cat(cond_list, dim=0)
        conditioning_extras = {}

        if has_pooled_output:
            try:
                # Filter out None values before concatenating
                valid_pooled = [p for p in pooled_list if p is not None]
                if valid_pooled:
                    final_pooled = torch.cat(valid_pooled, dim=0)
                    conditioning_extras["pooled_output"] = final_pooled
            except Exception as e:
                print(f"[Warning] Could not concatenate pooled_outputs: {e}")

        return ([[final_cond, conditioning_extras]], final_batch_count)
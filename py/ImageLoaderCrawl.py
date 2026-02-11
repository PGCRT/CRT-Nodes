import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np


class ImageLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "remove_extension": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image_output", "file_name", "total_images")
    FUNCTION = "load_image_incrementally"
    CATEGORY = "CRT/Load"

    def load_image_incrementally(self, folder_path, seed, crawl_subfolders, remove_extension):
        # Create a blank image as fallback
        def create_blank_image():
            blank = np.zeros((512, 512, 3), dtype=np.float32)
            return torch.from_numpy(blank)[None,]

        if not folder_path or not folder_path.strip():
            return (create_blank_image(), "Error: Folder path is empty", 0)

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"❌ Error: Folder '{folder}' not found.")
            return (create_blank_image(), "Error: Folder not found", 0)

        # --- Smart Caching Logic ---
        cache_key = str(folder.resolve()) + ("_sub" if crawl_subfolders else "")
        current_mtime = folder.stat().st_mtime

        # Check if cache is invalid (key doesn't exist or modification time has changed)
        if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
            print(f"🔎 Folder changed or not cached. Scanning '{folder}'...")
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.ti', '.gi', '.webp'}
            try:
                path_iterator = folder.rglob('*') if crawl_subfolders else folder.glob('*')
                files = sorted([p for p in path_iterator if p.is_file() and p.suffix.lower() in valid_extensions])

                # Update the cache with the new file list and the current modification time
                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"✅ Cached {len(files)} files from '{folder}'")
            except Exception as e:
                print(f"❌ Error accessing folder '{folder}': {str(e)}")
                # Clear bad cache entry if it exists
                if cache_key in self.cache:
                    del self.cache[cache_key]
                return (create_blank_image(), "Error accessing folder", 0)

        # Retrieve the list of files from the (now guaranteed to be up-to-date) cache
        files = self.cache[cache_key]['files']

        if not files:
            print(f"❌ Warning: No valid image files found in '{folder}'.")
            return (create_blank_image(), "No images found", 0)

        num_files = len(files)
        selected_index = seed % num_files
        selected_file = files[selected_index]

        try:
            with Image.open(selected_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]

            base_name = selected_file.stem if remove_extension else selected_file.name
            print(f"✅ Seed {seed} → Image {selected_index + 1}/{num_files}: '{base_name}' from '{selected_file.name}'")

            return (img_tensor, base_name, num_files)
        # Self-healing: If a file is in the cache but was deleted just before loading, this will catch it.
        except FileNotFoundError:
            print(
                f"❌ Error: File '{selected_file}' was in cache but not found on disk. Invalidating cache for next run."
            )
            # Forcing a rescan on the next execution by removing the invalid cache entry.
            if cache_key in self.cache:
                del self.cache[cache_key]
            return (create_blank_image(), "Error: Cached file not found", 0)
        except Exception as e:
            print(f"❌ Error loading image '{selected_file}': {str(e)}")
            return (create_blank_image(), "Error loading image", 0)


# Node mappings
NODE_CLASS_MAPPINGS = {"ImageLoaderCrawl": ImageLoaderCrawl}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageLoaderCrawl": "Image Loader Crawl (Smart)"}

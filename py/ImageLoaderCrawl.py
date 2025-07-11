import os
import random
from pathlib import Path
import torch
from PIL import Image
import numpy as np

class ImageLoaderCrawl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing image files"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic file selection"}),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "If true, include files in subfolders"}),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "If true, remove file extension from the output file name"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_output", "file_name")
    FUNCTION = "load_image_file"
    CATEGORY = "CRT/Load"
    DESCRIPTION = "Crawls a folder (optionally including subfolders) and loads an image file selected deterministically using a seed. Supports all common image formats."

    def load_image_file(self, folder_path, seed, crawl_subfolders, remove_extension):
        folder = Path(folder_path.strip())
        if not folder.exists():
            return (None, f"Error: Folder '{folder}' does not exist")
        if not folder.is_dir():
            return (None, f"Error: '{folder}' is not a directory")
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.ico', '.ppm', '.pgm', '.pbm'}
        try:
            if crawl_subfolders:
                files = []
                for ext in valid_extensions:
                    files.extend([f for f in folder.rglob(f'*{ext}') if f.is_file()])
                    files.extend([f for f in folder.rglob(f'*{ext.upper()}') if f.is_file()])  # Include uppercase
            else:
                files = []
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in valid_extensions:
                        files.append(file)
            files = sorted(list(set(files)))
            
        except Exception as e:
            return (None, f"Error accessing folder '{folder}': {str(e)}")

        if not files:
            supported_exts = ', '.join(sorted(valid_extensions))
            return (None, f"Error: No image files found in '{folder}'. Supported formats: {supported_exts}")
        random.seed(seed)
        selected_file = random.choice(files)
        try:
            with Image.open(selected_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]
            
            output_file_name = selected_file.stem if remove_extension else str(selected_file)
            print(f"✅ ImageLoaderCrawl: Loaded '{selected_file.name}' ({selected_file.suffix.lower()}) from {len(files)} available images")
            return (img_tensor, output_file_name)
            
        except Exception as e:
            return (None, f"Error loading image '{selected_file}': {str(e)}")
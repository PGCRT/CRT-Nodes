import os
import random
from pathlib import Path
import torch
import cv2
import numpy as np

class VideoLoaderCrawl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\videos", "tooltip": "Path to video files"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "Remove file extension from output name"}),
                "frames_limit": ("INT", {"default": 10, "min": -1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_output", "file_name")
    FUNCTION = "load_video_file"
    CATEGORY = "CRT/Load"

    def load_video_file(self, folder_path, seed, crawl_subfolders, remove_extension, frames_limit):
        folder = Path(folder_path.strip())
        if not folder.exists() or not folder.is_dir():
            print(f"❌ Folder error: {folder} not found")
            return (None, f"Error: Invalid folder {folder}")

        valid_extensions = {'.mp4', '.webm', '.mkv'}
        files = []
        try:
            if crawl_subfolders:
                files.extend(folder.rglob(f'*{ext}') for ext in valid_extensions)
            else:
                files.extend(f for f in folder.iterdir() if f.suffix.lower() in valid_extensions)
            files = sorted(set(files))
        except Exception as e:
            print(f"❌ Access error: {str(e)}")
            return (None, f"Error accessing folder: {str(e)}")

        if not files:
            print("❌ No video files found")
            return (None, "No video files found")

        random.seed(seed)
        selected_file = random.choice(files)
        print(f"📥 Loading: {selected_file}")

        try:
            cap = cv2.VideoCapture(str(selected_file))
            if not cap.isOpened():
                print(f"❌ Video open failed: {selected_file}")
                return (None, f"Error: Could not open {selected_file}")

            frames = []
            count = 0
            while cap.isOpened() and (frames_limit == -1 or count < frames_limit):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                count += 1

            cap.release()
            if not frames:
                print(f"❌ No frames: {selected_file}")
                return (None, f"No frames from {selected_file}")

            img_tensor = torch.from_numpy(np.stack(frames))
            file_name = selected_file.stem if remove_extension else str(selected_file)
            print(f"✅ Loaded {selected_file.name} with {len(frames)} frames")
            return (img_tensor, file_name)

        except Exception as e:
            print(f"❌ Load error: {str(e)}")
            return (None, f"Error loading video: {str(e)}")
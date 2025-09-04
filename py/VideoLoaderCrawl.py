import os
from pathlib import Path
import torch
import cv2
import numpy as np

class VideoLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\videos", "tooltip": "Path to video files"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "Remove file extension from output name"}),
                "frames_limit": ("INT", {"default": 16, "min": -1, "max": 10000, "tooltip": "Max frames to load. -1 for all frames."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_output", "file_name")
    FUNCTION = "load_video_file"
    CATEGORY = "CRT/Load"

    def load_video_file(self, folder_path, seed, crawl_subfolders, remove_extension, frames_limit):
        # Create a blank image batch as a fallback to prevent crashes
        def create_blank_output():
            blank_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (blank_frame, "Error: See console for details")

        if not folder_path or not folder_path.strip():
            print("❌ Error: Folder path is empty.")
            return create_blank_output()

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"❌ Error: Folder '{folder}' not found.")
            return create_blank_output()

        # --- Smart Caching Logic ---
        cache_key = str(folder.resolve()) + ("_sub" if crawl_subfolders else "")
        current_mtime = folder.stat().st_mtime

        if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
            print(f"🔎 Folder changed or not cached. Scanning '{folder}' for videos...")
            valid_extensions = {'.mp4', '.webm', '.mkv', '.avi', '.mov'}
            try:
                path_iterator = folder.rglob('*') if crawl_subfolders else folder.glob('*')
                files = sorted([f for f in path_iterator if f.is_file() and f.suffix.lower() in valid_extensions])
                
                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"✅ Cached {len(files)} video files from '{folder}'")
            except Exception as e:
                print(f"❌ Error accessing folder '{folder}': {str(e)}")
                if cache_key in self.cache:
                    del self.cache[cache_key]
                return create_blank_output()

        files = self.cache[cache_key]['files']
        
        if not files:
            print(f"❌ Warning: No valid video files found in '{folder}'.")
            return create_blank_output()

        # --- Deterministic File Selection using Seed ---
        num_files = len(files)
        selected_index = seed % num_files
        selected_file = files[selected_index]
        
        print(f"📥 Seed {seed} → Video {selected_index + 1}/{num_files}: '{selected_file.name}'")

        try:
            cap = cv2.VideoCapture(str(selected_file))
            if not cap.isOpened():
                print(f"❌ Video open failed: {selected_file}")
                return create_blank_output()

            frames = []
            count = 0
            while cap.isOpened() and (frames_limit == -1 or count < frames_limit):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert frame from BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1] float32
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
                count += 1
            cap.release()

            if not frames:
                print(f"❌ No frames could be read from video: {selected_file}")
                return create_blank_output()

            # Stack frames into a single tensor (batch of images)
            video_tensor = torch.from_numpy(np.stack(frames))
            file_name = selected_file.stem if remove_extension else selected_file.name
            
            print(f"✅ Loaded {selected_file.name} with {len(frames)} frames")
            return (video_tensor, file_name)

        except FileNotFoundError:
             print(f"❌ Error: File '{selected_file}' was in cache but not found on disk. Invalidating cache for next run.")
             if cache_key in self.cache:
                del self.cache[cache_key]
             return create_blank_output()
        except Exception as e:
            print(f"❌ Error loading video '{selected_file}': {str(e)}")
            return create_blank_output()
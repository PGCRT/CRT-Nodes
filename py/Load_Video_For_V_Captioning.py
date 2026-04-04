import os
from pathlib import Path
import cv2
import torch
import numpy as np

# A list of common video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpeg'}


class LoadVideoForVCaptioning:
    """
    An optimized and safe node to load videos from a directory sequentially.
    It caches the file list and automatically rescans if the folder changes.
    It can load all frames or a specified number of evenly-spaced frames efficiently.
    """

    def __init__(self):
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "C:/videos"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "num_evenly_spaced_frames": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Number of evenly spaced frames to load. 0 or -1 loads all frames.",
                    },
                ),
            }
        }

    # The corrected return types and names
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "VIDEO_PATH")
    RETURN_NAMES = ("image", "filename", "frame_count", "video_path")

    FUNCTION = "load_video_optimized"
    CATEGORY = "CRT/Load"

    def load_video_optimized(self, directory: str, index: int, num_evenly_spaced_frames: int):
        def create_blank_output():
            blank_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (blank_frame, "Error: See console", 0, "")

        try:
            folder = Path(directory)
            if not folder.is_dir():
                print(f"❌ Error: Directory not found: {directory}")
                return create_blank_output()

            cache_key = str(folder.resolve())
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"🔎 Folder changed or not cached. Scanning '{folder}' for videos...")
                files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"✅ Cached {len(files)} video files.")

            files = self.cache[cache_key]['files']

            if not files:
                print(f"❌ Warning: No video files found in {directory}")
                return create_blank_output()

            video_path = files[index % len(files)]
            filename_without_ext = video_path.stem

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file: {video_path}")
                return create_blank_output()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"❌ Warning: Video file seems empty or corrupted: {video_path}")
                cap.release()
                return create_blank_output()

            indices_to_load = set()
            if num_evenly_spaced_frames <= 0:
                indices_to_load = set(range(total_frames))
            else:
                num_to_pick = min(total_frames, num_evenly_spaced_frames)
                indices = np.linspace(0, total_frames - 1, num=num_to_pick, dtype=int)
                indices_to_load = set(indices)

            frames = []
            frame_count = 0
            ret = True

            while ret and len(frames) < len(indices_to_load):
                ret, frame = cap.read()
                if ret:
                    if frame_count in indices_to_load:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
                        frames.append(frame_tensor)
                    frame_count += 1

            cap.release()

            if not frames:
                print(f"❌ Error: Could not read any frames from the video: {video_path}")
                return create_blank_output()

            batch_tensor = torch.stack(frames)
            print(f"✅ Loaded {len(frames)} frames from '{video_path.name}'")

            return (batch_tensor, filename_without_ext, len(frames), str(video_path))

        except Exception as e:
            print(f"An unexpected error occurred in LoadVideo: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return create_blank_output()

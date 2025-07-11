import os
import cv2
import torch
import numpy as np

# A list of common video file extensions
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpeg')

class LoadVideoForVCaptioning:
    """
    An optimized node to load videos from a directory sequentially.
    It can load all frames or a specified number of evenly-spaced frames efficiently.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "C:/videos"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                # Renamed for clarity
                "num_evenly_spaced_frames": ("INT", {
                    "default": 0, 
                    "min": -1, 
                    "max": 256, 
                    "step": 1,
                    "display": "integer" 
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "frame_count")
    FUNCTION = "load_video_optimized"
    CATEGORY = "CRT/Load"

    def load_video_optimized(self, directory: str, index: int, num_evenly_spaced_frames: int):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        try:
            files = sorted([f for f in os.listdir(directory) if f.lower().endswith(VIDEO_EXTENSIONS)])
            if not files:
                raise FileNotFoundError(f"No video files with extensions {VIDEO_EXTENSIONS} found in {directory}")
        except Exception as e:
            raise IOError(f"Error reading directory {directory}: {e}")

        video_filename = files[index % len(files)]
        video_path = os.path.join(directory, video_filename)
        filename_without_ext = os.path.splitext(video_filename)[0]

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Video file seems empty or corrupted: {video_path}")

            # --- OPTIMIZATION & FEATURE LOGIC ---
            
            indices_to_load = set()
            if num_evenly_spaced_frames <= 0:
                # Load all frames - create a set of all indices
                indices_to_load = set(range(total_frames))
            else:
                # Load a specific number of evenly spaced frames
                num_to_pick = min(total_frames, num_evenly_spaced_frames)
                # Use np.linspace to get indices, ensuring the first and last are included.
                indices = np.linspace(0, total_frames - 1, num=num_to_pick, dtype=int)
                indices_to_load = set(indices)

            frames = []
            frame_count = 0
            ret = True
            
            # Read sequentially through the video ONCE
            while ret and len(frames) < len(indices_to_load):
                ret, frame = cap.read()
                if ret:
                    # If the current frame number is one we want, process and store it
                    if frame_count in indices_to_load:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_np = np.float32(frame_rgb) / 255.0
                        frame_tensor = torch.from_numpy(frame_np)
                        frames.append(frame_tensor)
                    frame_count += 1
            
            # --- END OF OPTIMIZATION ---

            if not frames:
                raise ValueError(f"Could not read any frames from the video: {video_path}")

            batch_tensor = torch.stack(frames)
            # Add a frame_count output for convenience
            return (batch_tensor, filename_without_ext, len(frames))

        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
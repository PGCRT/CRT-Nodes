import os
import cv2
import torch
import numpy as np
import re

def natural_sort_key(filename):
    # Split filename into parts, converting numeric parts to integers for natural sorting
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

class CRTLoadLastVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "X://insert/path/here"}),
                "sort_by": (["alphabetical", "date"], {"default": "date"}),
                "invert_order": ("BOOLEAN", {"default": False}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
            },
        }

    CATEGORY = "CRT/Load"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_last_video"

    def load_last_video(self, folder_path, sort_by, invert_order, frame_load_cap, skip_first_frames, select_every_nth):
        # Validate folder path
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid or non-existent folder path: {folder_path}")

        # Supported video extensions
        video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
        
        # Get all video files
        video_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(tuple(video_extensions))
        ]

        if not video_files:
            raise ValueError(f"No video files found in {folder_path}")

        # Sort files
        if sort_by == "date":
            video_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(folder_path, x)),
                reverse=invert_order
            )
        else:  # alphabetical with natural sorting
            video_files.sort(key=natural_sort_key, reverse=invert_order)

        # Select the last video
        video_path = os.path.join(folder_path, video_files[0])

        # Load video with OpenCV
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        total_frame_count = 0
        frames_added = 0
        total_frames_evaluated = -1

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            total_frame_count += 1
            if total_frame_count <= skip_first_frames:
                continue
            total_frames_evaluated += 1

            if total_frames_evaluated % select_every_nth != 0:
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to float32 and normalize to [0, 1]
            frame = np.array(frame, dtype=np.float32) / 255.0
            frames.append(frame)

            frames_added += 1
            if frame_load_cap > 0 and frames_added >= frame_load_cap:
                break

        video_cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")

        # Convert frames to tensor
        images = torch.from_numpy(np.stack(frames, axis=0))
        return (images,)

    @classmethod
    def IS_CHANGED(cls, folder_path, sort_by, invert_order, **kwargs):
        if not folder_path or not os.path.isdir(folder_path):
            return 0
        video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
        video_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(tuple(video_extensions))
        ]
        if not video_files:
            return 0
        if sort_by == "date":
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        else:
            video_files.sort(key=natural_sort_key)
        if invert_order:
            video_files.reverse()
        return os.path.getmtime(os.path.join(folder_path, video_files[0]))

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, **kwargs):
        if not folder_path or not os.path.isdir(folder_path):
            return f"Invalid folder path: {folder_path}"
        return True

NODE_CLASS_MAPPINGS = {
    "CRTLoadLastVideo": CRTLoadLastVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTLoadLastVideo": "Load Last Video (CRT)"
}
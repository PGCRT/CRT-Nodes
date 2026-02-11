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

    def _create_dummy_image(self):
        """Creates and returns a dummy image tensor that won't crash downstream nodes."""
        height, width = 512, 512
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black image
        text = "Dummy"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (255, 255, 255)  # White text
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(dummy_frame, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Convert to RGB, float32, and normalize to [0, 1]
        dummy_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
        dummy_frame = np.array(dummy_frame, dtype=np.float32) / 255.0
        images = torch.from_numpy(np.expand_dims(dummy_frame, axis=0))
        return (images,)

    def load_last_video(self, folder_path, sort_by, invert_order, frame_load_cap, skip_first_frames, select_every_nth):
        # Fallback if folder path is invalid or does not exist
        if not folder_path or not os.path.isdir(folder_path):
            print(f"⚠️ Invalid or non-existent folder path: {folder_path}. Returning dummy image.")
            return self._create_dummy_image()

        # Supported video extensions
        video_extensions = ['webm', 'mp4', 'mkv', 'gi', 'mov']

        video_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(video_extensions))
        ]

        # Fallback if no video files are found in the folder
        if not video_files:
            print(f"⚠️ No video files found in {folder_path}. Returning dummy image.")
            return self._create_dummy_image()

        if sort_by == "date":
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=not invert_order)
        else:
            video_files.sort(key=natural_sort_key, reverse=invert_order)

        video_path = os.path.join(folder_path, video_files[0])

        frames = []
        video_cap = None
        try:
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            total_frame_count = 0
            frames_added = 0

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                total_frame_count += 1
                if total_frame_count <= skip_first_frames:
                    continue

                if (total_frame_count - skip_first_frames - 1) % select_every_nth == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.array(frame, dtype=np.float32) / 255.0
                    frames.append(frame)

                    frames_added += 1
                    if frame_load_cap > 0 and frames_added >= frame_load_cap:
                        break
        except Exception as e:
            print(f"ERROR: Could not process video {video_path}. Reason: {e}. Returning dummy image.")
            return self._create_dummy_image()
        finally:
            if video_cap is not None and video_cap.isOpened():
                video_cap.release()

        if not frames:
            print(f"⚠️ No frames loaded from {video_path}, possibly due to settings. Returning dummy image.")
            return self._create_dummy_image()

        images = torch.from_numpy(np.stack(frames, axis=0))
        return (images,)

    @classmethod
    def IS_CHANGED(cls, folder_path, **kwargs):
        if not folder_path or not os.path.isdir(folder_path):
            return float("NaN")

        video_extensions = ['webm', 'mp4', 'mkv', 'gi', 'mov']
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(tuple(video_extensions))]

        if not files:
            return float("NaN")

        latest_time = max(os.path.getmtime(os.path.join(folder_path, f)) for f in files)
        return latest_time

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {"CRTLoadLastVideo": CRTLoadLastVideo}

NODE_DISPLAY_NAME_MAPPINGS = {"CRTLoadLastVideo": "Load Last Video (CRT)"}

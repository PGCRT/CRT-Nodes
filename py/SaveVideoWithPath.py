import os
import torch
import cv2
import numpy as np
import folder_paths
import tempfile
import subprocess

class SaveVideoWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "image": ("IMAGE", ),
                "folder_path": ("STRING", {"default": output_dir}),
                "subfolder_name": ("STRING", {"default": "videos"}),
                "filename": ("STRING", {"default": "output"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "frames_limit": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "activate": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "CRT/Save"

    def save_video(self, image, folder_path, subfolder_name, filename, fps, frames_limit, activate):
        # Check the activation trigger first
        if not activate:
            print("💡 SaveVideoWithPath is deactivated. Skipping video save.")
            return ()
            
        if image is None:
            print("❌ ERROR: No input image provided to SaveVideoWithPath.")
            return ()

        try:
            # Clean folder and file names
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')
            
            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, filename_clean + ".mp4")

            # Validate write permissions
            if not os.access(final_dir, os.W_OK):
                raise IOError(f"Error: No write permissions for directory {final_dir}")

            frames = (image.cpu().numpy() * 255).astype(np.uint8)
            
            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)
            
            if frames_limit != -1 and len(frames) > frames_limit:
                frames = frames[:frames_limit]

            height, width = frames.shape[1:3]

            # Write frames to temporary PNG files and use FFmpeg directly
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_paths.append(frame_path)

                # Construct FFmpeg command for lossless H.264
                ffmpeg_cmd = [
                    "ffmpeg", "-y",  # Overwrite output
                    "-framerate", str(fps),
                    "-i", os.path.join(temp_dir, "frame_%06d.png"),
                    "-c:v", "libx264",
                    "-crf", "0",  # Lossless
                    "-preset", "ultrafast",
                    "-pix_fmt", "yuv444p",  # Lossless pixel format
                    final_filepath
                ]

                # Run FFmpeg
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            print(f"✅ Video saved successfully to: {final_filepath}")
            return ()

        except Exception as e:
            print(f"❌ ERROR in SaveVideoWithPath: {str(e)}")
            # Re-raising the exception is good practice for debugging in ComfyUI
            raise e
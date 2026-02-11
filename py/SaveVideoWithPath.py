import os
import torch
import cv2
import numpy as np
import folder_paths
import tempfile
import subprocess
import json


class SaveVideoWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "image": ("IMAGE",),
                "folder_path": ("STRING", {"default": output_dir}),
                "subfolder_name": ("STRING", {"default": "videos"}),
                "filename": ("STRING", {"default": "output"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "frames_limit": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "activate": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "CRT/Save"

    def save_video(
        self, image, folder_path, subfolder_name, filename, fps, frames_limit, activate, prompt=None, extra_pnginfo=None
    ):
        if not activate:
            print("💡 SaveVideoWithPath is deactivated. Skipping video save.")
            return ()

        if image is None:
            print("❌ ERROR: No input image provided to SaveVideoWithPath.")
            return ()

        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')

            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, filename_clean + ".mp4")

            if not os.access(final_dir, os.W_OK):
                raise IOError(f"Error: No write permissions for directory {final_dir}")

            frames = (image.cpu().numpy() * 255).astype(np.uint8)

            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)

            if frames_limit != -1 and len(frames) > frames_limit:
                frames = frames[:frames_limit]

            with tempfile.TemporaryDirectory() as temp_dir:
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%06d.png"),
                ]

                metadata_str = ""
                video_metadata = {}
                if prompt is not None:
                    video_metadata["prompt"] = prompt
                if extra_pnginfo is not None:
                    video_metadata.update(extra_pnginfo)

                if video_metadata:
                    metadata_str = json.dumps(video_metadata)
                    metadata_file = os.path.join(temp_dir, "metadata.txt")

                    metadata_str = metadata_str.replace("\\", "\\\\")
                    metadata_str = metadata_str.replace(";", "\\;")
                    metadata_str = metadata_str.replace("#", "\\#")
                    metadata_str = metadata_str.replace("=", "\\=")
                    metadata_str = metadata_str.replace("\n", "\\\n")

                    with open(metadata_file, "w", encoding="utf-8") as f:
                        f.write(";FFMETADATA1\n")
                        f.write(f"comment={metadata_str}")

                    ffmpeg_cmd.extend(["-i", metadata_file])

                output_options = [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "3",
                    "-preset",
                    "fast",
                ]

                if video_metadata:
                    output_options.extend(["-map", "0:v", "-map_metadata", "1"])

                output_options.append(final_filepath)
                ffmpeg_cmd.extend(output_options)

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            print(f"✅ Video saved successfully to: {final_filepath}")
            return ({"ui": {"text": ["Video saved (Lossless)."]}},)

        except Exception as e:
            print(f"❌ ERROR in SaveVideoWithPath: {str(e)}")
            raise e

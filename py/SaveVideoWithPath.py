import os
import torch
import cv2
import numpy as np
import folder_paths

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
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "CRT/Save"

    def save_video(self, image, folder_path, subfolder_name, filename, fps, frames_limit):
        if image is None:
            return ()

        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')
            
            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, filename_clean + ".mp4")

            frames = (image.cpu().numpy() * 255).astype(np.uint8)
            
            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)
            
            if frames_limit != -1 and len(frames) > frames_limit:
                frames = frames[:frames_limit]

            height, width = frames.shape[1:3]
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(final_filepath, fourcc, fps, (width, height))

            if not out.isOpened():
                raise IOError(f"Error: Could not open video file for writing at {final_filepath}")

            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
            
            print(f"✅ Video saved successfully to: {final_filepath}")
            return ()

        except Exception as e:
            print(f"❌ ERROR in SaveVideoWithPath: {str(e)}")
            raise e
# /custom_nodes/AudioPreviewer.py

import torch
import numpy as np
import folder_paths
import torchaudio
import random
import os

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

class AudioPreviewer:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "preview_on_finish": (["ON", "OFF"], {"default": "ON"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "CRT/Audio"

    def process(self, audio, preview_on_finish):
        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        waveform_to_save = audio["waveform"][0] 
        sample_rate = audio["sample_rate"]
        
        file = f"{filename}_{counter:05}_.wav"
        path = os.path.join(full_output_folder, file)
        
        torchaudio.save(path, waveform_to_save.cpu(), sample_rate)

        saved_audio_info = { "filename": file, "subfolder": subfolder, "type": self.type }
        
        audio_np = waveform_to_save.cpu().numpy()

        peak_dbfs = 20 * np.log10(np.max(np.abs(audio_np))) if np.max(np.abs(audio_np)) > 0 else -np.inf
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(audio_np**2))) if np.mean(audio_np**2) > 0 else -np.inf
        
        lufs = -np.inf
        if pyln:
            try:
                meter = pyln.Meter(sample_rate)
                lufs = meter.integrated_loudness(audio_np.T)
            except Exception as e:
                print(f"AudioPreviewer: Could not calculate LUFS. Error: {e}")

        metrics = {
            "peak": f"{peak_dbfs:.1f}",
            "rms": f"{rms_dbfs:.1f}",
            "lufs": f"{lufs:.1f}" if lufs > -np.inf else "N/A"
        }
        
        return {"ui": {"audio": [saved_audio_info], "metrics": [metrics], "autoplay": [preview_on_finish == "ON"]}}

NODE_CLASS_MAPPINGS = { "AudioPreviewer": AudioPreviewer }
NODE_DISPLAY_NAME_MAPPINGS = { "AudioPreviewer": "Preview Audio (CRT)" }
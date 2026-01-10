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
                "preview_on_finish": (["ON", "OFF"], {"default": "ON"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.01}),
                "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.01}),
                "loaded_file": ("STRING", {"default": "", "forceInput": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "CRT/Audio"

    def process(self, preview_on_finish, trim_start=0.0, trim_end=0.0, audio=None, loaded_file=""):
        # Determine source audio
        if audio is None:
            if loaded_file and loaded_file.strip():
                try:
                    file_path = folder_paths.get_annotated_filepath(loaded_file)
                    waveform, sample_rate = torchaudio.load(file_path)
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                except Exception as e:
                    raise ValueError(f"Failed to load audio file '{loaded_file}': {e}")
            else:
                raise ValueError("No audio input provided and no file loaded.")

        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        waveform_to_save = audio["waveform"][0] 
        sample_rate = audio["sample_rate"]
        
        # Apply trimming if specified
        original_duration = waveform_to_save.shape[1] / sample_rate
        
        # Calculate trim in samples
        start_sample = int(trim_start * sample_rate)
        end_sample = int((original_duration - trim_end) * sample_rate)
        
        # Ensure valid range
        start_sample = max(0, start_sample)
        end_sample = min(waveform_to_save.shape[1], end_sample)
        # Prevent extending beyond the actual waveform length
        if end_sample < start_sample:
             end_sample = start_sample
        
        # Trim the waveform
        trimmed_waveform = waveform_to_save[:, start_sample:end_sample]
        trimmed_duration = trimmed_waveform.shape[1] / sample_rate
        
        file = f"{filename}_{counter:05}_.wav"
        path = os.path.join(full_output_folder, file)
        
        torchaudio.save(path, trimmed_waveform.cpu(), sample_rate)

        saved_audio_info = { "filename": file, "subfolder": subfolder, "type": self.type }
        
        audio_np = trimmed_waveform.cpu().numpy()

        peak_dbfs = -np.inf
        rms_dbfs = -np.inf
        lufs = -np.inf

        # Only calculate metrics if audio is not empty
        if audio_np.size > 0:
            if np.max(np.abs(audio_np)) > 0:
                peak_dbfs = 20 * np.log10(np.max(np.abs(audio_np)))
            
            mean_sq = np.mean(audio_np**2)
            if mean_sq > 0:
                rms_dbfs = 20 * np.log10(np.sqrt(mean_sq))
            
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
        
        trim_info = {
            "original_duration": f"{original_duration:.2f}",
            "trimmed_duration": f"{trimmed_duration:.2f}",
            "trim_start": f"{trim_start:.2f}",
            "trim_end": f"{trim_end:.2f}"
        }
        
        # Output trimmed audio for chaining
        output_audio = {
            "waveform": trimmed_waveform.unsqueeze(0),
            "sample_rate": sample_rate
        }
        
        return {
            "ui": {
                "audio": [saved_audio_info], 
                "metrics": [metrics], 
                "autoplay": [preview_on_finish == "ON"],
                "trim_info": [trim_info]
            },
            "result": (output_audio,)
        }

NODE_CLASS_MAPPINGS = { "AudioPreviewer": AudioPreviewer }
NODE_DISPLAY_NAME_MAPPINGS = { "AudioPreviewer": "Preview Audio (CRT)" }
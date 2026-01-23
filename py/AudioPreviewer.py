import torch
import numpy as np
import folder_paths
import soundfile as sf          # ← this is already present in ComfyUI embedded Python
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
                "trim_end":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.01}),
                "loaded_file": ("STRING", {"default": "", "forceInput": False}),
                "volume":      ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "CRT/Audio"

    def process(self, preview_on_finish, trim_start=0.0, trim_end=0.0, audio=None, loaded_file="", volume=1.0):
        # Determine source audio
        if audio is None:
            if loaded_file and loaded_file.strip():
                try:
                    file_path = folder_paths.get_annotated_filepath(loaded_file)
                    # ──────────────── CHANGE: use soundfile instead of torchaudio ────────────────
                    data, sample_rate = sf.read(file_path, dtype='float32')
                    # Convert to torch tensor in correct shape (channels, samples)
                    waveform = torch.from_numpy(data.T) if data.ndim == 2 else torch.from_numpy(data).unsqueeze(0)
                    audio = {"waveform": waveform, "sample_rate": sample_rate}
                except Exception as e:
                    raise ValueError(f"Failed to load audio file '{loaded_file}': {e}")
            else:
                raise ValueError("No audio input provided and no file loaded.")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Apply volume
        if abs(volume - 1.0) > 1e-6:
            waveform = waveform * volume

        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        
        waveform_to_save = waveform if waveform.dim() == 2 else waveform.squeeze(0)
        
        # Apply trimming if specified
        original_duration = waveform_to_save.shape[1] / sample_rate
        
        start_sample = int(trim_start * sample_rate)
        end_sample = int((original_duration - trim_end) * sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(waveform_to_save.shape[1], end_sample)
        if end_sample < start_sample:
            end_sample = start_sample
        
        trimmed_waveform = waveform_to_save[:, start_sample:end_sample]
        trimmed_duration = trimmed_waveform.shape[1] / sample_rate
        
        file = f"{filename}_{counter:05}_.wav"
        path = os.path.join(full_output_folder, file)
        
        # Save with soundfile (native format)
        sf.write(path, trimmed_waveform.T.cpu().numpy(), sample_rate)

        saved_audio_info = { "filename": file, "subfolder": subfolder, "type": self.type }
        
        audio_np = trimmed_waveform.cpu().numpy()

        peak_dbfs = -np.inf
        rms_dbfs = -np.inf
        lufs = -np.inf

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
        
        output_audio = {
            "waveform": trimmed_waveform.unsqueeze(0) if trimmed_waveform.dim() == 2 else trimmed_waveform,
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
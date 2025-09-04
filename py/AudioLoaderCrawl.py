import os
from pathlib import Path
import torch
import torchaudio

class AudioLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing audio files"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic file selection"}),
                "file_extension": (["wav", "mp3", "flac", "ogg"], {"default": "wav", "tooltip": "File extension to filter for"}),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "If true, include files in subfolders"}),
                "max_length_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "tooltip": "Maximum length of the audio in seconds (0 for no limit)"}),
                "start_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "tooltip": "Start loading the audio from this offset in seconds"}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 0.1, "tooltip": "Gain in decibels (dB)"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "file_name_no_ext", "file_path")
    FUNCTION = "load_audio"
    CATEGORY = "CRT/Load"

    def load_audio(self, folder_path, seed, file_extension, crawl_subfolders, max_length_seconds, start_offset_seconds, gain_db):
        # Create a single sample of silence for safe returns.
        # This prevents crashes in downstream nodes that cannot handle zero-length tensors.
        silent_waveform = torch.zeros(1, 1, 1) # [batch, channels, samples]
        silent_audio = {"waveform": silent_waveform, "sample_rate": 44100} # Use a common default sample rate
        safe_return = (silent_audio, "", "")

        if not folder_path or not folder_path.strip():
            print("❌ Error: Folder path is empty.")
            return safe_return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"❌ Error: Folder '{folder}' not found or is not a directory.")
            return safe_return

        # Ensure file extension has a dot
        if not file_extension.startswith('.'):
            file_extension = f".{file_extension}"

        try:
            # --- Smart Caching Logic ---
            cache_key = f"{str(folder.resolve())}_{crawl_subfolders}_{file_extension}"
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"🔎 Folder changed or not cached. Scanning '{folder}' for '{file_extension}' files...")
                pattern = f'*{file_extension}'
                if crawl_subfolders:
                    files = sorted([f for f in folder.rglob(pattern) if f.is_file()])
                else:
                    files = sorted([f for f in folder.glob(pattern) if f.is_file()])
                
                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"✅ Cached {len(files)} files.")

            files = self.cache[cache_key]['files']
            # --- End Caching Logic ---

            if not files:
                print(f"❌ Warning: No files with extension '{file_extension}' found in '{folder}'.")
                return safe_return

            # --- Deterministic and Safe Selection ---
            num_files = len(files)
            selected_index = seed % num_files
            selected_file = files[selected_index]
            # --- End Selection ---

            print(f"✅ Seed {seed} → File {selected_index + 1}/{num_files}: '{selected_file.name}'")
            
            # --- Load and Process Audio ---
            waveform, sample_rate = torchaudio.load(str(selected_file))

            # Apply start offset
            if start_offset_seconds > 0:
                offset_samples = int(start_offset_seconds * sample_rate)
                if offset_samples < waveform.shape[1]:
                    waveform = waveform[:, offset_samples:]
                else:
                    print(f"⚠️ Warning: Start offset is beyond the audio duration. Returning silent audio.")
                    return safe_return

            # Apply max length
            if max_length_seconds > 0:
                max_samples = int(max_length_seconds * sample_rate)
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]

            # Apply gain
            if gain_db != 0.0:
                gain_multiplier = 10 ** (gain_db / 20.0)
                waveform = waveform * gain_multiplier
                waveform = torch.clamp(waveform, -1.0, 1.0)

            # --- Format for ComfyUI ---
            waveform = waveform.unsqueeze(0)
            
            audio_out = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            filename_no_ext = selected_file.stem
            file_path_str = str(selected_file.resolve())

            return (audio_out, filename_no_ext, file_path_str)

        except Exception as e:
            print(f"❌ An unexpected error occurred in AudioLoaderCrawl: {str(e)}")
            return safe_return

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AudioLoaderCrawl": AudioLoaderCrawl
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLoaderCrawl": "Audio Loader Crawl (CRT)"
}
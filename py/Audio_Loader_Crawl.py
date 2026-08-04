import os
from pathlib import Path

import torch


def _load_audio_file(path):
    """Decode audio -> (waveform [channels, samples] float32, sample_rate).

    Mirrors ComfyUI core's own audio loader (comfy_extras/nodes_audio.py):
    pure PyAV, no torchaudio — torchaudio 2.11+ hard-requires torchcodec,
    which is not installed in this environment.
    """
    import av

    with av.open(str(path)) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        waveform = torch.cat(frames, dim=1)

    # f32 PCM conversion (as in core's f32_pcm)
    if waveform.dtype == torch.int16:
        waveform = waveform.float() / (2 ** 15)
    elif waveform.dtype == torch.int32:
        waveform = waveform.float() / (2 ** 31)
    elif not waveform.dtype.is_floating_point:
        raise ValueError(f"Unsupported wav dtype: {waveform.dtype}")

    return waveform, int(sample_rate)


class AudioLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing audio files"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for deterministic file selection",
                    },
                ),
                "file_extension": (
                    ["wav", "mp3", "flac", "ogg"],
                    {"default": "wav", "tooltip": "File extension to filter for"},
                ),
                "crawl_subfolders": ("BOOLEAN", {"default": False, "tooltip": "If true, include files in subfolders"}),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "Output filename without extension"}),
                "max_length_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Maximum length of the audio in seconds (0 for no limit)",
                    },
                ),
                "start_offset_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Start loading the audio from this offset in seconds",
                    },
                ),
                "gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -120.0, "max": 120.0, "step": 0.1, "tooltip": "Gain in decibels (dB)"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "file_name", "file_path")
    FUNCTION = "load_audio"
    CATEGORY = "CRT/Load"

    def load_audio(
        self,
        folder_path,
        seed,
        file_extension,
        crawl_subfolders,
        remove_extension,
        max_length_seconds,
        start_offset_seconds,
        gain_db,
    ):
        # Failure returns None for audio: downstream reference nodes (e.g. MiniMax H3)
        # skip None audio cleanly. Do NOT substitute fake silence here — a tiny silent
        # tensor resamples to 0 samples at 32 kHz and crashes the audio VAE instead.
        safe_return = (None, "", "")

        if not folder_path or not folder_path.strip():
            print("[ERROR] Error: Folder path is empty.")
            return safe_return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[ERROR] Error: Folder '{folder}' not found or is not a directory.")
            return safe_return

        # Ensure file extension has a dot
        if not file_extension.startswith('.'):
            file_extension = f".{file_extension}"

        try:
            # --- Smart Caching Logic ---
            cache_key = f"{str(folder.resolve())}_{crawl_subfolders}_{file_extension}"
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"[INFO] Folder changed or not cached. Scanning '{folder}' for '{file_extension}' files...")
                pattern = f'*{file_extension}'
                if crawl_subfolders:
                    files = sorted([f for f in folder.rglob(pattern) if f.is_file()])
                else:
                    files = sorted([f for f in folder.glob(pattern) if f.is_file()])

                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"[OK] Cached {len(files)} files.")

            files = self.cache[cache_key]['files']
            # --- End Caching Logic ---

            if not files:
                print(f"[ERROR] Warning: No files with extension '{file_extension}' found in '{folder}'.")
                return safe_return

            # --- Deterministic and Safe Selection ---
            num_files = len(files)
            selected_index = seed % num_files
            selected_file = files[selected_index]
            # --- End Selection ---

            print(f"[OK] Seed {seed} -> File {selected_index + 1}/{num_files}: '{selected_file.name}'")

            # --- Load and Process Audio (pure PyAV, like core LoadAudio) ---
            waveform, sample_rate = _load_audio_file(selected_file)

            # Apply start offset
            if start_offset_seconds > 0:
                offset_samples = int(start_offset_seconds * sample_rate)
                if offset_samples < waveform.shape[1]:
                    waveform = waveform[:, offset_samples:]
                else:
                    print("[WARN] Warning: Start offset is beyond the audio duration. Returning no audio.")
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

            audio_out = {"waveform": waveform, "sample_rate": sample_rate}

            file_name = selected_file.stem if remove_extension else selected_file.name
            file_path_str = str(selected_file.parent.resolve())

            return (audio_out, file_name, file_path_str)

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred in AudioLoaderCrawl: {str(e)}")
            return safe_return

# Node mappings
NODE_CLASS_MAPPINGS = {"AudioLoaderCrawl": AudioLoaderCrawl}

NODE_DISPLAY_NAME_MAPPINGS = {"AudioLoaderCrawl": "Audio Loader Crawl (CRT)"}

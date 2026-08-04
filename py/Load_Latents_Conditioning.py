from pathlib import Path

import torch
from safetensors import safe_open

from ._latent_conditioning_codec import decode_latent_conditioning, file_has_latent_conditioning

# Extensions probed (in priority order) next to the selected .safetensors file.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma")


def _find_audio_for(safetensors_path: Path):
    """Return the audio file sharing the safetensors' stem, or None."""
    for ext in AUDIO_EXTENSIONS:
        candidate = safetensors_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _load_with_av(audio_path: Path):
    """PyAV fallback decoder -> (waveform [channels, samples] float32, sample_rate)."""
    import av

    container = av.open(str(audio_path))
    try:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=stream.rate)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(torch.from_numpy(resampled.to_ndarray()))
        for resampled in resampler.resample(None):  # flush
            chunks.append(torch.from_numpy(resampled.to_ndarray()))
    finally:
        container.close()
    if not chunks:
        raise ValueError("no audio frames decoded")
    return torch.cat(chunks, dim=-1), stream.rate


def _load_audio_file(audio_path: Path):
    """Load audio as a ComfyUI AUDIO dict: {'waveform': [batch, channels, samples], 'sample_rate': int}.

    Pure PyAV (mirrors core comfy_extras/nodes_audio.py) — torchaudio 2.11+
    hard-requires torchcodec, which is not installed in this environment.
    """
    waveform, sample_rate = _load_with_av(audio_path)
    waveform = waveform.to(torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


class LoadLatentsConditioning:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "STRING", "INT", "AUDIO")
    RETURN_NAMES = ("latent", "conditioning", "file_name", "total_files", "audio")
    FUNCTION = "load_latents_conditioning"
    CATEGORY = "CRT/Load"
    DESCRIPTION = (
        "Loads a latent + conditioning pair saved with 'Save Latents Conditioning (CRT)', selected by seed. "
        "If an audio file with the same name sits next to the .safetensors file, it is returned on the "
        "'audio' output (None when no matching audio exists)."
    )

    def load_latents_conditioning(self, folder_path, seed, crawl_subfolders):
        if not folder_path or not folder_path.strip():
            return (None, None, "Error: Folder path is empty", 0, None)

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[ERROR] Error: Folder '{folder}' not found.")
            return (None, None, "Error: Folder not found", 0, None)

        # --- Smart Caching Logic ---
        cache_key = str(folder.resolve()) + ("_sub" if crawl_subfolders else "")
        current_mtime = folder.stat().st_mtime

        # Check if cache is invalid (key doesn't exist or modification time has changed)
        if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
            print(f"[INFO] Folder changed or not cached. Scanning '{folder}'...")
            try:
                path_iterator = folder.rglob('*.safetensors') if crawl_subfolders else folder.glob('*.safetensors')
                files = sorted([p for p in path_iterator if p.is_file() and file_has_latent_conditioning(p)])

                # Update the cache with the new file list and the current modification time
                self.cache[cache_key] = {'files': files, 'mtime': current_mtime}
                print(f"[OK] Cached {len(files)} latent + conditioning files from '{folder}'")
            except Exception as e:
                print(f"[ERROR] Error accessing folder '{folder}': {str(e)}")
                # Clear bad cache entry if it exists
                if cache_key in self.cache:
                    del self.cache[cache_key]
                return (None, None, "Error accessing folder", 0, None)

        # Retrieve the list of files from the (now guaranteed to be up-to-date) cache
        files = self.cache[cache_key]['files']

        if not files:
            print(f"[ERROR] Warning: No latent + conditioning files found in '{folder}'.")
            return (None, None, "No files found", 0, None)

        num_files = len(files)
        selected_index = seed % num_files
        selected_file = files[selected_index]

        try:
            with safe_open(str(selected_file), framework="pt", device="cpu") as f:
                latent, conditioning = decode_latent_conditioning(f)
        # Self-healing: If a file is in the cache but was deleted just before loading, this will catch it.
        except FileNotFoundError:
            print(
                f"[ERROR] Error: File '{selected_file}' was in cache but not found on disk. Invalidating cache for next run."
            )
            # Forcing a rescan on the next execution by removing the invalid cache entry.
            if cache_key in self.cache:
                del self.cache[cache_key]
            return (None, None, "Error: Cached file not found", 0, None)
        except Exception as e:
            print(f"[ERROR] Error loading file '{selected_file}': {str(e)}")
            return (None, None, "Error loading file", 0, None)

        # --- Optional sidecar audio (same stem, next to the .safetensors) ---
        audio = None
        audio_path = _find_audio_for(selected_file)
        if audio_path is not None:
            try:
                audio = _load_audio_file(audio_path)
                if audio["waveform"].shape[-1] == 0:
                    print(f"[WARN] Audio file '{audio_path.name}' decoded to 0 samples. Treating as no audio.")
                    audio = None
                else:
                    print(f"[OK] Found audio: '{audio_path.name}'")
            except Exception as e:
                print(f"[ERROR] Error loading audio '{audio_path}': {str(e)}")

        print(f"[OK] Seed {seed} -> File {selected_index + 1}/{num_files}: '{selected_file.name}'")
        return (latent, conditioning, selected_file.stem, num_files, audio)


# Node mappings
NODE_CLASS_MAPPINGS = {"LoadLatentsConditioning": LoadLatentsConditioning}

NODE_DISPLAY_NAME_MAPPINGS = {"LoadLatentsConditioning": "Load Latents Conditioning (CRT)"}

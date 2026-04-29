import os

import folder_paths
from comfy.utils import load_torch_file

from .download import download_file
from .mel_band_roformer import MelBandRoformer


MELBAND_DEFAULT_MODEL = "MelBandRoformer_fp16.safetensors"
MELBAND_DOWNLOAD_URL = (
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/"
    "MelBandRoformer_fp16.safetensors"
)

_MODEL_CACHE = {}


def _melband_config():
    return {
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0,
        "ff_dropout": 0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "sample_rate": 44100,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
    }


def ensure_melband_model_file(model_name, auto_download):
    model_path = folder_paths.get_full_path("diffusion_models", model_name)
    if model_path and os.path.isfile(model_path):
        return model_path

    fallback = os.path.join(folder_paths.models_dir, "diffusion_models", model_name)
    if os.path.isfile(fallback):
        return fallback

    if not auto_download:
        raise FileNotFoundError(
            f"MelBand model file not found and auto-download disabled: {model_name}"
        )

    if model_name != MELBAND_DEFAULT_MODEL:
        raise FileNotFoundError(
            "Auto-download is only supported for the default MelBand model "
            f"({MELBAND_DEFAULT_MODEL}). Missing: {model_name}"
        )

    return download_file(MELBAND_DOWNLOAD_URL, fallback, model_name)


def load_melband_model(model_name, auto_download):
    model_path = ensure_melband_model_file(model_name, auto_download)
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    model = MelBandRoformer(**_melband_config()).eval()
    model.load_state_dict(load_torch_file(model_path), strict=True)
    _MODEL_CACHE[model_path] = model
    return model

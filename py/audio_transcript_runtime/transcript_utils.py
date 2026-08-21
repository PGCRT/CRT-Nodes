"""Shared helpers for transcript nodes (extracted from Audio_Transcript)."""

import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import whisper

from comfy import model_management as mm
from comfy.utils import ProgressBar

FIXED_AUDIO_TRANSCRIPT_SEED = 0

WHISPER_LANGS_BY_NAME = None


def _get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def _apply_fixed_seed(seed=FIXED_AUDIO_TRANSCRIPT_SEED):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def _whisper_model_options():
    return [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "large-v3-turbo",
        "turbo",
    ]


def _whisper_language_code(language):
    global WHISPER_LANGS_BY_NAME
    if language == "auto":
        return None
    if WHISPER_LANGS_BY_NAME is None:
        WHISPER_LANGS_BY_NAME = {
            v.lower(): k for k, v in whisper.tokenizer.LANGUAGES.items()
        }
    code = WHISPER_LANGS_BY_NAME.get(language.lower())
    if code is None:
        raise ValueError(
            f"Unknown whisper language '{language}'. Use 'auto' or a full language "
            f"name such as 'english' or 'french'."
        )
    return code


def run_melband_isolation(model, audio):
    _apply_fixed_seed()
    audio_input = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if not isinstance(audio_input, torch.Tensor):
        audio_input = torch.as_tensor(audio_input)
    if audio_input.ndim != 3:
        raise ValueError(
            "Audio waveform must be [B, C, T] for MelBand processing. "
            f"Received shape: {tuple(audio_input.shape)}"
        )

    _, audio_channels, audio_length = audio_input.shape
    target_sr = 44100

    if audio_channels == 1:
        audio_input = audio_input.repeat(1, 2, 1)
        audio_channels = 2

    if sample_rate != target_sr:
        audio_input = TAF.resample(
            audio_input,
            orig_freq=sample_rate,
            new_freq=target_sr,
        )
        audio_length = int(audio_input.shape[-1])

    original_audio = audio_input[0]
    audio_input = original_audio

    chunk = 352800
    overlap_div = 2
    step = chunk // overlap_div
    fade_size = chunk // 10
    border = chunk - step

    if audio_length > 2 * border and border > 0:
        audio_input = F.pad(audio_input, (border, border), mode="reflect")

    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    window = _get_windowing_array(chunk, fade_size, device)

    audio_input = audio_input.to(device)
    vocals = torch.zeros_like(audio_input, dtype=torch.float32).to(device)
    counter = torch.zeros_like(audio_input, dtype=torch.float32).to(device)

    total_length = int(audio_input.shape[1])
    num_chunks = (total_length + step - 1) // step
    pbar = ProgressBar(num_chunks)

    model.to(device)
    with torch.no_grad():
        for i in range(0, total_length, step):
            part = audio_input[:, i : i + chunk]
            length = int(part.shape[-1])
            if length < chunk:
                if length > chunk // 2 + 1:
                    part = F.pad(part, (0, chunk - length), mode="reflect")
                else:
                    part = F.pad(
                        part, (0, chunk - length, 0, 0), mode="constant", value=0
                    )

            pred = model(part.unsqueeze(0))[0]
            chunk_window = window.clone()
            if i == 0:
                chunk_window[:fade_size] = 1
            elif i + chunk >= total_length:
                chunk_window[-fade_size:] = 1

            vocals[..., i : i + length] += (
                pred[..., :length] * chunk_window[..., :length]
            )
            counter[..., i : i + length] += chunk_window[..., :length]
            pbar.update(1)

    model.to(offload_device)
    estimated_sources = vocals / counter.clamp_min(1e-6)

    if audio_length > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    vocals_out = {
        "waveform": estimated_sources.unsqueeze(0).cpu(),
        "sample_rate": target_sr,
    }
    instruments_out = {
        "waveform": (original_audio.to(device) - estimated_sources)
        .unsqueeze(0)
        .cpu(),
        "sample_rate": target_sr,
    }
    return vocals_out, instruments_out

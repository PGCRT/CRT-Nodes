import importlib.util
import logging
import os
import sys
import tempfile
import types
import urllib.request
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import whisper

import folder_paths

from comfy import model_management as mm
from comfy.utils import ProgressBar, load_torch_file


LOGGER = logging.getLogger(__name__)

MELBAND_DEFAULT_MODEL = "MelBandRoformer_fp16.safetensors"
MELBAND_DOWNLOAD_URL = (
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/"
    "MelBandRoformer_fp16.safetensors"
)

WHISPER_MODEL_SUBDIR = os.path.join("stt", "whisper")

WHISPER_MODEL_CACHE = {}
MELBAND_MODEL_CACHE = {}
MELBAND_CLASS_CACHE = {"class": None}
WHISPER_LANGS_BY_NAME = None


def _save_audio_as_wav(path, audio):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {tuple(waveform.shape)}")

    pcm = (
        waveform.detach()
        .to(torch.float32)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    pcm = np.ascontiguousarray(pcm.T)
    channels = int(pcm.shape[1]) if pcm.ndim == 2 else 1

    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def _custom_nodes_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_melband_class():
    cached = MELBAND_CLASS_CACHE.get("class", None)
    if cached is not None:
        return cached

    plugin_model_dir = os.path.join(
        _custom_nodes_dir(), "ComfyUI-MelBandRoFormer", "model"
    )
    mel_converter_path = os.path.join(plugin_model_dir, "mel_converter.py")
    mel_model_path = os.path.join(plugin_model_dir, "mel_band_roformer.py")

    if not os.path.isfile(mel_converter_path) or not os.path.isfile(mel_model_path):
        raise FileNotFoundError(
            "MelBandRoFormer runtime sources were not found. "
            "Expected: custom_nodes/ComfyUI-MelBandRoFormer/model/*.py"
        )

    pkg_name = "_crt_melband_runtime"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [plugin_model_dir]
        sys.modules[pkg_name] = pkg

    converter_mod_name = f"{pkg_name}.mel_converter"
    if converter_mod_name not in sys.modules:
        converter_spec = importlib.util.spec_from_file_location(
            converter_mod_name,
            mel_converter_path,
        )
        converter_mod = importlib.util.module_from_spec(converter_spec)
        sys.modules[converter_mod_name] = converter_mod
        converter_spec.loader.exec_module(converter_mod)

    model_mod_name = f"{pkg_name}.mel_band_roformer"
    model_spec = importlib.util.spec_from_file_location(
        model_mod_name,
        mel_model_path,
    )
    model_mod = importlib.util.module_from_spec(model_spec)
    sys.modules[model_mod_name] = model_mod
    model_spec.loader.exec_module(model_mod)

    mel_class = getattr(model_mod, "MelBandRoformer", None)
    if mel_class is None:
        raise RuntimeError("Failed to import MelBandRoformer class.")

    MELBAND_CLASS_CACHE["class"] = mel_class
    return mel_class


def _ensure_melband_model_file(model_name, auto_download):
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

    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    tmp_path = f"{fallback}.part"
    LOGGER.info("Downloading MelBand model: %s", MELBAND_DOWNLOAD_URL)
    print(f"[CRT Audio Transcript] Downloading {model_name}...")

    req = urllib.request.Request(
        MELBAND_DOWNLOAD_URL,
        headers={"User-Agent": "CRT-Nodes AudioTranscript/1.0"},
    )
    with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    os.replace(tmp_path, fallback)
    print(f"[CRT Audio Transcript] Model downloaded: {fallback}")
    return fallback


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


def _load_melband_model(model_name, auto_download):
    model_path = _ensure_melband_model_file(model_name, auto_download)
    cached = MELBAND_MODEL_CACHE.get(model_path, None)
    if cached is not None:
        return cached

    mel_class = _load_melband_class()
    model = mel_class(**_melband_config()).eval()
    model.load_state_dict(load_torch_file(model_path), strict=True)
    MELBAND_MODEL_CACHE[model_path] = model
    return model


def _whisper_language_options():
    return ["auto"] + [
        s.capitalize() for s in sorted(list(whisper.tokenizer.LANGUAGES.values()))
    ]


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


def _get_whisper_model(model_name):
    model = WHISPER_MODEL_CACHE.get(model_name, None)
    if model is not None:
        return model

    download_root = os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR)
    os.makedirs(download_root, exist_ok=True)

    offload_device = mm.unet_offload_device()
    model = whisper.load_model(
        model_name, download_root=download_root, device=offload_device
    )
    WHISPER_MODEL_CACHE[model_name] = model
    return model


class CRT_AudioTranscript:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "isolate_voice": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "whisper_model": (
                    _whisper_model_options(),
                    {"default": "large-v3-turbo"},
                ),
                "melband_model_name": (
                    "STRING",
                    {"default": MELBAND_DEFAULT_MODEL},
                ),
                "auto_download_melband": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "language": (
                    "STRING",
                    {"default": "auto"},
                ),
                "prompt": ("STRING", {"default": ""}),
                "prefix": (
                    "STRING",
                    {"default": 'The subject says: "'},
                ),
                "suffix": ("STRING", {"default": '"'}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "AUDIO", "AUDIO")
    RETURN_NAMES = ("source_audio", "text", "vocals", "instruments")
    FUNCTION = "run"
    CATEGORY = "CRT/Audio"

    def _run_melband(self, model, audio):
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

    def _run_whisper(self, audio, whisper_model_name, language, prompt):
        global WHISPER_LANGS_BY_NAME

        whisper_model = _get_whisper_model(whisper_model_name)
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        try:
            whisper_model.to(load_device)
        except Exception:
            pass

        transcribe_args = {"initial_prompt": prompt}
        if language != "auto":
            if WHISPER_LANGS_BY_NAME is None:
                WHISPER_LANGS_BY_NAME = {
                    v.lower(): k for k, v in whisper.tokenizer.LANGUAGES.items()
                }
            transcribe_args["language"] = WHISPER_LANGS_BY_NAME[language.lower()]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _save_audio_as_wav(tmp_path, audio)
            result = whisper_model.transcribe(
                tmp_path,
                word_timestamps=False,
                **transcribe_args,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        try:
            whisper_model.to(offload_device)
        except Exception:
            pass
        mm.soft_empty_cache()

        return str(result.get("text", "")).strip()

    def run(
        self,
        audio,
        isolate_voice,
        whisper_model="large-v3-turbo",
        melband_model_name=MELBAND_DEFAULT_MODEL,
        auto_download_melband=True,
        language="auto",
        prompt="",
        prefix='The subject says: "',
        suffix='"',
    ):
        offload_device = mm.unet_offload_device()

        # Only load MelBand model if isolate_voice is enabled
        if bool(isolate_voice):
            mel_model = _load_melband_model(
                melband_model_name,
                bool(auto_download_melband),
            )
            vocals, instruments = self._run_melband(mel_model, audio)
            transcript_audio = vocals
        else:
            # Skip MelBand processing - return empty audio placeholders
            transcript_audio = audio
            sample_rate = int(audio.get("sample_rate", 44100))
            empty_waveform = torch.zeros((1, 2, 1))  # Minimal empty audio
            vocals = {"waveform": empty_waveform, "sample_rate": sample_rate}
            instruments = {"waveform": empty_waveform, "sample_rate": sample_rate}

        try:
            text = self._run_whisper(
                transcript_audio,
                whisper_model,
                language,
                prompt,
            )
            text = f"{prefix}{text}{suffix}"
        finally:
            # Ensure Whisper model is offloaded after use
            whisper_m = WHISPER_MODEL_CACHE.get(whisper_model, None)
            if whisper_m is not None:
                try:
                    whisper_m.to(offload_device)
                except Exception:
                    pass
            # Ensure MelBand model is offloaded after use
            if bool(isolate_voice):
                model_path = _ensure_melband_model_file(melband_model_name, False)
                mel_m = MELBAND_MODEL_CACHE.get(model_path, None)
                if mel_m is not None:
                    try:
                        mel_m.to(offload_device)
                    except Exception:
                        pass
            mm.soft_empty_cache()

        return (audio, text, vocals, instruments)


NODE_CLASS_MAPPINGS = {
    "CRT_AudioTranscript": CRT_AudioTranscript,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_AudioTranscript": "Audio Transcript (CRT)",
}

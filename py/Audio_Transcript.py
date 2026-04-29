import logging
import os
import tempfile
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import whisper

import folder_paths

from comfy import model_management as mm
from comfy.utils import ProgressBar

from ._cache_fingerprint import stable_fingerprint
from .audio_transcript_runtime import OMNIVOICE_LANGUAGES
from .audio_transcript_runtime.melband_runtime import (
    MELBAND_DEFAULT_MODEL,
    load_melband_model,
)
from .audio_transcript_runtime.omnivoice_runtime import generate_voice_clone


LOGGER = logging.getLogger(__name__)
FIXED_AUDIO_TRANSCRIPT_SEED = 0

WHISPER_MODEL_SUBDIR = os.path.join("stt", "whisper")

WHISPER_MODEL_CACHE = {}
WHISPER_LANGS_BY_NAME = None

EMPTY_AUDIO_CACHE = {}


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


def _empty_audio(sample_rate):
    sample_rate = int(sample_rate or 44100)
    cached = EMPTY_AUDIO_CACHE.get(sample_rate)
    if cached is None:
        cached = {
            "waveform": torch.zeros((1, 2, 1), dtype=torch.float32),
            "sample_rate": sample_rate,
        }
        EMPTY_AUDIO_CACHE[sample_rate] = cached
    return {
        "waveform": cached["waveform"].clone(),
        "sample_rate": cached["sample_rate"],
    }


def _apply_fixed_seed(seed=FIXED_AUDIO_TRANSCRIPT_SEED):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

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
    TRANSLATED_PREFIX = "The subject says : "
    @classmethod
    def IS_CHANGED(
        cls,
        audio,
        isolate_voice,
        enable_translation=False,
        translation_language="French",
        enable_omnivoice=False,
        whisper_model="large-v3-turbo",
        melband_model_name=MELBAND_DEFAULT_MODEL,
        auto_download_melband=True,
        language="auto",
        prompt="",
        prefix='The subject says: "',
        suffix='"',
    ):
        return stable_fingerprint(
            audio,
            bool(isolate_voice),
            bool(enable_translation),
            translation_language,
            bool(enable_omnivoice),
            whisper_model,
            melband_model_name,
            bool(auto_download_melband),
            language,
            prompt,
            prefix,
            suffix,
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "isolate_voice": ("BOOLEAN", {"default": False}),
                "enable_translation": ("BOOLEAN", {"default": False}),
                "translation_language": (OMNIVOICE_LANGUAGES, {"default": "French"}),
                "enable_omnivoice": ("BOOLEAN", {"default": False}),
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

    RETURN_TYPES = ("CRT_AUDIO_TRANSCRIPT_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "run"
    CATEGORY = "CRT/Audio"

    def _translate_with_llama_cpp(self, text, target_language):
        if not str(text or "").strip():
            return ""
        from .audio_transcript_runtime.llama_runtime import translate_text

        return translate_text(text, target_language)

    def _generate_with_omnivoice(self, reference_audio, reference_text, text, language):
        return generate_voice_clone(reference_audio, reference_text, text, language)

    def _run_melband(self, model, audio):
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

    def _run_whisper(self, audio, whisper_model_name, language, prompt):
        global WHISPER_LANGS_BY_NAME

        _apply_fixed_seed()

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
        enable_translation=False,
        translation_language="French",
        enable_omnivoice=False,
        whisper_model="large-v3-turbo",
        melband_model_name=MELBAND_DEFAULT_MODEL,
        auto_download_melband=True,
        language="auto",
        prompt="",
        prefix='The subject says: "',
        suffix='"',
    ):
        _apply_fixed_seed()
        offload_device = mm.unet_offload_device()
        sample_rate = int(audio.get("sample_rate", 44100))
        translated_text = ""
        translated_for_voice = ""
        omnivoice_audio = _empty_audio(sample_rate)
        status_parts = []

        # Only load MelBand model if isolate_voice is enabled
        if bool(isolate_voice):
            mel_model = load_melband_model(
                melband_model_name,
                bool(auto_download_melband),
            )
            vocals, instruments = self._run_melband(mel_model, audio)
            transcript_audio = vocals
            status_parts.append("voice isolation on")
        else:
            transcript_audio = audio
            vocals = _empty_audio(sample_rate)
            instruments = _empty_audio(sample_rate)
            status_parts.append("voice isolation off")

        raw_text = ""

        try:
            raw_text = self._run_whisper(
                transcript_audio,
                whisper_model,
                language,
                prompt,
            )
            text = f"{prefix}{raw_text}{suffix}"
            status_parts.append("whisper transcript ready")

            if bool(enable_translation):
                translated_for_voice = self._translate_with_llama_cpp(
                    raw_text,
                    translation_language,
                )
                translated_text = f"{self.TRANSLATED_PREFIX}{translated_for_voice}".strip()
                status_parts.append(
                    f"translated to {translation_language.lower()}"
                )
            else:
                translated_text = text

            if bool(enable_omnivoice):
                target_text = translated_for_voice if bool(enable_translation) else raw_text
                target_language = translation_language if bool(enable_translation) else "auto"
                omnivoice_audio, omnivoice_status = self._generate_with_omnivoice(
                    transcript_audio,
                    raw_text,
                    target_text,
                    target_language,
                )
                status_parts.append(omnivoice_status)
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
                try:
                    mel_model.to(offload_device)
                except Exception:
                    pass
            mm.soft_empty_cache()

        if not bool(enable_translation):
            status_parts.append("translation off")
        if not bool(enable_omnivoice):
            status_parts.append("omnivoice off")

        status_text = " | ".join(status_parts)
        pipe = {
            "source_audio": audio,
            "text": text,
            "vocals": vocals,
            "instruments": instruments,
            "translated_text": translated_text,
            "omnivoice_audio": omnivoice_audio,
            "status": status_text,
        }
        return (pipe,)


class CRT_AudioTranscriptPipeOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CRT_AUDIO_TRANSCRIPT_PIPE",),
            }
        }

    RETURN_TYPES = (
        "AUDIO",
        "STRING",
        "AUDIO",
        "AUDIO",
        "STRING",
        "AUDIO",
        "STRING",
    )
    RETURN_NAMES = (
        "source_audio",
        "text",
        "vocals",
        "instruments",
        "translated_text",
        "omnivoice_audio",
        "status",
    )
    FUNCTION = "unpack"
    CATEGORY = "CRT/Audio"

    def unpack(self, pipe):
        if not isinstance(pipe, dict):
            raise ValueError("Invalid Audio Transcript pipe.")

        return (
            pipe.get("source_audio"),
            pipe.get("text", ""),
            pipe.get("vocals"),
            pipe.get("instruments"),
            pipe.get("translated_text", ""),
            pipe.get("omnivoice_audio"),
            pipe.get("status", ""),
        )


NODE_CLASS_MAPPINGS = {
    "CRT_AudioTranscript": CRT_AudioTranscript,
    "CRT_AudioTranscriptPipeOut": CRT_AudioTranscriptPipeOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_AudioTranscript": "Audio Transcript (CRT)",
    "CRT_AudioTranscriptPipeOut": "Audio Transcript Pipe Out (CRT)",
}

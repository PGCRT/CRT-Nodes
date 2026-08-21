import logging

import torch

from comfy import model_management as mm

from ._cache_fingerprint import stable_fingerprint
from .audio_transcript_runtime.transcript_utils import (
    _apply_fixed_seed,
    _whisper_language_code,
    _whisper_model_options,
    run_melband_isolation,
)
from .audio_transcript_runtime.faster_whisper_runtime import (
    get_whisper_model as _fw_get_whisper_model,
    transcribe_audio as _fw_transcribe_audio,
    clear_cached_model as _fw_clear_cached_model,
)
from .audio_transcript_runtime.melband_runtime import (
    MELBAND_DEFAULT_MODEL,
    load_melband_model,
)


LOGGER = logging.getLogger(__name__)


class CRT_AudioTranscriptBatch:
    @classmethod
    def IS_CHANGED(
        cls,
        audio,
        isolate_voice,
        whisper_model="large-v3-turbo",
        compute_type="float16",
        vad_filter=True,
        transcribe_batch_size=8,
        beam_size=5,
        language="auto",
        keep_model_loaded=True,
    ):
        return stable_fingerprint(
            audio,
            bool(isolate_voice),
            whisper_model,
            compute_type,
            bool(vad_filter),
            int(transcribe_batch_size),
            int(beam_size),
            language,
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "isolate_voice": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run MelBandRoFormer voice isolation on each file before transcription.",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep the Whisper model loaded in VRAM after batch transcription. Disable to free VRAM immediately.",
                    },
                ),
            },
            "optional": {
                "whisper_model": (
                    _whisper_model_options(),
                    {"default": "large-v3-turbo"},
                ),
                "compute_type": (
                    ["float16", "int8_float16", "int8", "float32"],
                    {"default": "float16"},
                ),
                "vad_filter": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Cuts silences before inference for faster long-audio processing. Uses Silero VAD to skip silent regions.",
                    },
                ),
                "transcribe_batch_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of 30s chunks decoded in parallel per file. 1 disables batched decoding.",
                    },
                ),
                "beam_size": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Beam search width. 1 = greedy decoding (faster, slightly less accurate).",
                    },
                ),
                "language": (
                    "STRING",
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("strings_batch", "display", "status")
    OUTPUT_TOOLTIPS = (
        "One transcript per batch item as a list — pairs item by item with the batch loader's file_names in SaveTextWithPath.",
        "Same transcripts stitched into one display string, separated by a blank line per item.",
        "Summary status for the batch run.",
    )
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "run"
    CATEGORY = "CRT/Audio"
    DESCRIPTION = (
        "Transcribes a batched AUDIO input with faster-whisper (CTranslate2). "
        "Outputs one transcript per batch item as a list that pairs item by "
        "item with the batch loader's file_names in SaveTextWithPath."
    )

    def run(
        self,
        audio,
        isolate_voice,
        whisper_model="large-v3-turbo",
        compute_type="float16",
        vad_filter=True,
        transcribe_batch_size=8,
        beam_size=5,
        language="auto",
        keep_model_loaded=True,
    ):
        if audio is None or not isinstance(audio, dict) or audio.get("waveform") is None:
            raise ValueError("CRT_AudioTranscriptBatch received no audio.")

        _apply_fixed_seed()

        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim != 3:
            raise ValueError(
                "Batched audio waveform must be [B, C, T]. "
                f"Received shape: {tuple(waveform.shape)}"
            )

        sample_rate = int(audio.get("sample_rate", 44100))
        batch_count = int(waveform.shape[0])
        language_code = _whisper_language_code(language)

        fw_model = _fw_get_whisper_model(whisper_model, compute_type)

        mel_model = None
        if bool(isolate_voice):
            mel_model = load_melband_model(MELBAND_DEFAULT_MODEL, True)

        texts = []
        failures = 0
        try:
            for index in range(batch_count):
                item_audio = {
                    "waveform": waveform[index : index + 1],
                    "sample_rate": sample_rate,
                }
                try:
                    if mel_model is not None:
                        vocals, _ = run_melband_isolation(mel_model, item_audio)
                        item_audio = vocals

                    raw_text = _fw_transcribe_audio(
                        fw_model,
                        item_audio,
                        language=language_code,
                        prompt="",
                        vad_filter=vad_filter,
                        batch_size=transcribe_batch_size,
                        beam_size=beam_size,
                    )
                    texts.append(raw_text)
                    print(
                        "[CRT Audio Transcript Batch] "
                        f"[{index + 1}/{batch_count}] {len(raw_text)} chars"
                    )
                except Exception as exc:
                    failures += 1
                    texts.append("")
                    print(
                        "[CRT Audio Transcript Batch] "
                        f"ERROR on item {index + 1}/{batch_count}: {exc}"
                    )
        finally:
            if not bool(keep_model_loaded):
                _fw_clear_cached_model(whisper_model, compute_type)
            if mel_model is not None:
                try:
                    mel_model.to(mm.unet_offload_device())
                except Exception:
                    pass
            mm.soft_empty_cache()

        status_parts = [
            f"faster-whisper {whisper_model} ({compute_type})",
            f"{batch_count - failures}/{batch_count} files transcribed",
            f"voice isolation {'on' if bool(isolate_voice) else 'off'}",
        ]
        if failures:
            status_parts.append(f"{failures} failed")

        display = "\n\n".join(t if t.strip() else "(empty)" for t in texts)
        return (texts, display, " | ".join(status_parts))


NODE_CLASS_MAPPINGS = {
    "CRT_AudioTranscriptBatch": CRT_AudioTranscriptBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_AudioTranscriptBatch": "Audio Transcript Batch (CRT)",
}

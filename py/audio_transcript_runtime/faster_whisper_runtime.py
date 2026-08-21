import logging
import os

import torch
import torchaudio.functional as TAF

import folder_paths


LOGGER = logging.getLogger(__name__)

FW_MODEL_SUBDIR = os.path.join("stt", "faster-whisper")

_MODEL_CACHE = {}


def _import_faster_whisper():
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise ImportError(
            "The faster-whisper engine requires the 'faster-whisper' package. "
            "Install it into the ComfyUI python environment, e.g.: "
            "python_embeded\\python.exe -m pip install faster-whisper"
        ) from exc
    return WhisperModel, BatchedInferencePipeline


def get_whisper_model(model_name, compute_type="float16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type in ("float16", "int8_float16"):
        compute_type = "int8"

    key = (model_name, compute_type, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    WhisperModel, _ = _import_faster_whisper()

    download_root = os.path.join(folder_paths.models_dir, FW_MODEL_SUBDIR)
    os.makedirs(download_root, exist_ok=True)

    LOGGER.info(
        "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )
    _MODEL_CACHE[key] = model
    return model


def clear_cached_model(model_name, compute_type="float16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type in ("float16", "int8_float16"):
        compute_type = "int8"
    key = (model_name, compute_type, device)
    return _MODEL_CACHE.pop(key, None) is not None


def audio_to_numpy_16k(audio):
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

    waveform = waveform.to(torch.float32).mean(dim=0)

    if sample_rate != 16000:
        waveform = TAF.resample(
            waveform.unsqueeze(0),
            orig_freq=sample_rate,
            new_freq=16000,
        )[0]

    return waveform.cpu().numpy()


def transcribe_audio(
    model,
    audio,
    language=None,
    prompt="",
    vad_filter=True,
    batch_size=8,
    beam_size=5,
):
    np_audio = audio_to_numpy_16k(audio)

    transcribe_args = {
        "language": language or None,
        "beam_size": int(beam_size),
        "vad_filter": bool(vad_filter),
    }
    if prompt:
        transcribe_args["initial_prompt"] = prompt

    if int(batch_size) > 1:
        _, BatchedInferencePipeline = _import_faster_whisper()
        pipeline = BatchedInferencePipeline(model=model)
        segments, _ = pipeline.transcribe(
            np_audio,
            batch_size=int(batch_size),
            **transcribe_args,
        )
    else:
        segments, _ = model.transcribe(np_audio, **transcribe_args)

    return "".join(segment.text for segment in segments).strip()

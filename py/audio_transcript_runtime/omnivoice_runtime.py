import gc
import os
import threading
from dataclasses import dataclass

import folder_paths
import numpy as np
import torch


DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
FIXED_OMNIVOICE_SEED = 0


@dataclass(frozen=True)
class OmniVoiceModelKey:
    model_path: str
    device: str
    dtype: str


_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _log(level, message):
    print(f"[OmniVoice][{level}] {message}")


def _as_audio_dict(waveform, sample_rate):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 3:
        raise RuntimeError(f"Unexpected waveform shape from OmniVoice: {tuple(waveform.shape)}")
    return {
        "waveform": waveform.detach().cpu().float(),
        "sample_rate": int(sample_rate),
    }


def _audio_input_to_tuple(audio):
    if not audio:
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        return None
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return (waveform.detach().cpu().float(), sample_rate)


def resolve_device(device):
    d = (device or "auto").strip().lower()
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype, resolved_device):
    d = (dtype or "auto").strip().lower()
    if d == "float32":
        return torch.float32
    if d == "bfloat16":
        return torch.bfloat16
    if d == "float16":
        return torch.float16
    if resolved_device.startswith("cuda") or resolved_device == "mps":
        return torch.float16
    return torch.float32


def _dtype_name(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def _attention_candidates(resolved_device):
    if resolved_device.startswith("cuda"):
        return ["sdpa", "eager"]
    return ["eager"]


def get_model_cache_root(custom_root=""):
    if custom_root and custom_root.strip():
        return os.path.normpath(custom_root.strip())
    return os.path.join(folder_paths.models_dir, "omnivoice")


def ensure_model(model_id, model_cache_root):
    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    safe_name = model_id.replace("/", "--")
    local_dir = os.path.join(model_cache_root, safe_name)
    config_path = os.path.join(local_dir, "config.json")

    if os.path.isfile(config_path):
        return local_dir

    os.makedirs(model_cache_root, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        _log("INFO", f"Downloading model '{model_id}' to {local_dir}")
        snapshot_download(repo_id=model_id, local_dir=local_dir)
    except Exception as e:
        raise RuntimeError(
            "Failed to download OmniVoice model from Hugging Face. "
            f"repo={model_id} error={e}"
        ) from e

    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"Model download finished but config.json was not found at: {config_path}"
        )
    return local_dir


def get_or_load_model(model_id, model_cache_root, device="auto", dtype="auto"):
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    model_path = ensure_model(model_id=model_id, model_cache_root=model_cache_root)
    key = OmniVoiceModelKey(
        model_path=os.path.normpath(model_path),
        device=resolved_device,
        dtype=_dtype_name(resolved_dtype),
    )

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            _log("INFO", f"Loading OmniVoice model from {model_path} on {resolved_device} ({key.dtype})")
            try:
                from omnivoice import OmniVoice
            except Exception as e:
                raise RuntimeError(
                    "Python package 'omnivoice' is not installed in this ComfyUI environment. "
                    "Install it with: python -m pip install --no-deps omnivoice>=0.1.0"
                ) from e

            load_error = None
            model = None
            selected_attention = None
            for candidate in _attention_candidates(resolved_device):
                try:
                    model = OmniVoice.from_pretrained(
                        model_path,
                        device_map=resolved_device,
                        dtype=resolved_dtype,
                        load_asr=False,
                        attn_implementation=candidate,
                    )
                    selected_attention = candidate
                    break
                except Exception as e:
                    load_error = e
                    _log("WARN", f"Attention backend '{candidate}' failed, trying fallback")

            if model is None:
                raise RuntimeError(
                    f"Failed to load OmniVoice with available attention backends. Last error: {load_error}"
                )

            if selected_attention is not None:
                _log("INFO", f"Using attention backend: {selected_attention}")
            model._crt_attention_backend = selected_attention
            _MODEL_CACHE[key] = model

    if getattr(model, "_crt_attention_backend", None) is None:
        model._crt_attention_backend = "eager"

    return key, model


def unload_all_models():
    with _CACHE_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return count


def generate_voice_clone(reference_audio, reference_text, text, language, device="auto", dtype="auto"):
    clean_text = (text or "").strip()
    if not clean_text:
        raise RuntimeError("[OmniVoice][ERROR] Text input is empty")

    ref_audio_obj = _audio_input_to_tuple(reference_audio)
    if ref_audio_obj is None:
        raise RuntimeError("[OmniVoice][ERROR] Reference audio is required for voice clone")

    ref_text = (reference_text or "").strip()
    if not ref_text:
        raise RuntimeError("[OmniVoice][ERROR] Reference transcript is required for voice clone")

    _, model = get_or_load_model(
        model_id=DEFAULT_MODEL_ID,
        model_cache_root=get_model_cache_root(""),
        device=device,
        dtype=dtype,
    )
    sample_rate = int(getattr(model, "sampling_rate", 24000) or 24000)

    try:
        kwargs = {
            "text": clean_text,
            "num_step": 32,
            "guidance_scale": 2.0,
            "t_shift": 0.1,
            "layer_penalty_factor": 5.0,
            "position_temperature": 5.0,
            "class_temperature": 0.0,
            "speed": 1.0,
            "postprocess_output": True,
            "ref_audio": ref_audio_obj,
            "ref_text": ref_text,
        }
        if language != "auto":
            kwargs["language"] = str(language).strip()

        torch.manual_seed(FIXED_OMNIVOICE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FIXED_OMNIVOICE_SEED)

        audios = model.generate(**kwargs)
        if not audios:
            raise RuntimeError("[OmniVoice][ERROR] OmniVoice returned no audio")

        output = _as_audio_dict(audios[0], sample_rate)
        status = (
            f"Generated {output['waveform'].shape[-1]} samples at {sample_rate} Hz "
            f"using mode=voice_clone | seed={FIXED_OMNIVOICE_SEED} | model offloaded"
        )
        return output, status
    finally:
        unload_all_models()

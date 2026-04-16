import logging
import os
import urllib.request
import urllib.error
import threading
from typing import Dict, List, Tuple, Optional

import folder_paths

LOGGER = logging.getLogger(__name__)

# Model definitions with their URLs and target directories
MODEL_DEFINITIONS = {
    "diffusion_model": {
        "filename": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
        "subdir": "diffusion_models",
    },
    "ic_lora": {
        "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "subdir": "loras",
    },
    "ic_lora_outpaint": {
        "filename": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "url": "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "subdir": "loras",
    },
    "video_vae": {
        "filename": "LTX23_video_vae_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
        "subdir": "vae",
    },
    "audio_vae": {
        "filename": "LTX23_audio_vae_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
        "subdir": "vae",
    },
    "tae_vae": {
        "filename": "taeltx2_3.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
        "subdir": "vae",
    },
    "clip_gemma": {
        "filename": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
        "url": "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors",
        "subdir": "text_encoders",
    },
    "clip_projection": {
        "filename": "ltx-2.3_text_projection_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "subdir": "text_encoders",
    },
    "spatial_upscaler": {
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "subdir": "upscale_models",
    },
}

# Global download state
_download_progress: Dict[str, dict] = {}
_download_threads: Dict[str, threading.Thread] = {}


def _get_model_path(subdir: str, filename: str) -> str:
    """Get the full path for a model file."""
    base_dir = folder_paths.models_dir
    full_path = os.path.join(base_dir, subdir, filename)
    return full_path


def _check_model_exists(subdir: str, filename: str) -> bool:
    """Check if a model file exists in the expected location."""
    full_path = _get_model_path(subdir, filename)
    return os.path.isfile(full_path)


def check_all_models() -> Dict[str, bool]:
    """Check which models are present."""
    result = {}
    for key, info in MODEL_DEFINITIONS.items():
        result[key] = _check_model_exists(info["subdir"], info["filename"])
    return result


def download_model(model_type: str, progress_callback=None) -> Tuple[bool, str]:
    """Download a specific model."""
    if model_type not in MODEL_DEFINITIONS:
        return False, f"Unknown model type: {model_type}"

    info = MODEL_DEFINITIONS[model_type]
    target_path = _get_model_path(info["subdir"], info["filename"])

    # Check if already exists
    if os.path.isfile(target_path):
        return True, "Model already exists"

    # Create directory if needed
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    temp_path = f"{target_path}.part"

    try:
        LOGGER.info(f"Downloading {info['filename']} from {info['url']}")
        print(f"[LTX23 AutoDownload] Starting download: {info['filename']}")

        headers = {"User-Agent": "CRT-LTX23-AutoDownload/1.0"}
        req = urllib.request.Request(info["url"], headers=headers)

        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and progress_callback:
                        progress = (downloaded / total_size) * 100
                        progress_callback(model_type, progress, downloaded, total_size)

        # Move temp file to final location
        os.replace(temp_path, target_path)

        print(f"[LTX23 AutoDownload] Download complete: {info['filename']}")
        return True, "Download complete"

    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        error_msg = f"HTTP {e.code}: {e.reason}"
        LOGGER.error(f"Download failed for {info['filename']}: {error_msg}")
        return False, error_msg
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        error_msg = str(e)
        LOGGER.error(f"Download failed for {info['filename']}: {error_msg}")
        return False, error_msg


def start_download(model_type: str) -> bool:
    """Start downloading a model in a background thread."""
    if model_type in _download_threads and _download_threads[model_type].is_alive():
        return False  # Already downloading

    def progress_callback(model, progress, downloaded, total):
        _download_progress[model] = {
            "progress": progress,
            "downloaded": downloaded,
            "total": total,
            "complete": False,
            "error": None,
        }

    def download_task():
        _download_progress[model_type] = {
            "progress": 0,
            "downloaded": 0,
            "total": 0,
            "complete": False,
            "error": None,
        }

        success, message = download_model(model_type, progress_callback)

        _download_progress[model_type]["complete"] = success
        _download_progress[model_type]["error"] = None if success else message

        if success:
            _download_progress[model_type]["progress"] = 100

    thread = threading.Thread(target=download_task, daemon=True)
    thread.start()
    _download_threads[model_type] = thread
    return True


def get_download_status(model_type: str) -> Optional[dict]:
    """Get the download status for a model."""
    return _download_progress.get(model_type)


class CRT_LTX23AutoDownload:
    """LTX 2.3 AutoDownload - A utility node to check and download LTX 2.3 models.

    This is NOT a model loader - it only checks for missing models and downloads them.
    Use the standard ComfyUI loader nodes (DiffusionModelLoaderKJ, VAELoaderKJ, etc.)
    to actually load the models after downloading.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"
    CATEGORY = "CRT/LTX2.3"
    OUTPUT_NODE = True

    def run(self):
        # This node has no outputs - it's purely a utility for downloading models
        # The actual work is done via the web API and JavaScript UI
        return ()


# Web API endpoints for the frontend
class LTX23AutoDownloadAPI:
    """API endpoints for the LTX23 AutoDownload node."""

    @staticmethod
    def check_models():
        """API endpoint to check model status."""
        return check_all_models()

    @staticmethod
    def download_model_endpoint(model_type: str):
        """API endpoint to start a download."""
        success = start_download(model_type)
        return {"started": success, "model": model_type}

    @staticmethod
    def get_download_status_endpoint(model_type: str):
        """API endpoint to get download status."""
        status = get_download_status(model_type)
        if status is None:
            # Check if model exists now
            if model_type in MODEL_DEFINITIONS:
                exists = _check_model_exists(
                    MODEL_DEFINITIONS[model_type]["subdir"],
                    MODEL_DEFINITIONS[model_type]["filename"],
                )
                if exists:
                    return {"complete": True, "progress": 100, "error": None}
            return {"complete": False, "progress": 0, "error": None}
        return status


NODE_CLASS_MAPPINGS = {
    "CRT_LTX23AutoDownload": CRT_LTX23AutoDownload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_LTX23AutoDownload": "LTX 2.3 AutoDownload (CRT)",
}

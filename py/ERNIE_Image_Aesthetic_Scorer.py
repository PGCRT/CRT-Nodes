from __future__ import annotations

import gc
import importlib.util
import math
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .download_progress import snapshot_download_with_progress
from .ERNIE_Image_Aes_Compat import patch_checkpoint_compatibility


MODEL_ID = "baidu/ERNIE-Image-Aes"
MODEL_FOLDER_NAME = "ERNIE-Image-Aes"
IMAGE_SIZE = 448
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

SCORE_PROMPT = """
Rate the aesthetics score of the image in 0-100.
In the output format, numbers are replaced by 2 corresponding letters,
and the mapping relationship is:
score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

The answer only outputs 2 corresponding letters.
""".strip()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_LOADED_BACKEND: str | None = None
_MODEL_LOCK = threading.RLock()


def _score_to_token(score: int) -> str:
    if 0 <= score <= 25:
        first, offset = "a", score
    elif 26 <= score <= 50:
        first, offset = "c", score - 26
    elif 51 <= score <= 75:
        first, offset = "d", score - 51
    elif 76 <= score <= 100:
        first, offset = "e", score - 76
    else:
        raise ValueError("Aesthetic score token must be in the range 0..100.")
    return first + chr(ord("a") + offset)


SCORE_TOKENS = tuple(_score_to_token(score) for score in range(101))


def _model_directory() -> Path:
    import folder_paths

    return Path(folder_paths.models_dir) / "aesthetic" / MODEL_FOLDER_NAME


def _checkpoint_is_complete(model_dir: Path) -> bool:
    required = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer_config.json",
    )
    return all((model_dir / name).is_file() for name in required)


def _download_model() -> Path:
    model_dir = _model_directory()
    if _checkpoint_is_complete(model_dir):
        patch_checkpoint_compatibility(model_dir)
        return model_dir

    try:
        from huggingface_hub import constants as hf_constants
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download ERNIE-Image-Aes. "
            "Install this node's requirements and restart ComfyUI."
        ) from exc

    # Multi-gigabyte shards regularly exceed Hugging Face Hub's 10-second
    # default read timeout. Download one shard at a time and allow slow links
    # to keep making progress; completed files and HTTP ranges are reused.
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = max(
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT, 900
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download_with_progress(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        label="ERNIE Image Aesthetic Scorer",
        console_prefix="ERNIE Image Aes",
        etag_timeout=60,
        max_workers=1,
    )

    if not _checkpoint_is_complete(model_dir):
        raise RuntimeError(
            f"The ERNIE-Image-Aes download at {model_dir} is incomplete. "
            "Run the node again to resume it."
        )
    patch_checkpoint_compatibility(model_dir)
    return model_dir


def _flash_attention_available() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("flash_attn") is not None
    )


def _resolve_backend(requested: str) -> tuple[str, bool]:
    if requested == "auto":
        if _flash_attention_available():
            return "flash_attention_2", True
        return "eager", False
    if requested == "flash_attention_2":
        if not _flash_attention_available():
            raise RuntimeError(
                "FlashAttention 2 was selected but flash_attn is not importable. "
                "Choose auto/eager or install a compatible flash-attn build."
            )
        return requested, True
    return "eager", False


def _comfy_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_comfy_models() -> None:
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception as exc:
        print(f"[ERNIE Image Aes] ComfyUI memory cleanup warning: {exc}")


def _release_model() -> None:
    global _MODEL, _TOKENIZER, _LOADED_BACKEND

    _MODEL = None
    _TOKENIZER = None
    _LOADED_BACKEND = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _offload_model_to_cpu() -> None:
    global _MODEL

    if _MODEL is None:
        return

    try:
        _MODEL.to(device=torch.device("cpu"))
    except Exception as exc:
        print(
            "[ERNIE Image Aes] CPU offload failed; fully releasing the model "
            f"instead: {exc}"
        )
        _release_model()
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[ERNIE Image Aes] Model offloaded to CPU RAM")


def _load_model(attention_backend: str) -> tuple[Any, Any, torch.device, torch.dtype]:
    global _MODEL, _TOKENIZER, _LOADED_BACKEND

    resolved_backend, use_flash_attention = _resolve_backend(attention_backend)
    device = _comfy_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if _MODEL is not None and _LOADED_BACKEND == resolved_backend:
        first_parameter = next(_MODEL.parameters(), None)
        if (
            first_parameter is not None
            and first_parameter.device.type == "cpu"
            and device.type != "cpu"
        ):
            _free_comfy_models()
            print("[ERNIE Image Aes] Restoring model from CPU RAM")
        _MODEL.eval().to(device=device, dtype=dtype)
        return _MODEL, _TOKENIZER, device, dtype

    if _MODEL is not None:
        _release_model()

    _free_comfy_models()
    model_dir = _download_model()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load ERNIE-Image-Aes. "
            "Install this node's requirements and restart ComfyUI."
        ) from exc

    print(
        f"[ERNIE Image Aes] Loading on {device} with "
        f"{resolved_backend} attention ({dtype})"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attention,
        local_files_only=True,
    )
    model.eval().to(device)

    _MODEL = model
    _TOKENIZER = tokenizer
    _LOADED_BACKEND = resolved_backend
    return model, tokenizer, device, dtype


def _prepare_image(image: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image tensor, received shape {tuple(image.shape)}.")

    channels = image.shape[-1]
    if channels == 1:
        image = image.repeat(1, 1, 3)
    elif channels >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 channels; received {channels}.")

    image = image.detach().to(device="cpu", dtype=torch.float32)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = F.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp_(0.0, 1.0)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    image = (image - mean) / std
    return image.to(device=device, dtype=dtype)


def _save_above_threshold_images(
    image: torch.Tensor,
    continuous_scores: list[float],
    indices: list[int],
    raw_save_path: str,
    extension: str,
) -> None:
    if not raw_save_path.strip() or not indices:
        return

    save_directory = Path(raw_save_path.strip()).expanduser()
    if not save_directory.is_absolute():
        import folder_paths

        save_directory = Path(folder_paths.get_output_directory()) / save_directory
    save_directory.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for batch_index in indices:
        tensor = image[batch_index].detach().to(device="cpu", dtype=torch.float32)
        tensor = tensor.clamp(0.0, 1.0)
        array = tensor.mul(255).round().to(torch.uint8).numpy()

        channels = array.shape[-1]
        if channels == 1:
            array = array[..., 0]
            mode = "L"
        elif channels >= 4 and extension == "png":
            array = array[..., :4]
            mode = "RGBA"
        elif channels >= 3:
            array = array[..., :3]
            mode = "RGB"
        else:
            raise ValueError(
                f"Cannot save an image with {channels} channels."
            )

        score_name = f"{continuous_scores[batch_index]:.4f}"
        output_path = save_directory / f"{score_name}.{extension}"
        duplicate_index = 1
        while output_path.exists():
            output_path = save_directory / (
                f"{score_name}_{duplicate_index}.{extension}"
            )
            duplicate_index += 1

        pil_image = Image.fromarray(array, mode=mode)
        if extension == "jpg":
            pil_image.save(output_path, format="JPEG", quality=95, subsampling=0)
        else:
            pil_image.save(output_path, format="PNG", compress_level=4)

    print(
        f"[ERNIE Image Aes] Saved {len(indices)} image(s) above threshold "
        f"to {save_directory}"
    )


@torch.inference_mode()
def _continuous_aesthetic_score(
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    device: torch.device,
) -> float:
    question = f"<image>\n{SCORE_PROMPT}"

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token
        + IMG_END_TOKEN
    )
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    visual_embeddings = model.extract_feature(pixel_values)
    input_embeddings = model.language_model.get_input_embeddings()(input_ids)
    batch, sequence, hidden = input_embeddings.shape
    flat_embeddings = input_embeddings.reshape(batch * sequence, hidden)
    flat_ids = input_ids.reshape(batch * sequence)
    selected = flat_ids == model.img_context_token_id

    expected_visual_tokens = visual_embeddings.numel() // hidden
    if int(selected.sum().item()) != expected_visual_tokens:
        raise RuntimeError(
            "The image-token count does not match the visual embedding count: "
            f"{int(selected.sum().item())} prompt tokens versus "
            f"{expected_visual_tokens} visual tokens."
        )

    flat_embeddings[selected] = visual_embeddings.reshape(-1, hidden).to(
        device=flat_embeddings.device,
        dtype=flat_embeddings.dtype,
    )
    input_embeddings = flat_embeddings.reshape(batch, sequence, hidden)

    outputs = model.language_model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )

    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in SCORE_TOKENS]
    if any(token_id is None or token_id < 0 for token_id in token_ids):
        raise RuntimeError("The tokenizer is missing one or more aesthetic score tokens.")

    score_logits = outputs.logits[:, -1, token_ids].float()
    probabilities = torch.softmax(score_logits, dim=-1)
    values = torch.arange(101, device=device, dtype=torch.float32)
    score = float((probabilities @ values).item())

    if not math.isfinite(score):
        raise RuntimeError(f"The model returned a non-finite aesthetic score: {score}")
    return score


class ErnieImageAestheticScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image(s)": ("IMAGE",),
                "Threshold": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 99,
                        "step": 1,
                        "tooltip": "Images with a rounded score strictly greater than this value are returned and optionally saved.",
                    },
                ),
                "Path Save If above threshold": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional output folder for images above Threshold. Relative paths are resolved below ComfyUI/output; leave empty to disable saving.",
                    },
                ),
                "Extension": (
                    ["jpg", "png"],
                    {"default": "jpg"},
                ),
                "attention_backend": (
                    ["auto", "flash_attention_2", "eager"],
                    {
                        "default": "auto",
                        "tooltip": "Attention implementation used when loading ERNIE. Auto prefers FlashAttention 2 when installed and compatible, otherwise eager attention.",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep the model resident on the active device after scoring for faster subsequent runs at the cost of VRAM.",
                    },
                ),
                "offload_to_cpu": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When keep_model_loaded is off, retain the cached model on CPU instead of releasing it completely. This reduces reload time but uses system RAM.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "aesthetic_score",
        "aesthetic_score_text",
        "Best score",
        "Above Threshold",
    )
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "score"
    CATEGORY = "CRT/Image Scorer"
    DESCRIPTION = (
        "Scores each input image from 0 to 100 with baidu/ERNIE-Image-Aes. "
        "Returns parallel INT and STRING lists in batch order, plus the "
        "highest-scoring image and the batch scoring above Threshold. "
        "The model downloads on first use."
    )

    def score(
        self,
        attention_backend: str = "auto",
        keep_model_loaded: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ):
        # The public socket name contains punctuation, so receive it through
        # kwargs. The legacy key keeps already-saved workflows executable.
        image = kwargs.get("Image(s)", kwargs.get("image"))
        threshold = int(kwargs.get("Threshold", kwargs.get("threshold", 50)))
        save_path = str(
            kwargs.get("Path Save If above threshold", kwargs.get("save_path", ""))
            or ""
        )
        extension = str(kwargs.get("Extension", kwargs.get("extension", "jpg"))).lower()
        if image is None:
            raise ValueError("No image batch was provided to the Image(s) input.")
        if not 1 <= threshold <= 99:
            raise ValueError(f"Threshold must be between 1 and 99; received {threshold}.")
        if extension not in {"jpg", "png"}:
            raise ValueError(f"Extension must be jpg or png; received {extension}.")
        if image.ndim != 4:
            raise ValueError(
                f"ComfyUI IMAGE input must have shape [B,H,W,C], received {tuple(image.shape)}."
            )
        if image.shape[0] < 1:
            raise ValueError("The IMAGE batch is empty.")

        with _MODEL_LOCK:
            model = tokenizer = None
            scores: list[int] = []
            continuous_scores: list[float] = []
            score_texts: list[str] = []
            best_index = 0
            best_continuous_score = -math.inf
            try:
                model, tokenizer, device, dtype = _load_model(attention_backend)

                try:
                    from comfy.utils import ProgressBar

                    progress = ProgressBar(image.shape[0])
                except Exception:
                    progress = None

                for batch_index in range(image.shape[0]):
                    pixel_values = _prepare_image(image[batch_index], device, dtype)
                    score = _continuous_aesthetic_score(
                        model,
                        tokenizer,
                        pixel_values,
                        device,
                    )
                    rounded_score = max(0, min(100, int(round(score))))
                    scores.append(rounded_score)
                    continuous_scores.append(score)
                    score_texts.append(str(rounded_score))
                    if score > best_continuous_score:
                        best_continuous_score = score
                        best_index = batch_index
                    print(
                        f"[ERNIE Image Aes] Image {batch_index + 1}/"
                        f"{image.shape[0]}: {score:.4f} -> {rounded_score}"
                    )
                    if progress is not None:
                        progress.update(1)
                    del pixel_values
            finally:
                if not keep_model_loaded:
                    if offload_to_cpu:
                        _offload_model_to_cpu()
                    else:
                        _release_model()

        best_image = image[best_index : best_index + 1]
        above_indices = [
            index for index, rounded_score in enumerate(scores)
            if rounded_score > threshold
        ]
        above_threshold = image[above_indices]
        print(
            f"[ERNIE Image Aes] Best image: {best_index + 1}/{image.shape[0]} "
            f"({best_continuous_score:.4f} -> {scores[best_index]})"
        )
        print(
            f"[ERNIE Image Aes] Above threshold > {threshold}: "
            f"{len(above_indices)}/{image.shape[0]} image(s)"
        )
        _save_above_threshold_images(
            image=image,
            continuous_scores=continuous_scores,
            indices=above_indices,
            raw_save_path=save_path,
            extension=extension,
        )
        return (scores, score_texts, best_image, above_threshold)


NODE_CLASS_MAPPINGS = {
    "ErnieImageAestheticScore": ErnieImageAestheticScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErnieImageAestheticScore": "ERNIE Image Aesthetic Score (CRT)",
}

"""
Standalone CLIPSeg isolate input node.
Produces the same pipe structure as CRT_IsolateInput so CRT_IsolateOutput can reuse it unchanged.
"""

import gc
import os
import threading
import time
import io
from contextlib import nullcontext
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import folder_paths

from ._cache_fingerprint import stable_fingerprint
from .Isolate import (
    ISOLATE_PERFORMANCE_PRESETS,
    _effective_chunk_size,
    _batch_crop,
    _fill_mask_holes,
    _get_sam31_cache_entry,
    _print_timing,
    _preferred_compute_device,
    _resolve_isolate_preset,
    _resize_batch_masks_to_shape,
    _scale_to_mp,
    _scale_for_detection,
    _store_sam31_cache_entry,
    _timing_ui,
    _chunk_size_for_batch,
    _pad,
)


CLIPSEG_MODEL_ID = "Kijai/clipseg-rd64-refined-fp16"

_CLIPSEG_CACHE = {}
_CLIPSEG_CACHE_LOCK = threading.Lock()


def _clipseg_cache_root():
    return os.path.join(folder_paths.models_dir, "clip_seg")


def _ensure_clipseg_model(model_id):
    model_id = (model_id or CLIPSEG_MODEL_ID).strip()
    safe_name = model_id.replace("/", "--")
    local_dir = os.path.join(_clipseg_cache_root(), safe_name)
    config_path = os.path.join(local_dir, "config.json")

    if os.path.isfile(config_path):
        return local_dir

    os.makedirs(_clipseg_cache_root(), exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        print(f"[CRT CLIPSeg] Downloading model '{model_id}' to {local_dir}")
        snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as e:
        raise RuntimeError(
            "[CRT CLIPSeg] Failed to download model from Hugging Face. "
            f"repo={model_id} error={e}"
        ) from e

    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"[CRT CLIPSeg] Model download finished but config.json was not found at: {config_path}"
        )
    return local_dir


def _get_clipseg_runtime(model_id, device, dtype):
    model_path = _ensure_clipseg_model(model_id)
    key = (os.path.normpath(model_path), str(device), str(dtype))

    with _CLIPSEG_CACHE_LOCK:
        runtime = _CLIPSEG_CACHE.get(key)
        if runtime is not None:
            return runtime

        try:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except Exception as e:
            raise RuntimeError(
                "[CRT CLIPSeg] Python package 'transformers' is required. "
                "Install it in this ComfyUI environment with: pip install transformers huggingface_hub"
            ) from e

        print(f"[CRT CLIPSeg] Loading model from {model_path} on {device} ({dtype})")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            processor = CLIPSegProcessor.from_pretrained(model_path)
            model = CLIPSegForImageSegmentation.from_pretrained(model_path)
        runtime = {"processor": processor, "model": model}
        _CLIPSEG_CACHE[key] = runtime
        return runtime


def _normalize_clipseg_logits(mask_tensor):
    mins = mask_tensor.amin(dim=(-2, -1), keepdim=True)
    maxs = mask_tensor.amax(dim=(-2, -1), keepdim=True)
    denom = (maxs - mins).clamp_min(1e-6)
    return (mask_tensor - mins) / denom


def _gaussian_blur_masks(mask_tensor, blur_sigma):
    if blur_sigma <= 0:
        return mask_tensor

    radius = max(1, int(blur_sigma * 3))
    kernel_size = radius * 2 + 1
    sigma = max(0.1, float(blur_sigma))
    coords = torch.arange(kernel_size, device=mask_tensor.device, dtype=mask_tensor.dtype) - radius
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)

    work = mask_tensor.unsqueeze(1)
    work = F.pad(work, (radius, radius, 0, 0), mode="reflect")
    work = F.conv2d(work, kernel_x)
    work = F.pad(work, (0, 0, radius, radius), mode="reflect")
    work = F.conv2d(work, kernel_y)
    return work.squeeze(1)


def _segment_batch_clipseg(images, text, threshold, blur_sigma, device, dtype, model_id, chunk_size_override=0):
    runtime = _get_clipseg_runtime(model_id, device, dtype)
    processor = runtime["processor"]
    model = runtime["model"]

    model.to(device=device, dtype=dtype)
    batch_size, height, width, _ = images.shape
    chunk_size = _effective_chunk_size(batch_size, height, width, device, chunk_size_override)
    prompt = str(text).strip() or "face"
    masks = []

    autocast_enabled = dtype != torch.float32 and not comfy.model_management.is_device_mps(device)
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        chunk = images[start:end]
        pil_images = [
            Image.fromarray((frame.detach().cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8))
            for frame in chunk
        ]
        inputs = processor(text=[prompt] * len(pil_images), images=pil_images, return_tensors="pt", padding=True)
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_enabled else nullcontext():
            outputs = model(**inputs)

        logits = outputs.logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        chunk_masks = _normalize_clipseg_logits(torch.sigmoid(logits))
        chunk_masks = torch.where(chunk_masks > float(threshold), chunk_masks, torch.zeros_like(chunk_masks))
        chunk_masks = F.interpolate(chunk_masks.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
        chunk_masks = _gaussian_blur_masks(chunk_masks, float(blur_sigma))
        chunk_masks = (chunk_masks > 0).to(dtype=torch.float32)
        masks.append(chunk_masks.cpu())

        del outputs, logits, chunk_masks, inputs

    model.to(comfy.model_management.unet_offload_device())
    comfy.model_management.soft_empty_cache()
    return torch.cat(masks, dim=0)


class CRT_IsolateInputCLIPSeg:
    """
    Pads the image batch, uses CLIPSeg to detect a subject,
    crops around it, and passes a reconstruction pipe to CRT_IsolateOutput.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "what_to_detect": ("STRING", {"default": "face", "multiline": False, "tooltip": "Text prompt used by CLIPSeg, for example 'face' or 'person'."}),
                "performance_preset": (ISOLATE_PERFORMANCE_PRESETS, {"default": "Balanced", "tooltip": "Fast = quickest detection, Balanced = default tradeoff, Quality = highest available quality in the current custom-node path and slower processing."}),
                "detect_chunk_size": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1, "tooltip": "Detection batch size. 0 = process the whole batch at once, 1 = per-image processing, 2+ = fixed chunk size."}),
                "padding": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 64, "tooltip": "Extra border added around the input batch before detection so subjects near the edges are not cropped too tightly."}),
                "threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Detection confidence threshold. Lower values find more mask areas, higher values are stricter."}),
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Optional blur applied to CLIPSeg masks before crop extraction."}),
                "bbox_expansion": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.05, "tooltip": "Expands the detected crop box around the subject. Higher values give a looser crop."}),
                "crop_smooth_alpha": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Temporal smoothing for crop position. Lower = steadier crop, higher = follows motion more closely."}),
                "crop_megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.25, "tooltip": "Target resolution for the cropped output images. 0 disables crop rescaling."}),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        images,
        what_to_detect,
        performance_preset,
        detect_chunk_size,
        padding,
        threshold,
        blur_sigma,
        bbox_expansion,
        crop_smooth_alpha,
        crop_megapixels,
    ):
        return stable_fingerprint(
            images,
            what_to_detect,
            str(performance_preset),
            CLIPSEG_MODEL_ID,
            int(detect_chunk_size),
            int(padding),
            float(threshold),
            float(blur_sigma),
            float(bbox_expansion),
            float(crop_smooth_alpha),
            float(crop_megapixels),
        )

    RETURN_TYPES = ("IMAGE", "CRT_ISOLATE_PIPE")
    RETURN_NAMES = ("cropped_images", "pipe")
    FUNCTION = "execute"
    CATEGORY = "CRT/Utils/Isolate"

    def execute(
        self,
        images,
        what_to_detect,
        performance_preset,
        detect_chunk_size,
        padding,
        threshold,
        blur_sigma,
        bbox_expansion,
        crop_smooth_alpha,
        crop_megapixels,
    ):
        start_time = time.perf_counter()
        fill_mask = True
        preset = _resolve_isolate_preset(
            performance_preset,
            0.5,
            False,
        )
        detect_megapixels = preset["detect_megapixels"]
        reuse_first_pass_masks = preset["reuse_first_pass_masks"]

        cache_key = stable_fingerprint(
            images,
            what_to_detect,
            str(performance_preset),
            CLIPSEG_MODEL_ID,
            int(detect_chunk_size),
            int(padding),
            float(threshold),
            float(blur_sigma),
            float(bbox_expansion),
            float(crop_smooth_alpha),
            float(crop_megapixels),
        )
        cached = _get_sam31_cache_entry(cache_key)
        if cached is not None:
            elapsed = time.perf_counter() - start_time
            _print_timing("Isolate Input CLIPSeg", elapsed, images.shape[0], cache_hit=True)
            return {
                **_timing_ui("Isolate Input CLIPSeg", elapsed, images.shape[0], cache_hit=True),
                "result": cached,
            }

        batch_size, orig_h, orig_w, _ = images.shape
        # Detect directly on original images — avoids allocating a giant padded
        # tensor that can exhaust RAM on large video batches.
        detect_input = _scale_for_detection(images, float(detect_megapixels))
        device = _preferred_compute_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        pad_masks = _segment_batch_clipseg(
            detect_input,
            text=what_to_detect,
            threshold=float(threshold),
            blur_sigma=float(blur_sigma),
            device=device,
            dtype=dtype,
            model_id=CLIPSEG_MODEL_ID,
            chunk_size_override=int(detect_chunk_size),
        )
        del detect_input
        pad_masks = _resize_batch_masks_to_shape(
            pad_masks, orig_h, orig_w,
        )
        if bool(fill_mask):
            pad_masks = _fill_mask_holes(pad_masks)

        _, cropped, crop_masks, _, bboxes = _batch_crop(
            images,
            pad_masks,
            crop_size_mult=float(bbox_expansion),
            smooth_alpha=float(crop_smooth_alpha),
        )
        del pad_masks
        gc.collect()
        if bool(reuse_first_pass_masks):
            face_masks = crop_masks.float()
        else:
            detect_cropped = _scale_for_detection(cropped, float(detect_megapixels))

            face_masks = _segment_batch_clipseg(
                detect_cropped,
                text=what_to_detect,
                threshold=float(threshold),
                blur_sigma=float(blur_sigma),
                device=device,
                dtype=dtype,
                model_id=CLIPSEG_MODEL_ID,
                chunk_size_override=int(detect_chunk_size),
            )
            del detect_cropped
            face_masks = _resize_batch_masks_to_shape(
                face_masks,
                int(cropped.shape[1]),
                int(cropped.shape[2]),
            )
        if bool(fill_mask):
            face_masks = _fill_mask_holes(face_masks)

        cropped_scaled = _scale_to_mp(cropped, float(crop_megapixels))
        source_indices = list(range(batch_size))
        pipe = {
            "original_images": images,
            "original_crops": cropped,
            "face_masks": face_masks.cpu(),
            "source_indices": source_indices,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "padding": 0,
            "bboxes": bboxes,
            "crop_masks": crop_masks,
            "what_to_detect": what_to_detect,
        }

        result = (cropped_scaled, pipe)
        _store_sam31_cache_entry(cache_key, result)
        elapsed = time.perf_counter() - start_time
        _print_timing("Isolate Input CLIPSeg", elapsed, images.shape[0])
        return {
            **_timing_ui("Isolate Input CLIPSeg", elapsed, images.shape[0]),
            "result": result,
        }


NODE_CLASS_MAPPINGS = {
    "CRT_IsolateInputCLIPSeg": CRT_IsolateInputCLIPSeg,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_IsolateInputCLIPSeg": "Isolate Input CLIPSeg (CRT)",
}

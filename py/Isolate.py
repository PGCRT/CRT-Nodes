"""
CRT Isolate Input / Isolate Output — SAM 3.1 edition
Standalone subject-isolation nodes. Zero dependency on any custom node.
Requires:  ComfyUI with SAM3 support (PR #13408 or later)
           sam3.1_multiplex_fp16.safetensors  in  models/checkpoints/
           (https://huggingface.co/Comfy-Org/sam3.1)
"""

import gc
import math
import os
import time
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms import Resize, CenterCrop, InterpolationMode

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
from ._cache_fingerprint import stable_fingerprint

# ─── Sanity check – warn early if SAM3 support is missing ─────────────────────
try:
    import comfy.ldm.sam3  # noqa: F401
    _SAM31_OK = True
except ImportError:
    _SAM31_OK = False
    print(
        "[CRT Isolate] WARNING: comfy.ldm.sam3 not found. "
        "Apply ComfyUI PR #13408 (SAM3.1 support) before using these nodes."
    )


SAM31_DEFAULT_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
SAM31_DEFAULT_DOWNLOAD_URL = (
    "https://huggingface.co/Comfy-Org/SAM3/resolve/main/checkpoints/"
    "sam3.1_multiplex_fp16.safetensors?download=true"
)
SAM31_DEFAULT_MAX_DETECTIONS = 32

_SAM31_INPUT_CACHE = {}
_SAM31_INPUT_CACHE_ORDER = []
_SAM31_INPUT_CACHE_LIMIT = 0


# ─── Helpers: tensor ↔ PIL ────────────────────────────────────────────────────

def _t2pil(t):
    """(B,H,W,C) float[0-1] → list of PIL RGB images."""
    return [
        Image.fromarray((f.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8))
        for f in t
    ]


def _pil2t(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    return torch.stack([
        torch.from_numpy(np.array(i.convert("RGB")).astype(np.float32) / 255.0)
        for i in imgs
    ])


def _trim_batch_items(name_to_value):
    lengths = {}
    for name, value in name_to_value.items():
        try:
            lengths[name] = int(len(value))
        except TypeError:
            continue

    if not lengths:
        return name_to_value, None

    target_len = min(lengths.values())
    if target_len <= 0:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise RuntimeError(f"[CRT Isolate] Cannot composite empty batch. Batch sizes: {details}")

    if len(set(lengths.values())) == 1:
        return name_to_value, target_len

    details = ", ".join(f"{name}={length}" for name, length in lengths.items())
    print(
        f"[CRT Isolate] Batch mismatch detected ({details}). "
        f"Auto-trimming longer batch inputs to {target_len}."
    )

    trimmed = {}
    for name, value in name_to_value.items():
        if name in lengths and lengths[name] != target_len:
            trimmed[name] = value[:target_len]
        else:
            trimmed[name] = value
    return trimmed, target_len


def _timing_ui(label, total_seconds, batch_size, cache_hit=False):
    batch = max(1, int(batch_size) if batch_size is not None else 1)
    per_image = float(total_seconds) / float(batch)
    suffix = " (cache hit)" if cache_hit else ""
    text = f"{label}: total {total_seconds:.2f}s | per image {per_image:.3f}s{suffix}"
    return {"ui": {"text": [text]}}


def _print_timing(label, total_seconds, batch_size, cache_hit=False):
    batch = max(1, int(batch_size) if batch_size is not None else 1)
    per_image = float(total_seconds) / float(batch)
    suffix = " (cache hit)" if cache_hit else ""
    print(f"[CRT Isolate] {label}: total {total_seconds:.2f}s | per image {per_image:.3f}s{suffix}")


ISOLATE_PERFORMANCE_PRESETS = ["Fast", "Balanced", "Quality"]


def _resolve_isolate_preset(preset_name, detect_megapixels=0.5, reuse_first_pass_masks=False, refine_iterations=None):
    preset = str(preset_name or "Balanced")
    resolved = {
        "preset": preset,
        "detect_megapixels": float(detect_megapixels),
        "reuse_first_pass_masks": bool(reuse_first_pass_masks),
    }
    if refine_iterations is not None:
        resolved["refine_iterations"] = int(refine_iterations)

    if preset == "Fast":
        resolved["detect_megapixels"] = 0.25
        resolved["reuse_first_pass_masks"] = True
        if refine_iterations is not None:
            resolved["refine_iterations"] = 0
    elif preset == "Balanced":
        resolved["detect_megapixels"] = 0.5
        resolved["reuse_first_pass_masks"] = False
        if refine_iterations is not None:
            resolved["refine_iterations"] = 1
    elif preset == "Quality":
        resolved["detect_megapixels"] = 0.0
        resolved["reuse_first_pass_masks"] = False
        if refine_iterations is not None:
            resolved["refine_iterations"] = 2

    return resolved


# ─── SAM3.1 – model loading / release ────────────────────────────────────────

def _list_sam3_checkpoints():
    all_ckpts = folder_paths.get_filename_list("checkpoints")
    sam3 = [f for f in all_ckpts if "sam3" in f.lower()]
    return sam3 if sam3 else [SAM31_DEFAULT_CHECKPOINT]


def _download_sam31_checkpoint(checkpoint_name):
    if checkpoint_name != SAM31_DEFAULT_CHECKPOINT:
        raise RuntimeError(
            f"[CRT Isolate] '{checkpoint_name}' not found in models/checkpoints/. "
            f"Auto-download is only supported for {SAM31_DEFAULT_CHECKPOINT}."
        )

    checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
    target_path = os.path.join(checkpoints_dir, checkpoint_name)
    if os.path.isfile(target_path):
        return target_path

    os.makedirs(checkpoints_dir, exist_ok=True)
    tmp_path = f"{target_path}.part"
    print(f"[CRT Isolate] Downloading SAM3.1 checkpoint: {checkpoint_name}")

    request = urllib.request.Request(
        SAM31_DEFAULT_DOWNLOAD_URL,
        headers={"User-Agent": "CRT-Nodes Isolate/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print(f"[CRT Isolate] SAM3.1 checkpoint downloaded: {target_path}")
    return target_path


def _load_sam31(checkpoint_name):
    """Load SAM3.1 via ComfyUI's standard checkpoint loader. Returns (model_patcher, clip)."""
    if not _SAM31_OK:
        raise RuntimeError(
            "[CRT Isolate] comfy.ldm.sam3 not found. "
            "Apply ComfyUI PR #13408 (SAM3.1 support) first."
        )
    ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
    if ckpt_path is None:
        ckpt_path = _download_sam31_checkpoint(checkpoint_name)
    print(f"[CRT Isolate] Loading SAM3.1: {checkpoint_name}")
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_model=True, output_clip=True,
        output_vae=False, output_clipvision=False,
    )
    return out[0], out[1]  # model_patcher, CLIP


def _fill_mask_holes(mask_batch):
    """Fill interior holes in a (B,H,W) binary mask batch."""
    try:
        import scipy.ndimage as nd
    except ImportError:
        print("[CRT Isolate] WARNING: scipy not installed - fill_mask skipped")
        return mask_batch

    filled = []
    for i in range(mask_batch.shape[0]):
        mask_np = (mask_batch[i].detach().cpu().numpy() > 0.5)
        filled_np = nd.binary_fill_holes(mask_np)
        filled.append(torch.from_numpy(filled_np.astype(np.float32)))
    return torch.stack(filled).to(mask_batch.dtype)


def _clone_cache_value(value):
    if isinstance(value, torch.Tensor):
        # Isolate cache entries can be very large video tensors. Reusing the tensor
        # reference avoids duplicating tens of GB on cache store/read.
        return value
    if isinstance(value, dict):
        return {k: _clone_cache_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cache_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(v) for v in value)
    return value


def _get_sam31_cache_entry(cache_key):
    if _SAM31_INPUT_CACHE_LIMIT <= 0:
        return None
    cached = _SAM31_INPUT_CACHE.get(cache_key)
    if cached is None:
        return None
    print("[CRT Isolate] Input cache hit")
    return _clone_cache_value(cached)


def _store_sam31_cache_entry(cache_key, value):
    if _SAM31_INPUT_CACHE_LIMIT <= 0:
        return
    if cache_key in _SAM31_INPUT_CACHE:
        try:
            _SAM31_INPUT_CACHE_ORDER.remove(cache_key)
        except ValueError:
            pass

    _SAM31_INPUT_CACHE[cache_key] = _clone_cache_value(value)
    _SAM31_INPUT_CACHE_ORDER.append(cache_key)

    while len(_SAM31_INPUT_CACHE_ORDER) > _SAM31_INPUT_CACHE_LIMIT:
        stale_key = _SAM31_INPUT_CACHE_ORDER.pop(0)
        _SAM31_INPUT_CACHE.pop(stale_key, None)


def _release_sam31(model, clip):
    """Move SAM3.1 and its CLIP encoder off the GPU."""
    try:
        model.model.diffusion_model.to("cpu")
    except Exception:
        pass
    try:
        clip.cond_stage_model.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()
    print("[CRT Isolate] SAM3.1 released from VRAM")


def _parse_max_detections(text, single_item=False):
    prompts = [part.strip() for part in str(text or "").split(",")]
    max_detections = []
    default_max = 1 if bool(single_item) else SAM31_DEFAULT_MAX_DETECTIONS
    for prompt in prompts:
        if not prompt:
            continue
        head, sep, tail = prompt.rpartition(":")
        if sep and tail.strip().isdigit():
            max_detections.append(max(1, int(tail.strip())))
        else:
            max_detections.append(default_max)
    if not max_detections:
        max_detections.append(default_max)
    return max_detections


# ─── SAM3.1 – text encoding ───────────────────────────────────────────────────

def _encode_text(clip, text, device, dtype, single_item=False):
    """
    Encode a plain-text prompt for SAM3 detection.
    Returns list of  (embeddings, attention_mask, max_detections).
    Supports the 'subject:N' syntax (e.g. 'face:3' → detect up to 3 faces).
    """
    clip.load_model()
    tokens = clip.tokenize(text)
    out = clip.encode_from_tokens(tokens, return_dict=True)
    parsed_max_detections = _parse_max_detections(text, single_item=single_item)

    multi = out.get("sam3_multi_cond")
    if multi:
        result = []
        for index, entry in enumerate(multi):
            emb = entry["cond"].to(device=device, dtype=dtype)
            mask = entry.get("attention_mask")
            default_max_det = parsed_max_detections[min(index, len(parsed_max_detections) - 1)]
            max_det = entry.get("max_detections", default_max_det)
            if mask is None:
                mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
            else:
                mask = mask.to(device)
            result.append((emb, mask, max_det))
        return result

    # Fallback for non-SAM3 CLIP (shouldn't happen with a proper SAM3.1 ckpt)
    cond = out.get("cond", out).to(device=device, dtype=dtype)
    mask = torch.ones(cond.shape[0], cond.shape[1], dtype=torch.int64, device=device)
    return [(cond, mask, parsed_max_detections[0])]


# ─── SAM3.1 – mask refinement (from PR nodes_sam3.py) ────────────────────────

def _refine_mask(sam3, orig_hwc, coarse_mask, box_xyxy, H, W, device, dtype, iterations=2):
    """Refine a coarse detector mask with the SAM decoder. Returns [1, H, W]."""
    def _fallback():
        return (F.interpolate(
            coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False,
        )[0] > 0).float()

    if iterations <= 0:
        return _fallback()

    pad = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad));  cy1 = max(0, int(y1 - bh * pad))
    cx2 = min(W, int(x2 + bw * pad));  cy2 = min(H, int(y2 + bh * pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return _fallback()

    crop_input = comfy.utils.common_upscale(
        orig_hwc[cy1:cy2, cx1:cx2].unsqueeze(0).movedim(-1, 1),
        1008,
        1008,
        "bilinear",
        crop="disabled",
    ).to(device=device, dtype=dtype)

    mh, mw = coarse_mask.shape[-2:]
    mx1 = max(0,      int(cx1 / W * mw));  my1 = max(0,      int(cy1 / H * mh))
    mx2 = max(mx1+1,  int(cx2 / W * mw));  my2 = max(my1+1,  int(cy2 / H * mh))

    logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        inp = F.interpolate(logit, size=(1008, 1008), mode="bilinear", align_corners=False)
        logit = sam3.forward_segment(crop_input, mask_inputs=inp)

    refined = F.interpolate(logit, size=(cy2-cy1, cx2-cx1), mode="bilinear", align_corners=False)
    full = torch.zeros(1, 1, H, W, device=device)
    full[..., cy1:cy2, cx1:cx2] = refined
    return (full[0] > 0).float()


# ─── SAM3.1 – batch segmentation ─────────────────────────────────────────────

def _segment_batch_instances(model, clip, images, text, threshold, refine_iters, single_item=False, chunk_size_override=0):
    """
    Detect subject in every frame using SAM3.1.
    images : (B, H, W, C) float[0-1]
    Returns: list of per-frame mask tensors, each shaped (N, H, W)
    """
    B, H, W, _ = images.shape

    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype  = model.model.get_dtype()
    sam3   = model.model.diffusion_model

    imgs_working = images.movedim(-1, 1)
    cond_list = _encode_text(clip, text, device, dtype, single_item=single_item)

    chunk_size = _effective_chunk_size(B, H, W, device, chunk_size_override)
    all_masks = [[] for _ in range(B)]

    def _get_batch_item(value, idx):
        if isinstance(value, (list, tuple)):
            return value[idx]
        return value[idx]

    for start in range(0, B, chunk_size):
        end = min(B, start + chunk_size)
        frame_chunk = imgs_working[start:end].to(device=device, dtype=dtype)

        for text_emb, text_mask, max_det in cond_list:
            results = sam3(
                frame_chunk,
                text_embeddings=text_emb,
                text_mask=text_mask,
                threshold=threshold,
                orig_size=(H, W),
            )

            for local_idx, b in enumerate(range(start, end)):
                pred_boxes = _get_batch_item(results["boxes"], local_idx)
                scores = _get_batch_item(results["scores"], local_idx)
                raw_masks = _get_batch_item(results["masks"], local_idx)

                probs = scores.sigmoid()
                keep = probs > threshold
                kept_boxes = pred_boxes[keep].cpu()
                kept_scores = probs[keep].cpu()
                kept_masks = raw_masks[keep]

                order = kept_scores.argsort(descending=True)[:max_det]
                kept_masks = kept_masks[order]
                kept_boxes = kept_boxes[order]

                for m, box in zip(kept_masks, kept_boxes):
                    all_masks[b].append(
                        _refine_mask(sam3, images[b], m, box, H, W, device, dtype, refine_iters)
                    )

    finalized_masks = []
    for frame_masks in all_masks:
        if frame_masks:
            finalized_masks.append(
                torch.cat(frame_masks, dim=0).to(device=comfy.model_management.intermediate_device())
            )
        else:
            finalized_masks.append(
                torch.zeros(0, H, W, device=comfy.model_management.intermediate_device())
            )

    return finalized_masks


def _segment_batch(model, clip, images, text, threshold, refine_iters, single_item=False, chunk_size_override=0):
    """
    Detect subject in every frame using SAM3.1.
    images : (B, H, W, C) float[0-1]
    Returns: (B, H, W) float[0-1]
    """
    B, H, W, _ = images.shape
    all_instances = _segment_batch_instances(
        model,
        clip,
        images,
        text,
        threshold,
        refine_iters,
        single_item=single_item,
        chunk_size_override=chunk_size_override,
    )

    combined_masks = []
    for masks in all_instances:
        if masks.shape[0] > 0:
            combined_masks.append((masks > 0).any(dim=0).float())
        else:
            combined_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))

    return torch.stack(combined_masks)


def _batch_crop_per_item(images, masks_by_frame, crop_size_mult=1.0, smooth_alpha=0.0):
    """Return one crop per detected item instead of combining detections per frame.

    When the crop bbox extends beyond the image boundary (subjects near edges),
    the extracted crop is zero-padded to the full max_bb × max_bb size. The pad
    amounts are returned so the Output node can strip them before uncropping,
    preserving correct spatial correspondence without pre-allocating a padded batch.
    """
    def _bbox(mn):
        nz = np.nonzero(mn)
        if len(nz[0]) == 0:
            return 0, 0, 0, 0, 0
        min_y, max_y = int(nz[0].min()), int(nz[0].max())
        min_x, max_x = int(nz[1].min()), int(nz[1].max())
        return min_x, max_x, min_y, max_y, max(max_x-min_x, max_y-min_y)

    B, H, W, C = images.shape
    all_sizes = []
    for frame_masks in masks_by_frame:
        if frame_masks.shape[0] == 0:
            continue
        for i in range(frame_masks.shape[0]):
            all_sizes.append(_bbox(frame_masks[i].cpu().numpy())[4])

    curr_max = max(all_sizes) if any(s > 0 for s in all_sizes) else 64
    max_bb = max(16, math.ceil(round(curr_max * crop_size_mult) / 16) * 16)
    if max_bb > H or max_bb > W:
        max_bb = math.floor(min(H, W) / 2) * 2

    source_indices = []
    bboxes = []
    crop_edge_pads = []  # (pad_left, pad_right, pad_top, pad_bottom) per crop
    crops = []
    crop_masks = []
    prev_centers = {}

    for frame_idx in range(B):
        frame_masks = masks_by_frame[frame_idx]
        if frame_masks.shape[0] == 0:
            frame_masks = torch.zeros(1, H, W, device=images.device, dtype=images.dtype)

        for item_idx, mask in enumerate(frame_masks):
            mn = mask.cpu().numpy()
            nz = np.nonzero(mn)

            if len(nz[0]) > 0:
                cx = float(np.mean(nz[1]))
                cy = float(np.mean(nz[0]))
                if smooth_alpha > 0.0:
                    prev_center = prev_centers.get(item_idx)
                    if prev_center is not None:
                        cx = smooth_alpha * cx + (1.0 - smooth_alpha) * prev_center[0]
                        cy = smooth_alpha * cy + (1.0 - smooth_alpha) * prev_center[1]
                    prev_centers[item_idx] = (cx, cy)
                cx = round(cx)
                cy = round(cy)
                half = max_bb // 2
                x0 = max(0, cx - half)
                x1 = min(W, cx + half)
                y0 = max(0, cy - half)
                y1 = min(H, cy + half)
                # Padding needed on each side when bbox exceeds image bounds
                pl = max(0, half - cx)
                pr = max(0, cx + half - W)
                pt = max(0, half - cy)
                pb = max(0, cy + half - H)
            else:
                x0, y0 = 0, 0
                x1, y1 = min(max_bb, W), min(max_bb, H)
                pl, pr = 0, max(0, max_bb - W)
                pt, pb = 0, max(0, max_bb - H)

            source_indices.append(frame_idx)
            bboxes.append((x0, y0, x1 - x0, y1 - y0))
            crop_edge_pads.append((pl, pr, pt, pb))

            cimg = images[frame_idx, y0:y1, x0:x1, :]
            cmsk = mask[y0:y1, x0:x1]
            # Zero-pad on the side(s) where the bbox exceeded the image boundary.
            # This is equivalent to the old full-batch _pad() but only allocates
            # one frame at a time, so memory is trivial.
            if pl or pr or pt or pb:
                cimg = F.pad(cimg.permute(2, 0, 1).unsqueeze(0), (pl, pr, pt, pb)).squeeze(0).permute(1, 2, 0)
                cmsk = F.pad(cmsk.unsqueeze(0).unsqueeze(0), (pl, pr, pt, pb)).squeeze(0).squeeze(0)

            new_sz = max(cimg.shape[0], cimg.shape[1])
            if new_sz > 0:
                rs = Resize(new_sz, interpolation=InterpolationMode.BILINEAR, antialias=True)
                rm = Resize(new_sz, interpolation=InterpolationMode.NEAREST)
                ri = rs(cimg.permute(2, 0, 1))
                rm_ = rm(cmsk.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                csz = min(max_bb, ri.shape[1], ri.shape[2])
                cc = CenterCrop(csz)
                crops.append(cc(ri).permute(1, 2, 0))
                crop_masks.append(cc(rm_.unsqueeze(0)).squeeze(0))
            else:
                crops.append(cimg)
                crop_masks.append(cmsk)

    return (
        torch.stack(crops),
        torch.stack(crop_masks),
        bboxes,
        source_indices,
        crop_edge_pads,
    )


def _batch_uncrop_grouped(original, cropped, bboxes, source_indices, border_blending=0.0, crop_edge_pads=None):
    if border_blending <= 0.0:
        result = original.clone()
        for i, ((bx, by, bw, bh), source_idx) in enumerate(zip(bboxes, source_indices)):
            ow = int(result.shape[2])
            oh = int(result.shape[1])
            x0 = max(0, int(bx))
            y0 = max(0, int(by))
            x1 = min(ow, int(bx + bw))
            y1 = min(oh, int(by + bh))
            pw = x1 - x0
            ph = y1 - y0
            if pw <= 0 or ph <= 0:
                continue

            crop_i = cropped[i : i + 1].permute(0, 3, 1, 2)
            if crop_edge_pads is not None:
                pl, pr, pt, pb = crop_edge_pads[i]
                ch_c, cw_c = crop_i.shape[2], crop_i.shape[3]
                # Strip zero-padding before resizing so the content maps correctly
                # back to the clamped bbox region in the original image.
                crop_i = crop_i[
                    :, :,
                    pt : (ch_c - pb) if pb > 0 else ch_c,
                    pl : (cw_c - pr) if pr > 0 else cw_c,
                ]

            crop_resized = F.interpolate(
                crop_i,
                size=(ph, pw),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)[0]
            result[source_idx, y0:y1, x0:x1, :] = crop_resized.to(result.device, dtype=result.dtype)
        return result

    orig_pil = _t2pil(original)
    crop_pil = _t2pil(cropped)
    result = [img.copy() for img in orig_pil]

    for crop, (bx, by, bw, bh), source_idx in zip(crop_pil, bboxes, source_indices):
        img = result[source_idx]
        r = (max(0, bx), max(0, by), min(img.size[0], bx+bw), min(img.size[1], by+bh))
        pw, ph = r[2] - r[0], r[3] - r[1]
        if pw <= 0 or ph <= 0:
            continue

        crop_r = crop.resize((pw, ph), Image.LANCZOS).convert("RGB")
        blend_ratio = (max(pw, ph) / 2) * float(border_blending)
        mask = Image.new("L", img.size, 0)
        blk = Image.new("L", (pw, ph), 255)
        if blend_ratio > 0:
            bw_px = round(blend_ratio / 2)
            ImageDraw.Draw(blk).rectangle((0, 0, pw - 1, ph - 1), outline=0, width=bw_px)
        mask.paste(blk, r[:2])
        if blend_ratio > 0:
            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend = img.convert("RGBA")
        blend.paste(crop_r, r[:2])
        blend.putalpha(mask)
        result[source_idx] = Image.alpha_composite(img.convert("RGBA"), blend).convert("RGB")

    return _pil2t(result)

    return torch.stack(all_masks)   # (B, H, W)


# ─── Image padding ────────────────────────────────────────────────────────────

def _pad(images, padding):
    if padding == 0:
        return images.clone()
    B, H, W, C = images.shape
    out = torch.zeros(B, H + 2*padding, W + 2*padding, C, dtype=images.dtype)
    out[:, padding:padding+H, padding:padding+W, :] = images
    return out


# ─── Batch crop (KJNodes BatchCropFromMaskAdvanced logic, inlined) ─────────────

def _batch_crop(images, masks, crop_size_mult=1.0, smooth_alpha=1.0):
    """
    Returns: (original_images, cropped, cropped_masks, combined_crop_masks, bboxes)
    """
    def _bbox(mn):
        nz = np.nonzero(mn)
        if len(nz[0]) == 0:
            return 0, 0, 0, 0, 0
        min_y, max_y = int(nz[0].min()), int(nz[0].max())
        min_x, max_x = int(nz[1].min()), int(nz[1].max())
        return min_x, max_x, min_y, max_y, max(max_x-min_x, max_y-min_y)

    B, H, W, C = images.shape

    sizes = [_bbox(masks[i].cpu().numpy())[4] for i in range(B)]
    curr_max = max(sizes) if any(s > 0 for s in sizes) else 64
    max_bb = max(16, math.ceil(round(curr_max * crop_size_mult) / 16) * 16)
    if max_bb > H or max_bb > W:
        max_bb = math.floor(min(H, W) / 2) * 2

    # Combined bbox for combined_crop_masks (used by uncrop)
    cmb_np = torch.max(masks, dim=0)[0].cpu().numpy()
    cx0, cx1, cy0, cy1, _ = _bbox(cmb_np)
    if cx0 == cx1 or cy0 == cy1:
        cx0, cy0 = 0, 0
        cx1, cy1 = min(max_bb, W), min(max_bb, H)
    else:
        ccx, ccy = (cx0+cx1)/2, (cy0+cy1)/2
        h = max_bb // 2
        cx0 = max(0, round(ccx-h));  cx1 = min(W, round(ccx+h))
        cy0 = max(0, round(ccy-h));  cy1 = min(H, round(ccy+h))

    bboxes, crops, crop_masks_list, cmb_masks_list = [], [], [], []
    prev_center = None

    for i in range(B):
        mn = masks[i].cpu().numpy()
        img, mask = images[i], masks[i]
        nz = np.nonzero(mn)

        if len(nz[0]) > 0:
            cx = float(np.mean(nz[1]));  cy = float(np.mean(nz[0]))
            curr = (round(cx), round(cy))
            if prev_center is None or i == 0:
                center = curr
            else:
                center = (
                    round(smooth_alpha * curr[0] + (1-smooth_alpha) * prev_center[0]),
                    round(smooth_alpha * curr[1] + (1-smooth_alpha) * prev_center[1]),
                )
            prev_center = center

            h = max_bb // 2
            x0 = max(0, center[0]-h);  x1 = min(W, center[0]+h)
            y0 = max(0, center[1]-h);  y1 = min(H, center[1]+h)
            bboxes.append((x0, y0, x1-x0, y1-y0))

            cimg = img[y0:y1, x0:x1, :]
            cmsk = mask[y0:y1, x0:x1]
            new_sz = max(cimg.shape[0], cimg.shape[1])
            if new_sz > 0:
                rs = Resize(new_sz, interpolation=InterpolationMode.BILINEAR, antialias=True)
                rm = Resize(new_sz, interpolation=InterpolationMode.NEAREST)
                ri = rs(cimg.permute(2, 0, 1))
                rm_ = rm(cmsk.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                csz = min(max_bb, ri.shape[1], ri.shape[2])
                cc  = CenterCrop(csz)
                crops.append(cc(ri).permute(1, 2, 0))
                crop_masks_list.append(cc(rm_.unsqueeze(0)).squeeze(0))
            else:
                crops.append(cimg)
                crop_masks_list.append(cmsk)
        else:
            bboxes.append((0, 0, W, H))
            crops.append(img)
            crop_masks_list.append(mask)

        cmb_masks_list.append(masks[i][cy0:cy1, cx0:cx1])

    # Ensure uniform spatial size for combined crop masks before stacking
    cmb_h, cmb_w = cy1-cy0, cx1-cx0
    safe_cmb = []
    for m in cmb_masks_list:
        if m.shape[0] == cmb_h and m.shape[1] == cmb_w:
            safe_cmb.append(m)
        else:
            p = torch.zeros(cmb_h, cmb_w, dtype=m.dtype)
            h_, w_ = min(m.shape[0], cmb_h), min(m.shape[1], cmb_w)
            p[:h_, :w_] = m[:h_, :w_]
            safe_cmb.append(p)

    return (
        images,
        torch.stack(crops),
        torch.stack(crop_masks_list),
        torch.stack(safe_cmb),
        bboxes,
    )


# ─── Grow + blur mask (KJNodes GrowMaskWithBlur logic, inlined) ───────────────

def _grow_blur_mask(mask, expand, blur_radius):
    """mask: (B,H,W) → (B,H,W)"""
    if expand == 0 and blur_radius == 0:
        return mask

    work = mask.to(dtype=torch.float32)

    if expand > 0 and torch.count_nonzero(work).item() > 0:
        kernel_size = int(expand) * 2 + 1
        work = F.max_pool2d(
            work.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=int(expand),
        ).squeeze(1)

    if blur_radius > 0:
        radius = max(1, int(math.ceil(float(blur_radius))))
        kernel_size = radius * 2 + 1
        sigma = max(0.1, float(blur_radius) / 3.0)
        coords = torch.arange(kernel_size, device=work.device, dtype=work.dtype) - radius
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)

        kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1)

        work_4d = work.unsqueeze(1)
        work_4d = F.pad(work_4d, (radius, radius, 0, 0), mode="reflect")
        work_4d = F.conv2d(work_4d, kernel_x)
        work_4d = F.pad(work_4d, (0, 0, radius, radius), mode="reflect")
        work_4d = F.conv2d(work_4d, kernel_y)
        work = work_4d.squeeze(1)

    return work.clamp_(0.0, 1.0).to(dtype=mask.dtype)


def _temporal_smooth_mask(mask, source_indices, radius):
    """Stabilize masks across the batch. Best results are with stable/single-item crops."""
    if radius <= 0 or mask.shape[0] < 2:
        return mask

    work = mask.to(dtype=torch.float32)
    out = work.clone()
    groups = {}
    per_frame_slots = {}

    # Keep multi-item crops from the same frame out of the same temporal group.
    for index, source_idx in enumerate(source_indices):
        frame_idx = int(source_idx)
        slot = per_frame_slots.get(frame_idx, 0)
        per_frame_slots[frame_idx] = slot + 1
        groups.setdefault(slot, []).append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue

        # If the requested range is larger than the available temporal span,
        # just use the whole local sequence instead of failing or oversmoothing oddly.
        effective_radius = min(int(radius), len(indices) - 1)
        if effective_radius <= 0:
            continue

        sequence = work[indices].unsqueeze(0).unsqueeze(0)
        sigma = max(0.5, float(effective_radius) / 2.0)
        coords = (
            torch.arange(
                effective_radius * 2 + 1,
                device=work.device,
                dtype=work.dtype,
            )
            - effective_radius
        )
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        kernel = kernel.view(1, 1, -1, 1, 1)

        padded = F.pad(
            sequence,
            (0, 0, 0, 0, effective_radius, effective_radius),
            mode="replicate",
        )
        smoothed = F.conv3d(padded, kernel).squeeze(0).squeeze(0)
        out[indices] = smoothed

    return out.clamp_(0.0, 1.0).to(dtype=mask.dtype)


# ─── Batch uncrop (KJNodes BatchUncropAdvanced logic, inlined) ────────────────

def _batch_uncrop(original, cropped, bboxes, border_blending=0.0):
    """Paste `cropped` back into `original` at each bbox, with optional edge blend."""
    if border_blending <= 0.0:
        result = original.clone()
        for i, (bx, by, bw, bh) in enumerate(bboxes):
            ow = int(result.shape[2])
            oh = int(result.shape[1])
            x0 = max(0, int(bx))
            y0 = max(0, int(by))
            x1 = min(ow, int(bx + bw))
            y1 = min(oh, int(by + bh))
            pw = x1 - x0
            ph = y1 - y0
            if pw <= 0 or ph <= 0:
                continue

            crop_resized = F.interpolate(
                cropped[i : i + 1].permute(0, 3, 1, 2),
                size=(ph, pw),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)[0]
            result[i, y0:y1, x0:x1, :] = crop_resized.to(result.device, dtype=result.dtype)
        return result

    orig_pil = _t2pil(original)
    crop_pil = _t2pil(cropped)
    result   = []

    for img, crop, (bx, by, bw, bh) in zip(orig_pil, crop_pil, bboxes):
        r = (max(0, bx), max(0, by), min(img.size[0], bx+bw), min(img.size[1], by+bh))
        pw, ph = r[2]-r[0], r[3]-r[1]
        if pw <= 0 or ph <= 0:
            result.append(img);  continue

        crop_r      = crop.resize((pw, ph), Image.LANCZOS).convert("RGB")
        blend_ratio = (max(pw, ph) / 2) * float(border_blending)

        mask = Image.new("L", img.size, 0)
        blk  = Image.new("L", (pw, ph), 255)
        if blend_ratio > 0:
            bw_px = round(blend_ratio / 2)
            ImageDraw.Draw(blk).rectangle((0, 0, pw-1, ph-1), outline=0, width=bw_px)
        mask.paste(blk, r[:2])
        if blend_ratio > 0:
            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio/4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio/4))

        blend = img.convert("RGBA")
        blend.paste(crop_r, r[:2])
        blend.putalpha(mask)
        result.append(Image.alpha_composite(img.convert("RGBA"), blend).convert("RGB"))

    return _pil2t(result)


# ─── Color matching ──────────────────────────────────────────────────────────

_CM_METHODS = ["none", "mkl", "hm", "reinhard", "mvgd", "hm-mkl-hm", "hm-mvgd-hm"]

def _color_match(image_ref, image_target, method, strength):
    """
    Match colors of image_target to image_ref using color-matcher.
    Both tensors: (B, H, W, C) float[0-1].
    Returns color-corrected image_target tensor (same shape).
    """
    if method == "none" or strength == 0.0:
        return image_target
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        print("[CRT Isolate] WARNING: 'color-matcher' not installed — skipping color match. "
              "Run: pip install color-matcher")
        return image_target

    cm = ColorMatcher()
    ref_cpu    = image_ref.cpu()
    target_cpu = image_target.cpu()
    B          = target_cpu.shape[0]
    out        = []

    for i in range(B):
        ref_np    = ref_cpu[i if ref_cpu.shape[0] > 1 else 0].numpy()
        target_np = target_cpu[i].numpy()
        try:
            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            if strength != 1.0:
                result = target_np + strength * (result - target_np)
        except Exception as e:
            print(f"[CRT Isolate] color_match frame {i} failed: {e}")
            result = target_np
        out.append(torch.from_numpy(result))

    return torch.stack(out).clamp_(0, 1).to(image_target.dtype)


# ─── Megapixel rescale ───────────────────────────────────────────────────────

def _scale_to_mp(images, megapixels):
    """Scale (B,H,W,C) to target megapixels, preserving aspect ratio. 0 = no-op."""
    if megapixels <= 0:
        return images
    B, H, W, C = images.shape
    current_mp = (H * W) / 1_000_000
    if abs(current_mp - megapixels) < 0.01:
        return images
    scale = (megapixels / current_mp) ** 0.5
    new_h = max(16, round(H * scale / 16) * 16)
    new_w = max(16, round(W * scale / 16) * 16)
    scaled = comfy.utils.common_upscale(
        images.movedim(-1, 1), new_w, new_h, "lanczos", crop="disabled"
    ).movedim(1, -1)
    return scaled


def _scale_for_detection(images, megapixels):
    megapixels = float(megapixels)
    if megapixels <= 0:
        return images
    return _scale_to_mp(images, megapixels)


def _scale_for_sam_detection(images, megapixels, max_side=1008):
    megapixels = float(megapixels)
    _, H, W, _ = images.shape
    current_side = max(int(H), int(W))
    if megapixels <= 0:
        target_side = min(int(max_side), current_side)
    else:
        target_side = int(round((max(1e-6, megapixels) * 1_000_000) ** 0.5 / 16) * 16)
        target_side = max(16, min(int(max_side), target_side, current_side))

    if int(H) == target_side and int(W) == target_side:
        return images

    scaled = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        target_side,
        target_side,
        "lanczos",
        crop="disabled",
    ).movedim(1, -1)
    return scaled


def _sam_safe_side_from_shape(height, width, max_side=1008):
    side = max(16, min(int(max_side), int(max(height, width))))
    return max(16, int(round(side / 16) * 16))


def _resize_instance_masks_to_shape(mask_groups, target_h, target_w):
    resized = []
    for group in mask_groups:
        if group.shape[0] == 0:
            resized.append(torch.zeros(0, target_h, target_w, device=group.device, dtype=group.dtype))
            continue
        out = F.interpolate(
            group.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        resized.append((out > 0.5).to(group.dtype))
    return resized


def _resize_batch_masks_to_shape(masks, target_h, target_w):
    if masks.shape[-2:] == (target_h, target_w):
        return masks
    return F.interpolate(
        masks.unsqueeze(1),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _preferred_compute_device():
    try:
        device = comfy.model_management.get_torch_device()
        if device is not None and str(device).startswith("cuda"):
            return device
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("[CRT Isolate] CUDA is required for GPU-only processing.")


def _chunk_size_for_batch(batch_size, height, width, device):
    if not str(device).startswith("cuda"):
        raise RuntimeError("[CRT Isolate] CUDA is required for GPU-only processing.")

    pixels = max(1, int(height) * int(width))
    if pixels >= 1_500_000:
        return min(batch_size, 4)
    if pixels >= 900_000:
        return min(batch_size, 6)
    return min(batch_size, 8)


def _effective_chunk_size(batch_size, height, width, device, requested_chunk_size=0):
    batch_size = max(1, int(batch_size))
    requested = int(requested_chunk_size)
    if requested == 0:
        return batch_size
    if requested >= 1:
        return min(batch_size, requested)
    return _chunk_size_for_batch(batch_size, height, width, device)


# ─── Node: CRT_IsolateInput ───────────────────────────────────────────────────

class CRT_IsolateInput:
    """
    Pads the image batch, uses SAM 3.1 to detect a subject,
    crops around it, and passes a reconstruction pipe to CRT_IsolateOutput.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":            ("IMAGE",),
                "what_to_detect":    ("STRING",  {"default": "face",  "multiline": False, "tooltip": "Text prompt used for subject detection, for example 'face' or 'person'."}),
                "performance_preset": (ISOLATE_PERFORMANCE_PRESETS, {"default": "Quality", "tooltip": "Fast = quickest detection, Balanced = default tradeoff, Quality = highest available quality in the current custom-node path and slower processing. The SAM path still uses a model-safe square working resolution."}),
                "single_item":       ("BOOLEAN", {"default": False, "tooltip": "Limit detection to one item per frame. Enable this when you only want the main face/subject."}),
                "detect_chunk_size": ("INT",     {"default": 0,  "min": 0,   "max": 4096, "step": 1, "tooltip": "Detection batch size. 0 = process the whole batch at once, 1 = per-image processing, 2+ = fixed chunk size."}),
                "padding":           ("INT",     {"default": 512, "min": 0,   "max": 2048, "step": 64, "tooltip": "Extra border added around the input batch before detection so subjects near the edges are not cropped too tightly."}),
                "threshold":         ("FLOAT",   {"default": 0.50, "min": 0.0, "max": 1.0,  "step": 0.01, "tooltip": "Detection confidence threshold. Lower values find more masks, higher values are stricter."}),
                "bbox_expansion":    ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 4.0,  "step": 0.05, "tooltip": "Expands the detected crop box around the subject. Higher values give a looser crop."}),
                "crop_smooth_alpha": ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.05, "tooltip": "Temporal smoothing for crop position. Lower = steadier crop, higher = follows motion more closely."}),
                "crop_megapixels":   ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 8.0,  "step": 0.25, "tooltip": "Target resolution for the cropped output images. 0 disables crop rescaling."}),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        images,
        what_to_detect,
        performance_preset,
        single_item,
        detect_chunk_size,
        padding,
        threshold,
        bbox_expansion,
        crop_smooth_alpha,
        crop_megapixels,
    ):
        return stable_fingerprint(
            images,
            what_to_detect,
            str(performance_preset),
            bool(single_item),
            int(detect_chunk_size),
            int(padding),
            float(threshold),
            float(bbox_expansion),
            float(crop_smooth_alpha),
            float(crop_megapixels),
        )

    RETURN_TYPES  = ("IMAGE", "CRT_ISOLATE_PIPE")
    RETURN_NAMES  = ("cropped_images", "pipe")
    FUNCTION      = "execute"
    CATEGORY      = "CRT/Utils/Isolate"

    def execute(self, images, what_to_detect, performance_preset, single_item, detect_chunk_size, padding,
                threshold, bbox_expansion, crop_smooth_alpha, crop_megapixels):
        start_time = time.perf_counter()
        sam3_checkpoint = SAM31_DEFAULT_CHECKPOINT
        fill_mask = True

        preset = _resolve_isolate_preset(
            performance_preset,
            refine_iterations=2,
        )
        detect_megapixels = preset["detect_megapixels"]
        reuse_first_pass_masks = preset["reuse_first_pass_masks"]
        refine_iterations = preset["refine_iterations"]

        cache_key = stable_fingerprint(
            images,
            what_to_detect,
            str(performance_preset),
            bool(single_item),
            int(detect_chunk_size),
            int(padding),
            float(threshold),
            float(bbox_expansion),
            float(crop_smooth_alpha),
            float(crop_megapixels),
        )
        cached = _get_sam31_cache_entry(cache_key)
        if cached is not None:
            elapsed = time.perf_counter() - start_time
            _print_timing("Isolate Input SAM3.1", elapsed, images.shape[0], cache_hit=True)
            return {
                **_timing_ui("Isolate Input SAM3.1", elapsed, images.shape[0], cache_hit=True),
                "result": cached,
            }

        B, H, W, C = images.shape

        # Detect directly on original images — avoids allocating a giant padded
        # tensor that can exhaust RAM on large video batches. Edge subjects near
        # the frame border get slightly tighter crops; for subjects within the
        # frame this produces identical results.
        detect_input = _scale_for_sam_detection(images, float(detect_megapixels))

        # 2. Load SAM3.1 once — detect then crop, then unload
        model, clip = _load_sam31(sam3_checkpoint)
        try:
            pad_mask_groups = _segment_batch_instances(
                model, clip, detect_input, what_to_detect,
                float(threshold), int(refine_iterations), bool(single_item),
                chunk_size_override=int(detect_chunk_size),
            )
            del detect_input
            pad_mask_groups = _resize_instance_masks_to_shape(
                pad_mask_groups, H, W,
            )
            if bool(fill_mask):
                pad_mask_groups = [
                    _fill_mask_holes(group) if group.shape[0] > 0 else group
                    for group in pad_mask_groups
                ]
            # 3. Crop around detections (directly from original images)
            cropped, crop_masks, bboxes, source_indices, crop_edge_pads = _batch_crop_per_item(
                images, pad_mask_groups,
                crop_size_mult=float(bbox_expansion),
                smooth_alpha=float(crop_smooth_alpha),
            )
            del pad_mask_groups
            gc.collect()
            if bool(reuse_first_pass_masks):
                face_masks = crop_masks.float()
            else:
                detect_cropped = _scale_for_sam_detection(cropped, float(detect_megapixels))
                # 4. Detect face in the crop (pre-inference) for compositing later
                face_mask_groups = _segment_batch_instances(
                    model, clip, detect_cropped, what_to_detect,
                    float(threshold), int(refine_iterations), False,
                    chunk_size_override=int(detect_chunk_size),
                )
                del detect_cropped
                face_mask_groups = _resize_instance_masks_to_shape(
                    face_mask_groups,
                    int(cropped.shape[1]),
                    int(cropped.shape[2]),
                )
                face_masks = []
                for i, group in enumerate(face_mask_groups):
                    if group.shape[0] > 0:
                        reference = crop_masks[i].to(group.device, dtype=torch.float32)
                        candidates = (group > 0).float()
                        overlaps = (candidates * reference.unsqueeze(0)).flatten(1).sum(dim=1)
                        areas = candidates.flatten(1).sum(dim=1) + reference.sum().clamp_min(1e-8) - overlaps
                        best_idx = torch.argmax(overlaps / areas.clamp_min(1e-8)).item()
                        mask = candidates[best_idx].float()
                    else:
                        mask = crop_masks[i].float()
                    face_masks.append(mask)
                del face_mask_groups
                face_masks = torch.stack(face_masks)
            if bool(fill_mask):
                face_masks = _fill_mask_holes(face_masks)
        finally:
            _release_sam31(model, clip)

        # 5. Rescale crop to target megapixels for inference
        cropped_scaled = _scale_to_mp(cropped, float(crop_megapixels))

        pipe = {
            "original_images":  images,
            "original_crops":   cropped,
            "face_masks":       face_masks.cpu(),
            "source_indices":   source_indices,
            "orig_w": W, "orig_h": H,
            "padding":          0,
            "bboxes":           bboxes,
            "crop_masks":       crop_masks,
            "crop_edge_pads":   crop_edge_pads,
            "what_to_detect":   what_to_detect,
        }

        result = (cropped_scaled, pipe)
        _store_sam31_cache_entry(cache_key, result)
        elapsed = time.perf_counter() - start_time
        _print_timing("Isolate Input SAM3.1", elapsed, images.shape[0])
        return {
            **_timing_ui("Isolate Input SAM3.1", elapsed, images.shape[0]),
            "result": result,
        }


# ─── Node: CRT_IsolateOutput ──────────────────────────────────────────────────

class CRT_IsolateOutput:
    """
    Pure compositing — no SAM3 inference.
    Uses the face_masks pre-computed by CRT_IsolateInput on the original crop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enhanced_images":      ("IMAGE",),
                "pipe":                 ("CRT_ISOLATE_PIPE",),
                "smooth_range":         ("INT",   {"default": 5,    "min": 0,   "max": 32,    "step": 1}),
                "expand_mask":          ("INT",   {"default": 8,    "min": 0,   "max": 100,   "step": 1}),
                "blur_radius":          ("FLOAT", {"default": 16.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "border_blending":      ("FLOAT", {"default": 0.20,  "min": 0.0, "max": 1.0,   "step": 0.01}),
                "color_match_method":   (_CM_METHODS, {"default": "mkl"}),
                "color_match_strength": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0,   "step": 0.05}),
                "resize_back_to_original": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If disabled, upscale the original canvas to keep enhanced crop detail",
                    },
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("images_batch", "enhanced_images_masked")
    FUNCTION      = "execute"
    CATEGORY      = "CRT/Utils/Isolate"

    def execute(self, enhanced_images, pipe, smooth_range,
                expand_mask, blur_radius, border_blending,
                color_match_method, color_match_strength,
                resize_back_to_original=True):
        start_time = time.perf_counter()
        original_images = pipe["original_images"]
        original_crops  = pipe["original_crops"]
        face_masks      = pipe["face_masks"]
        source_indices  = pipe["source_indices"]
        orig_w, orig_h  = pipe["orig_w"], pipe["orig_h"]
        padding         = pipe["padding"]
        bboxes          = pipe["bboxes"]
        crop_edge_pads  = pipe.get("crop_edge_pads")  # None for pipes from older node versions

        trimmed, _ = _trim_batch_items({
            "enhanced_images": enhanced_images,
            "original_crops": original_crops,
            "face_masks": face_masks,
            "source_indices": source_indices,
            "bboxes": bboxes,
        })
        enhanced_images = trimmed["enhanced_images"]
        original_crops = trimmed["original_crops"]
        face_masks = trimmed["face_masks"]
        source_indices = trimmed["source_indices"]
        bboxes = trimmed["bboxes"]
        if crop_edge_pads is not None:
            target = len(bboxes)
            crop_edge_pads = crop_edge_pads[:target]

        target_frame_count = int(enhanced_images.shape[0])
        if int(original_images.shape[0]) != target_frame_count:
            print(
                f"[CRT Isolate] Trimming original frame batch from {int(original_images.shape[0])} "
                f"to {target_frame_count} to match enhanced_images."
            )
            original_images = original_images[:target_frame_count]

        # 1. Match working resolution. Current behavior downsamples enhanced crops;
        # optionally upscale the original canvas instead to preserve enhanced detail.
        base_ch, base_cw = original_crops.shape[1], original_crops.shape[2]
        eh, ew = enhanced_images.shape[1], enhanced_images.shape[2]
        if bool(resize_back_to_original):
            target_h, target_w = base_ch, base_cw
            scale_y, scale_x = 1.0, 1.0
        else:
            target_h, target_w = eh, ew
            scale_y = float(eh) / float(max(1, base_ch))
            scale_x = float(ew) / float(max(1, base_cw))

        if eh != target_h or ew != target_w:
            enh = F.interpolate(
                enhanced_images.permute(0, 3, 1, 2),
                size=(target_h, target_w), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            enh = enhanced_images

        ch, cw = target_h, target_w
        if original_crops.shape[1] != ch or original_crops.shape[2] != cw:
            original_crops = F.interpolate(
                original_crops.permute(0, 3, 1, 2),
                size=(ch, cw), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        if not bool(resize_back_to_original):
            target_canvas_h = max(1, int(round(int(original_images.shape[1]) * scale_y)))
            target_canvas_w = max(1, int(round(int(original_images.shape[2]) * scale_x)))
            if target_canvas_h != int(original_images.shape[1]) or target_canvas_w != int(original_images.shape[2]):
                original_images = F.interpolate(
                    original_images.permute(0, 3, 1, 2),
                    size=(target_canvas_h, target_canvas_w), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
            bboxes = [
                (
                    int(round(float(x) * scale_x)),
                    int(round(float(y) * scale_y)),
                    max(1, int(round(float(w) * scale_x))),
                    max(1, int(round(float(h) * scale_y))),
                )
                for x, y, w, h in bboxes
            ]
            if crop_edge_pads is not None:
                crop_edge_pads = [
                    (
                        int(round(float(pl) * scale_x)),
                        int(round(float(pr) * scale_x)),
                        int(round(float(pt) * scale_y)),
                        int(round(float(pb) * scale_y)),
                    )
                    for pl, pr, pt, pb in crop_edge_pads
                ]

        # 2. Color match enhanced → reference is original crop (pre-inference)
        enh = _color_match(original_crops, enh, color_match_method, float(color_match_strength))

        work_device = _preferred_compute_device()
        chunk_size = _chunk_size_for_batch(face_masks.shape[0], ch, cw, work_device)

        # 3. Resize face_masks to crop size if needed (they were computed on original crop)
        fh, fw = face_masks.shape[1], face_masks.shape[2]
        if fh != ch or fw != cw:
            face_masks = F.interpolate(
                face_masks.unsqueeze(1), size=(ch, cw), mode="bilinear", align_corners=False,
            ).squeeze(1)

        face_masks = face_masks.to(work_device)
        face_masks = _temporal_smooth_mask(
            face_masks,
            source_indices,
            int(smooth_range),
        )

        # 3. Grow + blur
        composited_chunks = []
        enhanced_masked_chunks = []
        for start in range(0, face_masks.shape[0], chunk_size):
            end = min(face_masks.shape[0], start + chunk_size)
            mask_chunk = face_masks[start:end]
            enh_chunk = enh[start:end].to(work_device)
            crop_chunk = original_crops[start:end].to(work_device)
            mask_chunk = _grow_blur_mask(mask_chunk, int(expand_mask), float(blur_radius))
            alpha = mask_chunk.unsqueeze(-1)
            composited_chunk = enh_chunk * alpha + crop_chunk * (1.0 - alpha)
            enhanced_masked_chunk = enh_chunk * alpha + (1.0 - alpha)
            composited_chunks.append(composited_chunk.cpu())
            enhanced_masked_chunks.append(enhanced_masked_chunk.cpu())
        composited = torch.cat(composited_chunks, dim=0)
        enhanced_images_masked = torch.cat(enhanced_masked_chunks, dim=0)

        # 5. Uncrop back into canvas. padding=0 means bboxes are in original image
        # coordinates and no stripping is needed — just use original_images directly.
        canvas_padding = int(padding)
        if padding > 0 and not bool(resize_back_to_original):
            canvas_padding = int(round(float(padding) * max(scale_x, scale_y)))
        if canvas_padding > 0:
            canvas = _pad(original_images, canvas_padding)
        else:
            canvas = original_images
        original_images = None
        uncropped = _batch_uncrop_grouped(
            canvas,
            composited, bboxes,
            source_indices,
            border_blending=float(border_blending),
            crop_edge_pads=crop_edge_pads,
        )
        del canvas

        # 6. Strip padding (only when canvas was pre-padded)
        if canvas_padding > 0 and bool(resize_back_to_original):
            result = uncropped[:, canvas_padding:canvas_padding+orig_h, canvas_padding:canvas_padding+orig_w, :]
        elif canvas_padding > 0:
            scaled_orig_w = int(round(float(orig_w) * scale_x))
            scaled_orig_h = int(round(float(orig_h) * scale_y))
            result = uncropped[
                :,
                canvas_padding:canvas_padding+scaled_orig_h,
                canvas_padding:canvas_padding+scaled_orig_w,
                :,
            ]
        else:
            result = uncropped

        elapsed = time.perf_counter() - start_time
        _print_timing("Isolate Output", elapsed, result.shape[0])
        return {
            **_timing_ui("Isolate Output", elapsed, result.shape[0]),
            "result": (result, enhanced_images_masked),
        }


# ─── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CRT_IsolateInput":  CRT_IsolateInput,
    "CRT_IsolateOutput": CRT_IsolateOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_IsolateInput":  "Isolate Input (CRT)",
    "CRT_IsolateOutput": "Isolate Output (CRT)",
}

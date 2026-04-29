import gc
import math
import time
import builtins
import logging
from contextlib import redirect_stderr, redirect_stdout
from contextlib import contextmanager
import io

import numpy as np
import torch
import torch.nn as nn

import comfy.samplers
import comfy.model_management as mm
import comfy.utils
import latent_preview
import nodes
import node_helpers
import folder_paths
from PIL import Image
from comfy.sd import VAE
from comfy.taesd.taehv import TAEHV

from comfy_extras.nodes_custom_sampler import (
    CFGGuider,
    KSamplerSelect,
    ManualSigmas,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVConcatAVLatent,
    LTXVConditioning,
    LTXVAddGuide,
    LTXVCropGuides,
    LTXVImgToVideoInplace,
    LTXVPreprocess,
    LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_audio import (
    LTXVAudioVAEDecode,
    LTXVAudioVAEEncode,
    LTXVEmptyLatentAudio,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

from ._cache_fingerprint import stable_fingerprint
from .Isolate import (
    SAM31_DEFAULT_CHECKPOINT,
    _load_sam31,
    _release_sam31,
    _resize_batch_masks_to_shape,
    _scale_for_sam_detection,
    _segment_batch,
)


_LTX_MOUTH_MASK_CACHE = {}
_LTX_MOUTH_MASK_CACHE_ORDER = []
_LTX_MOUTH_MASK_CACHE_LIMIT = 0
_LTX23_DEPTH_PREVIEW_ROUTE_REGISTERED = globals().get("_LTX23_DEPTH_PREVIEW_ROUTE_REGISTERED", False)

# ── Depth preview storage + API route ────────────────────────────────────────
_ltx23_depth_preview = None  # PIL Image of last depth guide first-frame

try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web

    if not _LTX23_DEPTH_PREVIEW_ROUTE_REGISTERED:
        @PromptServer.instance.routes.get("/crt/ltx23/depth_preview")
        async def _ltx23_depth_preview_route(request):
            global _ltx23_depth_preview
            if _ltx23_depth_preview is None:
                return _aiohttp_web.Response(status=404, text="No depth preview yet - run the sampler first.")
            buf = io.BytesIO()
            _ltx23_depth_preview.save(buf, format="PNG")
            buf.seek(0)
            return _aiohttp_web.Response(body=buf.read(), content_type="image/png")

        _LTX23_DEPTH_PREVIEW_ROUTE_REGISTERED = True
except Exception as _route_e:
    print(f"[CRT LTX23] depth_preview route not registered: {_route_e}")


class _CRTPreviewState:
    last_latent_shapes = None
    fps_override = None


class _CRTPreviewFixGuider:
    def __init__(self, guider, fps_override=None):
        self._guider = guider
        self._fps_override = fps_override

    def __getattr__(self, key):
        return getattr(self._guider, key)

    def sample(self, noise, latent_image, *args, **kwargs):
        latent_shapes = (
            (tuple(latent_image.shape),)
            if not getattr(latent_image, "is_nested", False)
            else tuple(tuple(t.shape) for t in latent_image.unbind())
        )
        _CRT_PREVIEW_STATE.last_latent_shapes = latent_shapes

        if self._fps_override:
            _CRT_PREVIEW_STATE.fps_override = float(self._fps_override)

        try:
            return self._guider.sample(noise, latent_image, *args, **kwargs)
        finally:
            _CRT_PREVIEW_STATE.last_latent_shapes = None
            _CRT_PREVIEW_STATE.fps_override = None


def _crt_build_taehv(model_path):
    """Build standard TAEHV(latent_channels=128) and load state dict directly."""
    taehv = TAEHV(latent_channels=128)
    sd = comfy.utils.load_torch_file(model_path)
    taehv.load_state_dict(sd, strict=True)
    taehv.eval()
    taehv.show_progress_bar = False
    return taehv


class _CRTLTXPreviewer(latent_preview.TAEHVPreviewerImpl):
    _blank = Image.new("RGB", (256, 256), (8, 8, 8))

    def __init__(self, wide_model):
        # TAEHVPreviewerImpl just does self.taesd = taesd; we bypass VAE and use model directly
        self.taesd = None
        self._wide_model = wide_model

    @staticmethod
    def _ensure_video_latent_shape(x0):
        if not hasattr(x0, "shape"):
            return None
        if getattr(x0, "is_nested", False):
            try:
                x0 = x0.unbind()[0]
            except Exception:
                return None

        if x0.ndim == 5 and x0.shape[0] > 0:
            return x0
        if x0.ndim == 4 and x0.shape[0] > 0:
            return x0.unsqueeze(2)

        last_shapes = _CRT_PREVIEW_STATE.last_latent_shapes
        if not last_shapes or not hasattr(comfy.utils, "unpack_latents"):
            return None

        try:
            last_numel = sum(math.prod(shape) for shape in last_shapes)
            if int(last_numel) != int(x0.numel()):
                return None
            unpacked = comfy.utils.unpack_latents(x0, last_shapes)
            if not unpacked:
                return None
            target = unpacked[0]
            if target.ndim == 4:
                target = target.unsqueeze(2)
            if target.ndim == 5 and target.shape[0] > 0:
                return target
        except Exception:
            return None

        return None

    def decode_latent_to_preview(self, x0):
        fixed = self._ensure_video_latent_shape(x0)
        if fixed is None:
            return self._blank
        try:
            inp = fixed[:1]  # (1, C, F, H, W)
            # Decode the middle frame as a single representative preview
            mid = inp.shape[2] // 2
            with torch.no_grad():
                out = self._wide_model.to(inp.device).decode(inp[:, :, mid : mid + 1])
            frame = out[0, :, 0].float().cpu().movedim(0, -1)  # (H, W, 3)
            return latent_preview.preview_to_image(frame, do_scale=False)
        except Exception as e:
            import traceback

            print(f"[CRT-preview] decode error: {e}")
            traceback.print_exc()
            return self._blank


_CRT_PREVIEW_STATE = _CRTPreviewState()
_CRT_ORIG_GET_PREVIEWER = latent_preview.get_previewer
_CRT_PREVIEW_PATCHED = False
_CRT_PREVIEWER_CACHE = {}


def _crt_is_ltx_format(latent_format):
    format_name = latent_format.__class__.__name__.lower()
    decoder_name = str(getattr(latent_format, "taesd_decoder_name", "") or "").lower()
    return ("ltx" in format_name) or decoder_name.startswith("taeltx")


def _crt_find_ltx_preview_models(latent_format):
    """Return ordered list of candidate model paths to try. Standard arch before wide."""
    files = folder_paths.get_filename_list("vae_approx")
    lower_files = [(fn, fn.lower()) for fn in files]
    decoder_name = str(getattr(latent_format, "taesd_decoder_name", "") or "").lower()

    # Non-wide standard TAEHV first (loads with unmodified TAEHV class),
    # wide/fallbacks after. Each prefix matches the first filename starting with it.
    prefixes = [
        "taeltx2_3",  # matches taeltx2_3.safetensors (alphabetically before _wide)
        "taeltx_2_3",
        "taeltx2_3_wide",
        "taeltx_2_3_wide",
        "taeltx2",
        "taeltx_2",
    ]
    if decoder_name:
        prefixes.insert(0, decoder_name)

    seen_paths = set()
    result = []
    for prefix in prefixes:
        for fn, lower in lower_files:
            if lower.startswith(prefix.lower()):
                full = folder_paths.get_full_path("vae_approx", fn)
                if full and full not in seen_paths:
                    seen_paths.add(full)
                    result.append(full)
                break
    return result


def _crt_get_previewer(device, latent_format, *args, **kwargs):
    fallback = _CRT_ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)
    if not _crt_is_ltx_format(latent_format):
        return fallback

    for model_path in _crt_find_ltx_preview_models(latent_format):
        cached = _CRT_PREVIEWER_CACHE.get(model_path)
        if cached is not None:
            return _CRTLTXPreviewer(cached)
        try:
            model = _crt_build_taehv(model_path)
            _CRT_PREVIEWER_CACHE[model_path] = model
            return _CRTLTXPreviewer(model)
        except Exception as e:
            print(f"[CRT-preview] skipping {model_path}: {e}")
            continue

    return fallback


def ensure_crt_previewer():
    global _CRT_PREVIEW_PATCHED
    if _CRT_PREVIEW_PATCHED:
        return
    latent_preview.get_previewer = _crt_get_previewer
    _CRT_PREVIEW_PATCHED = True
    print("[CRT-preview] patched latent_preview.get_previewer")


def _append_guide_attention_entry(
    positive, negative, pre_filter_count, latent_shape, strength=1.0
):
    new_entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": None,
        "latent_shape": latent_shape,
    }
    results = []
    for cond in (positive, negative):
        existing = []
        for t in cond:
            found = t[1].get("guide_attention_entries", None)
            if found is not None:
                existing = found
                break
        entries = [*existing, new_entry]
        results.append(
            node_helpers.conditioning_set_values(
                cond,
                {"guide_attention_entries": entries},
            )
        )
    return results[0], results[1]


class CRT_LTX23USModelsPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_union_control": (
                    "MODEL",
                    {
                        "forceInput": True,
                        "tooltip": "V2V Depth Control: model already merged with UnionControl / IC-LoRA guide support.",
                    },
                ),
                "model_outpaint": (
                    "MODEL",
                    {
                        "forceInput": True,
                        "tooltip": "V2V Outpaint mode: model merged with IC-LoRA Outpaint.",
                    },
                ),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "clip": ("CLIP",),
            },
            "optional": {
                "spatial_upscale_model": ("LATENT_UPSCALE_MODEL",),
                "da3_model": ("DA3MODEL",),
                "latent_downscale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 16.0,
                        "step": 1.0,
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX23_US_MODELS_PIPE",)
    RETURN_NAMES = ("models_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/LTX2.3"

    def build_pipe(
        self,
        model,
        model_union_control,
        model_outpaint,
        vae,
        audio_vae,
        clip,
        spatial_upscale_model=None,
        da3_model=None,
        latent_downscale_factor=1.0,
    ):
        pipe = {
            "model": model,
            "vae": vae,
            "audio_vae": audio_vae,
            "clip": clip,
            "spatial_upscale_model": spatial_upscale_model,
            "da3_model": da3_model,
            "model_union_control": model_union_control,
            "model_outpaint": model_outpaint,
            "latent_downscale_factor": float(latent_downscale_factor),
        }
        return (pipe,)


class CRT_LTX23USConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            },
            "optional": {
                "Image I2V / V2V FirstFrame": ("IMAGE",),
                "Video (V2V image batch)": ("IMAGE",),
                "source_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("LTX23_US_CONFIG_PIPE",)
    RETURN_NAMES = ("config_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/LTX2.3"

    def build_pipe(self, prompt, seed, source_audio=None, **kwargs):
        framerate_value = 24.0

        image = kwargs.get("Image I2V / V2V FirstFrame", None)
        if image is not None and image.shape[0] > 1:
            image = image[:1]
        video = kwargs.get("Video (V2V image batch)", None)

        pipe = {
            "prompt": str(prompt),
            "seed": int(seed),
            "framerate": framerate_value,
            "image": image,
            "video": video,
            "source_audio": source_audio,
        }
        return (pipe,)


class CRT_LTX23UnifiedSampler:
    COLOR_INFO = "\033[38;5;117m"
    COLOR_WARN = "\033[38;5;208m"
    COLOR_OK = "\033[38;5;120m"
    COLOR_RESET = "\033[0m"
    MOUTH_DETECT_MEGAPIXELS_FAST = 0.25
    MOUTH_DETECT_SINGLE_ITEM = True
    MOUTH_DETECT_CHUNK_SIZE = 10
    MOUTH_MASK_MAX_COVERAGE = 0.20
    ASPECT_RATIOS = [
        "1:1 (Square)",
        "2:3 (Portrait)",
        "3:4 (Portrait)",
        "4:5 (Portrait)",
        "5:7 (Portrait)",
        "5:8 (Portrait)",
        "7:9 (Portrait)",
        "9:16 (Portrait)",
        "9:19 (Portrait)",
        "9:21 (Portrait)",
        "3:2 (Landscape)",
        "4:3 (Landscape)",
        "5:3 (Landscape)",
        "5:4 (Landscape)",
        "7:5 (Landscape)",
        "8:5 (Landscape)",
        "9:7 (Landscape)",
        "16:9 (Landscape)",
        "19:9 (Landscape)",
        "21:9 (Landscape)",
    ]

    SIGMAS_MAIN_DEFAULT = (
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    )
    SIGMAS_REFINE_DEFAULT = "0.85, 0.7250, 0.4219, 0.0"
    CFG_DEFAULT = 1.0
    I2V_STRENGTH = 1.0
    I2V_PREPROCESS_COMPRESSION = 25

    @classmethod
    def IS_CHANGED(
        cls,
        models_pipe,
        config_pipe,
        workflow_mode,
        hq,
        live_preview,
        frame_count_from_audio,
        vae_decode_tiled,
        megapixels_target,
        aspect_ratio,
        frame_count,
        v2v_mode,
        v2v_guide_strength,
        depth_megapixels,
        v2v_aspect_ratio,
        sampler_main,
        sampler_refine,
        steps,
        generated_audio_gain_db,
        firstframe_strength,
        depth_mouth_mask,
        mouth_detect_megapixels,
        mouth_single_item,
        mouth_detect_chunk_size,
        mouth_mask_expand,
        mouth_mask_blur,
    ):
        return stable_fingerprint(
            models_pipe,
            config_pipe,
            workflow_mode,
            bool(hq),
            bool(live_preview),
            bool(frame_count_from_audio),
            bool(vae_decode_tiled),
            float(megapixels_target),
            aspect_ratio,
            int(frame_count),
            v2v_mode,
            float(v2v_guide_strength),
            float(depth_megapixels),
            v2v_aspect_ratio,
            sampler_main,
            sampler_refine,
            int(steps),
            float(generated_audio_gain_db),
            float(firstframe_strength),
            bool(depth_mouth_mask),
            float(mouth_detect_megapixels),
            bool(mouth_single_item),
            int(mouth_detect_chunk_size),
            int(mouth_mask_expand),
            float(mouth_mask_blur),
        )

    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = list(comfy.samplers.SAMPLER_NAMES)
        sampler_main_default = (
            "euler_ancestral_cfg_pp"
            if "euler_ancestral_cfg_pp" in sampler_names
            else sampler_names[0]
        )
        sampler_refine_default = (
            "euler_cfg_pp" if "euler_cfg_pp" in sampler_names else sampler_names[0]
        )

        return {
            "required": {
                "models_pipe": ("LTX23_US_MODELS_PIPE",),
                "config_pipe": ("LTX23_US_CONFIG_PIPE",),
                "workflow_mode": (
                    ["I2V", "T2V", "V2V"],
                    {"default": "I2V"},
                ),
                "hq": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "live_preview": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "frame_count_from_audio": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "vae_decode_tiled": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "megapixels_target": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                    },
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {"default": "3:2 (Landscape)"},
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 161,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "v2v_mode": (
                    ["Depth Control", "Outpaint"],
                    {"default": "Depth Control"},
                ),
                "v2v_guide_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "depth_megapixels": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.05,
                        "max": 8.0,
                        "step": 0.05,
                    },
                ),
                "v2v_aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {"default": "16:9 (Landscape)"},
                ),
                "sampler_main": (
                    sampler_names,
                    {"default": sampler_main_default},
                ),
                "sampler_refine": (
                    sampler_names,
                    {"default": sampler_refine_default},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 9,
                        "min": 4,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "generated_audio_gain_db": (
                    "FLOAT",
                    {
                        "default": -12.0,
                        "min": -60.0,
                        "max": 24.0,
                        "step": 0.1,
                    },
                ),
                "firstframe_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "depth_mouth_mask": ("BOOLEAN", {"default": False}),
                "mouth_detect_megapixels": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "Deprecated compatibility input. Ignored; mouth detection uses hardcoded fast mode."}),
                "mouth_single_item": ("BOOLEAN", {"default": True, "tooltip": "Deprecated compatibility input. Ignored; hardcoded internally."}),
                "mouth_detect_chunk_size": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1, "tooltip": "Deprecated compatibility input. Ignored; hardcoded internally."}),
                "mouth_mask_expand": ("INT",   {"default": 8,   "min": 0, "max": 100, "step": 1, "tooltip": "Expands the mouth mask before applying it to the depth guide so lip motion influence covers a larger area."}),
                "mouth_mask_blur":   ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "tooltip": "Softens the mouth mask edges to avoid harsh transitions in the depth guide."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "sample"
    CATEGORY = "CRT/LTX2.3"

    @staticmethod
    def _result_tuple(result):
        if result is None:
            return tuple()
        if isinstance(result, tuple):
            return result
        if isinstance(result, list):
            return tuple(result)
        if hasattr(result, "result"):
            node_result = getattr(result, "result")
            if node_result is None:
                return tuple()
            if isinstance(node_result, tuple):
                return node_result
            if isinstance(node_result, list):
                return tuple(node_result)
            return (node_result,)

        try:
            return tuple(result)
        except Exception:
            pass

        return (result,)

    _DA3_SUPPRESS_PREFIXES = (
        "[DepthAnythingV3]",
        "[VRAM",
        "[build]",
        "Model input size",
        "Input image size",
    )

    @staticmethod
    @contextmanager
    def _quiet_depthanything_logs():
        original_print = builtins.print
        tracked = []

        def filtered_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            for pfx in CRT_LTX23UnifiedSampler._DA3_SUPPRESS_PREFIXES:
                if pfx in text:
                    return
            return original_print(*args, **kwargs)

        builtins.print = filtered_print
        previous_disable = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            depth_logger = logging.getLogger("DepthAnythingV3")
            tracked.append((depth_logger, depth_logger.level, depth_logger.propagate))
            depth_logger.setLevel(logging.CRITICAL + 1)
            depth_logger.propagate = False

            for name in list(logging.root.manager.loggerDict.keys()):
                if "depthanything" not in str(name).lower():
                    continue
                logger = logging.getLogger(name)
                tracked.append((logger, logger.level, logger.propagate))
                logger.setLevel(logging.CRITICAL + 1)
                logger.propagate = False

            yield
        finally:
            logging.disable(previous_disable)
            builtins.print = original_print
            for logger, level, propagate in tracked:
                logger.setLevel(level)
                logger.propagate = propagate

    @classmethod
    def _timed_log(cls, label, fn, level="info"):
        _t0 = time.monotonic()
        result = fn()
        cls._log(f"{label} ({time.monotonic() - _t0:.1f}s)", level=level)
        return result

    @classmethod
    def _detect_mouth_masks(cls, images, detect_megapixels=0.0, single_item=True, detect_chunk_size=10):
        batch = int(images.shape[0]) if hasattr(images, "shape") else 0
        pbar = comfy.utils.ProgressBar(max(1, batch))
        detect_mp = float(detect_megapixels)
        single_item = bool(single_item)
        detect_chunk_size = int(max(0, detect_chunk_size))

        try:
            cls._log(f"[Mouth Mask] Loading SAM3.1 checkpoint: {SAM31_DEFAULT_CHECKPOINT}")
            model, clip = _load_sam31(SAM31_DEFAULT_CHECKPOINT)
            try:
                cls._log(f"[Mouth Mask] Model loaded. Segmenting {batch} frames...")
                detect_images = _scale_for_sam_detection(images, detect_mp)
                masks = _segment_batch(
                    model,
                    clip,
                    detect_images,
                    "mouth",
                    0.3,
                    0,
                    single_item=single_item,
                    chunk_size_override=detect_chunk_size,
                )
                masks = _resize_batch_masks_to_shape(
                    masks,
                    int(images.shape[1]),
                    int(images.shape[2]),
                )
            finally:
                _release_sam31(model, clip)

            if masks is None:
                return None

            pbar.update_absolute(max(1, batch), max(1, batch), None)
            cls._log(f"[Mouth Mask] loaded {masks.shape} nonzero={masks.sum():.0f}", level="ok")
            return masks.cpu()
        except Exception as e:
            cls._log(f"[Mouth Mask] Detection failed: {e}", level="warn")
            return None

    @staticmethod
    def _offload_da3_model(da3_model):
        try:
            if isinstance(da3_model, dict):
                model = da3_model.get("model")
                if model is not None and hasattr(model, "to"):
                    model.to(mm.unet_offload_device())
            mm.soft_empty_cache()
        except Exception:
            pass

    @staticmethod
    def _run_external_node(node_name, suppress_output=False, **kwargs):
        node_cls = nodes.NODE_CLASS_MAPPINGS.get(node_name)
        if node_cls is None:
            raise RuntimeError(f"Required node is missing: {node_name}")

        node_obj = node_cls()
        fn_name = getattr(node_cls, "FUNCTION", None)
        if fn_name is None:
            if hasattr(node_obj, "execute"):
                fn_name = "execute"
            elif hasattr(node_obj, "process"):
                fn_name = "process"
            elif hasattr(node_obj, "run"):
                fn_name = "run"
            else:
                raise RuntimeError(f"Could not resolve callable for node: {node_name}")

        fn = getattr(node_obj, fn_name)
        if suppress_output:
            with (
                CRT_LTX23UnifiedSampler._quiet_depthanything_logs(),
                redirect_stderr(io.StringIO()),
            ):
                return fn(**kwargs)
        return fn(**kwargs)

    @staticmethod
    def _latent_spatial_dims(latent):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None or not hasattr(samples, "shape"):
            return None, None
        shape = tuple(samples.shape)
        if len(shape) < 2:
            return None, None
        return int(shape[-1]), int(shape[-2])

    @classmethod
    def _compatible_downscale_factor(cls, latent, requested_factor):
        requested = max(1, int(round(float(requested_factor))))
        width, height = cls._latent_spatial_dims(latent)
        if width is None or height is None:
            return float(requested)

        factor = requested
        while factor > 1 and ((width % factor) != 0 or (height % factor) != 0):
            factor -= 1

        if factor != requested:
            cls._log(
                (
                    f"Adjusted latent_downscale_factor from {requested} to {factor} "
                    f"for latent size {width}x{height} (IC-LoRA divisibility requirement)."
                ),
                level="warn",
            )
        return float(max(1, factor))

    @classmethod
    def _log(cls, message, level="info"):
        color = cls.COLOR_INFO
        if level == "warn":
            color = cls.COLOR_WARN
        elif level == "ok":
            color = cls.COLOR_OK
        print(f"{color}[CRT LTX23]{cls.COLOR_RESET} {message}")

    @classmethod
    def _progress(cls, step, total, label):
        width = 18
        filled = max(0, min(width, int(round((step / float(max(1, total))) * width))))
        bar = "#" * filled + "-" * (width - filled)
        cls._log(f"[{step}/{total}] {bar} {label}")

    @staticmethod
    def _require_pipe_dict(pipe, pipe_name):
        if not isinstance(pipe, dict):
            raise ValueError(f"{pipe_name} is not a valid pipe object.")
        return pipe

    @classmethod
    def _unpack_models_pipe(cls, models_pipe):
        pipe = cls._require_pipe_dict(models_pipe, "models_pipe")

        model = pipe.get("model", None)
        vae = pipe.get("vae", None)
        audio_vae = pipe.get("audio_vae", None)
        clip = pipe.get("clip", None)
        spatial_upscale_model = pipe.get("spatial_upscale_model", None)
        da3_model = pipe.get("da3_model", None)
        model_union_control = pipe.get("model_union_control", None)

        if model is None:
            raise ValueError("models_pipe is missing model.")
        if vae is None:
            raise ValueError("models_pipe is missing vae.")
        if audio_vae is None:
            raise ValueError("models_pipe is missing audio_vae.")
        if clip is None:
            raise ValueError("models_pipe is missing clip.")

        return {
            "model": model,
            "vae": vae,
            "audio_vae": audio_vae,
            "clip": clip,
            "spatial_upscale_model": spatial_upscale_model,
            "da3_model": da3_model,
            "model_union_control": model_union_control,
            "model_outpaint": pipe.get("model_outpaint", None),
            "latent_downscale_factor": float(pipe.get("latent_downscale_factor", 1.0)),
        }

    @classmethod
    def _unpack_config_pipe(cls, config_pipe):
        pipe = cls._require_pipe_dict(config_pipe, "config_pipe")

        prompt = str(pipe.get("prompt", ""))
        seed = int(pipe.get("seed", 0))
        image = pipe.get("image", None)
        video = pipe.get("video", None)
        source_audio = pipe.get("source_audio", None)
        framerate = pipe.get("framerate", pipe.get("fps", 24.0))
        if framerate is None:
            framerate = 24.0
        framerate = float(framerate)

        return {
            "prompt": prompt,
            "seed": seed,
            "image": image,
            "video": video,
            "source_audio": source_audio,
            "framerate": framerate,
        }

    @staticmethod
    def _resize_image(image, width, height, method="lanczos"):
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return image

        _, h, w, _ = image.shape
        if w == width and h == height:
            return image

        return comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, method, "disabled"
        ).movedim(1, -1)

    @staticmethod
    def _resize_crop_cover(image, width, height, method="lanczos"):
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return image

        _, src_h, src_w, _ = image.shape
        if src_w == width and src_h == height:
            return image

        scale = max(width / float(max(1, src_w)), height / float(max(1, src_h)))
        scaled_w = max(width, int(math.ceil(src_w * scale)))
        scaled_h = max(height, int(math.ceil(src_h * scale)))

        upscaled = comfy.utils.common_upscale(
            image.movedim(-1, 1), scaled_w, scaled_h, method, "disabled"
        ).movedim(1, -1)

        x0 = max(0, (scaled_w - width) // 2)
        y0 = max(0, (scaled_h - height) // 2)
        x1 = x0 + width
        y1 = y0 + height
        return upscaled[:, y0:y1, x0:x1, :]

    @staticmethod
    def _log_orange(message):
        print(f"\033[38;5;208m[CRT LTX23]\033[0m {message}")

    @classmethod
    def _prepare_v2v_firstframe(cls, firstframe_image, target_width, target_height):
        if firstframe_image is None:
            return None

        target_width = int(target_width)
        target_height = int(target_height)
        _, src_h, src_w, _ = firstframe_image.shape
        if src_w == target_width and src_h == target_height:
            return firstframe_image

        cls._log_orange(
            "V2V FirstFrame dimensions differ from the video. "
            "The frame was auto-fitted with centered crop/resize to match the video size. "
            "For best results, prepare a first frame with matching dimensions first "
            "(for example with FluxKlein Edit) and use that as V2V FirstFrame."
        )
        return cls._resize_crop_cover(
            firstframe_image,
            target_width,
            target_height,
            method="lanczos",
        )

    @staticmethod
    def _scale_total_pixels(image, megapixels, method="lanczos"):
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        _, h, w, _ = image.shape
        scale = math.sqrt(total_pixels / float(max(1, w * h)))
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        if new_w == w and new_h == h:
            return image

        return comfy.utils.common_upscale(
            image.movedim(-1, 1), new_w, new_h, method, "disabled"
        ).movedim(1, -1)

    @staticmethod
    def _scale_to_multiple_cover(image, multiple, method="lanczos"):
        multiple = int(max(1, multiple))
        if multiple <= 1:
            return image

        _, height, width, _ = image.shape
        target_w = (width // multiple) * multiple
        target_h = (height // multiple) * multiple

        if target_w <= 0 or target_h <= 0:
            return image
        if target_w == width and target_h == height:
            return image

        s_w = target_w / float(width)
        s_h = target_h / float(height)
        if s_w >= s_h:
            scaled_w = target_w
            scaled_h = int(math.ceil(height * s_w))
            if scaled_h < target_h:
                scaled_h = target_h
        else:
            scaled_h = target_h
            scaled_w = int(math.ceil(width * s_h))
            if scaled_w < target_w:
                scaled_w = target_w

        upscaled = comfy.utils.common_upscale(
            image.movedim(-1, 1), scaled_w, scaled_h, method, "disabled"
        ).movedim(1, -1)

        x0 = (scaled_w - target_w) // 2
        y0 = (scaled_h - target_h) // 2
        x1 = x0 + target_w
        y1 = y0 + target_h
        return upscaled[:, y0:y1, x0:x1, :]

    @classmethod
    def _fit_image_batch_frames(cls, images, target_frames, label="image batch"):
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4:
            return images
        target_frames = int(max(1, target_frames))
        current_frames = int(images.shape[0])
        if current_frames == target_frames:
            return images

        if current_frames > target_frames:
            cls._log(
                f"{label} frame count cropped ({current_frames} -> {target_frames})",
                level="warn",
            )
            return images[:target_frames]

        pad_frames = target_frames - current_frames
        last = images[-1:].expand(pad_frames, -1, -1, -1)
        cls._log(
            f"{label} frame count padded ({current_frames} -> {target_frames})",
            level="warn",
        )
        return torch.cat([images, last], dim=0)

    @staticmethod
    def _dims_from_megapixels_aspect(megapixels, aspect_ratio, divisible_by=16):
        ratio_str = str(aspect_ratio).split(" ")[0]
        try:
            width_ratio, height_ratio = map(int, ratio_str.split(":"))
        except Exception:
            width_ratio, height_ratio = 1, 1

        ratio = float(width_ratio) / float(max(height_ratio, 1))
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        width = math.sqrt(total_pixels * ratio)
        height = math.sqrt(total_pixels / max(ratio, 1e-8))

        divisible_by = int(max(1, divisible_by))
        width = max(divisible_by, round(width / divisible_by) * divisible_by)
        height = max(divisible_by, round(height / divisible_by) * divisible_by)

        return int(width), int(height)

    @staticmethod
    def _apply_audio_gain(audio, gain_db):
        if not isinstance(audio, dict):
            return audio
        waveform = audio.get("waveform", None)
        if waveform is None:
            return audio

        gain = 10.0 ** (float(gain_db) / 20.0)
        out = dict(audio)
        out["waveform"] = waveform * gain
        return out

    @staticmethod
    def _build_conditioning(clip, prompt, frame_rate):
        positive = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        conditioned = LTXVConditioning.execute(positive, negative, frame_rate)
        conditioned = CRT_LTX23UnifiedSampler._result_tuple(conditioned)
        return conditioned[0], conditioned[1]

    @staticmethod
    def _make_noise(seed):
        return CRT_LTX23UnifiedSampler._result_tuple(RandomNoise.execute(seed))[0]

    @staticmethod
    def _make_sampler(sampler_name):
        return CRT_LTX23UnifiedSampler._result_tuple(
            KSamplerSelect.execute(sampler_name)
        )[0]

    @staticmethod
    def _make_sigmas(sigmas_text):
        return CRT_LTX23UnifiedSampler._result_tuple(ManualSigmas.execute(sigmas_text))[
            0
        ]

    @staticmethod
    def _sigmas_to_text(values):
        return ", ".join(f"{float(v):.6f}".rstrip("0").rstrip(".") for v in values)

    @classmethod
    def _build_parametric_sigma_texts(cls, steps):
        steps = int(max(4, steps))
        if steps == 1:
            return cls.SIGMAS_MAIN_DEFAULT, cls.SIGMAS_REFINE_DEFAULT

        main_values = []
        for idx in range(steps):
            t = idx / float(steps - 1)
            sigma = 1.0 - (t**3.0)
            main_values.append(max(0.0, min(1.0, sigma)))
        main_values[-1] = 0.0

        refine_steps = max(4, min(steps, int(round(steps * 0.45))))
        tail = main_values[-refine_steps:]
        tail_first = max(tail[0], 1e-6)
        scale = 0.85 / tail_first
        refine_values = [max(0.0, min(0.999, v * scale)) for v in tail]
        refine_values[-1] = 0.0

        return cls._sigmas_to_text(main_values), cls._sigmas_to_text(refine_values)

    @classmethod
    def _sample_latent(
        cls,
        model,
        positive,
        negative,
        cfg,
        noise,
        sampler,
        sigmas,
        latent,
        frame_rate=None,
        preview_enabled=True,
    ):
        if preview_enabled:
            ensure_crt_previewer()

        guider = cls._result_tuple(CFGGuider.execute(model, positive, negative, cfg))[0]
        if preview_enabled:
            guider = _CRTPreviewFixGuider(guider, fps_override=frame_rate)

        sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)
        sampled = cls._result_tuple(sampled)
        # SamplerCustomAdvanced output[0] is the sampled latent used by standard workflows.
        # output[1] is denoised output and can preserve conditioning structure too strongly.
        return sampled[0] if len(sampled) > 0 else latent

    @staticmethod
    def _decode_video_latent(video_latent, vae, use_tiled_decode=False):
        if use_tiled_decode:
            return nodes.VAEDecodeTiled().decode(
                vae,
                video_latent,
                512,
                64,
                64,
                8,
            )[0]
        return nodes.VAEDecode().decode(vae, video_latent)[0]

    @staticmethod
    def _decode_audio_latent(audio_latent, audio_vae):
        return CRT_LTX23UnifiedSampler._result_tuple(
            LTXVAudioVAEDecode.execute(audio_latent, audio_vae)
        )[0]

    @staticmethod
    def _lock_audio_latent(audio_latent):
        if not isinstance(audio_latent, dict):
            return audio_latent
        samples = audio_latent.get("samples", None)
        if samples is None or not hasattr(samples, "shape"):
            return audio_latent
        if len(samples.shape) < 4:
            return audio_latent

        batch = int(samples.shape[0])
        latent_t = int(samples.shape[-2])
        latent_f = int(samples.shape[-1])
        mask = torch.zeros(
            (batch, 1, latent_t, latent_f),
            dtype=torch.float32,
            device=samples.device,
        )

        out = dict(audio_latent)
        out["noise_mask"] = mask
        return out

    @classmethod
    def _fit_audio_latent_to_frames(
        cls, audio_latent, frame_count, frame_rate, audio_vae
    ):
        if not isinstance(audio_latent, dict):
            return audio_latent
        samples = audio_latent.get("samples", None)
        if samples is None or not hasattr(samples, "shape") or len(samples.shape) != 4:
            return audio_latent

        fps_i = max(1, int(round(frame_rate)))
        if hasattr(audio_vae, "num_of_latents_from_frames"):
            target_latents = int(
                audio_vae.num_of_latents_from_frames(
                    int(frame_count),
                    fps_i,
                )
            )
        else:
            probe = cls._result_tuple(
                LTXVEmptyLatentAudio.execute(
                    int(frame_count),
                    fps_i,
                    1,
                    audio_vae,
                )
            )
            if len(probe) == 0 or not isinstance(probe[0], dict):
                raise RuntimeError(
                    "Could not infer target audio latent length for source_audio fitting."
                )
            probe_samples = probe[0].get("samples", None)
            if (
                probe_samples is None
                or not hasattr(probe_samples, "shape")
                or len(probe_samples.shape) != 4
            ):
                raise RuntimeError(
                    "Audio latent length probe returned invalid samples for source_audio fitting."
                )
            target_latents = int(probe_samples.shape[2])
        target_latents = max(1, target_latents)

        current_latents = int(samples.shape[2])
        if current_latents == target_latents:
            return audio_latent

        fitted_samples = samples
        if current_latents > target_latents:
            fitted_samples = samples[:, :, :target_latents, :]
            cls._log(
                (
                    "source_audio latent cropped to match frame count "
                    f"({current_latents} -> {target_latents} latent frames)"
                ),
                level="warn",
            )
        else:
            pad_t = target_latents - current_latents
            fitted_samples = torch.nn.functional.pad(
                samples, (0, 0, 0, pad_t), value=0.0
            )
            cls._log(
                (
                    "source_audio latent padded to match frame count "
                    f"({current_latents} -> {target_latents} latent frames)"
                ),
                level="warn",
            )

        out = dict(audio_latent)
        out["samples"] = fitted_samples
        return out

    @classmethod
    def _build_audio_latent(
        cls,
        frame_count,
        frame_rate,
        audio_vae,
        source_audio=None,
    ):
        if source_audio is not None:
            try:
                encoded = cls._result_tuple(
                    LTXVAudioVAEEncode.execute(source_audio, audio_vae)
                )
                if len(encoded) > 0:
                    encoded_latent = cls._fit_audio_latent_to_frames(
                        encoded[0],
                        frame_count,
                        frame_rate,
                        audio_vae,
                    )
                    encoded_latent = cls._lock_audio_latent(encoded_latent)
                    cls._log(
                        "Using source_audio latent from LTXV Audio VAE Encode (noise-locked)",
                        level="ok",
                    )
                    return encoded_latent
                raise RuntimeError(
                    "LTXV Audio VAE Encode returned no latent outputs for source_audio."
                )
            except Exception as exc:
                raise RuntimeError(
                    "source_audio is connected but encoding failed in LTXV Audio VAE Encode: "
                    f"{exc}"
                ) from exc

        return cls._result_tuple(
            LTXVEmptyLatentAudio.execute(
                int(frame_count),
                max(1, int(round(frame_rate))),
                1,
                audio_vae,
            )
        )[0]

    def _build_initial_av_latent(
        self, width, height, frame_count, frame_rate, audio_vae, source_audio=None
    ):
        video_latent = self._result_tuple(
            EmptyLTXVLatentVideo.execute(int(width), int(height), int(frame_count), 1)
        )[0]

        audio_latent = self._build_audio_latent(
            frame_count,
            frame_rate,
            audio_vae,
            source_audio=source_audio,
        )

        av_latent = self._result_tuple(
            LTXVConcatAVLatent.execute(video_latent, audio_latent)
        )[0]
        return video_latent, audio_latent, av_latent

    @staticmethod
    def _frame_count_from_audio(audio, fps):
        if not isinstance(audio, dict):
            return None
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            return None
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        try:
            sample_rate = int(sample_rate)
        except Exception:
            return None
        if sample_rate <= 0 or waveform.numel() == 0:
            return None

        num_samples = int(waveform.shape[-1])
        duration_seconds = float(num_samples) / float(sample_rate)
        frames = int(duration_seconds * float(fps))
        if frames <= 0 and duration_seconds > 0.0:
            frames = 1
        return max(1, frames)

    @staticmethod
    def _fit_audio_to_frame_count(audio, frame_count, fps):
        if not isinstance(audio, dict):
            return audio
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            return audio

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)

        try:
            sample_rate = int(sample_rate)
        except Exception:
            return audio

        if sample_rate <= 0:
            return audio

        target_samples = int(round((float(frame_count) / float(max(1e-6, fps))) * float(sample_rate)))
        target_samples = max(1, target_samples)
        current_samples = int(waveform.shape[-1])

        if current_samples == target_samples:
            return audio

        out = dict(audio)
        if current_samples > target_samples:
            out["waveform"] = waveform[..., :target_samples]
        else:
            pad = target_samples - current_samples
            out["waveform"] = torch.nn.functional.pad(waveform, (0, pad), value=0.0)
        return out

    def _inject_i2v_condition(
        self,
        vae,
        image,
        latent,
        strength,
        compression,
        use_preprocess=True,
    ):
        conditioned_image = image
        if use_preprocess:
            conditioned_image = self._result_tuple(
                LTXVPreprocess.execute(image, int(compression))
            )[0]
        return self._result_tuple(
            self._run_external_node(
                "LTXVImgToVideoConditionOnly",
                vae=vae,
                image=conditioned_image,
                latent=latent,
                strength=float(strength),
                bypass=False,
            )
        )[0]

    @staticmethod
    def _dilate_latent_sparse(latent, horizontal_scale, vertical_scale):
        horizontal_scale = int(max(1, horizontal_scale))
        vertical_scale = int(max(1, vertical_scale))
        if horizontal_scale == 1 and vertical_scale == 1:
            return latent

        samples = latent["samples"]
        mask = latent.get("noise_mask", None)
        dilated_shape = samples.shape[:3] + (
            int(samples.shape[3]) * vertical_scale,
            int(samples.shape[4]) * horizontal_scale,
        )
        dilated_samples = torch.zeros(
            dilated_shape,
            device=samples.device,
            dtype=samples.dtype,
            requires_grad=False,
        )
        dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

        dilated_mask_shape = (
            dilated_samples.shape[0],
            1,
            dilated_samples.shape[2],
            dilated_samples.shape[3],
            dilated_samples.shape[4],
        )
        dilated_mask = torch.full(
            dilated_mask_shape,
            -1.0,
            device=samples.device,
            dtype=samples.dtype,
            requires_grad=False,
        )
        dilated_mask[..., ::vertical_scale, ::horizontal_scale] = (
            mask if mask is not None else 1.0
        )
        return {"samples": dilated_samples, "noise_mask": dilated_mask}

    def _apply_v2v_guide(
        self,
        positive,
        negative,
        vae,
        latent,
        guide_image,
        guide_strength,
        latent_downscale_factor,
        model_union_control=None,
    ):
        _ = model_union_control
        effective_downscale = self._compatible_downscale_factor(
            latent,
            latent_downscale_factor,
        )
        return self._result_tuple(
            self._run_external_node(
                "LTXAddVideoICLoRAGuide",
                positive=positive,
                negative=negative,
                vae=vae,
                latent=latent,
                image=guide_image,
                frame_idx=0,
                strength=float(guide_strength),
                latent_downscale_factor=float(effective_downscale),
                crop="disabled",
                use_tiled_encode=False,
                tile_size=256,
                tile_overlap=64,
            )
        )

    def _run_i2v_or_t2v(
        self,
        mode,
        hq,
        model,
        vae,
        audio_vae,
        clip,
        prompt,
        seed,
        cfg,
        megapixels_target,
        aspect_ratio,
        frame_count,
        frame_rate,
        sampler_main,
        sigmas_main,
        sampler_refine,
        sigmas_refine,
        generated_audio_gain_db,
        spatial_upscale_model,
        image,
        source_audio,
        use_tiled_vae_decode,
        preview_enabled,
        firstframe_strength=1.0,
    ):
        total_steps = 5 if not hq else 7
        self._progress(1, total_steps, f"Preparing {mode} inputs")

        is_i2v = mode == "I2V"
        if is_i2v and image is None:
            raise ValueError(
                "I2V mode requires 'Image I2V / V2V FirstFrame' connected in the Config node."
            )

        if is_i2v:
            target_image = self._scale_total_pixels(image, megapixels_target)
            target_image = self._scale_to_multiple_cover(target_image, 32 if hq else 16)
            target_width = int(target_image.shape[2])
            target_height = int(target_image.shape[1])
        else:
            target_width, target_height = self._dims_from_megapixels_aspect(
                megapixels_target,
                aspect_ratio,
                divisible_by=32 if hq else 16,
            )
            target_image = None

        noise_obj = self._make_noise(seed)
        sampler_main_obj = self._make_sampler(sampler_main)
        sigmas_main_obj = self._make_sigmas(sigmas_main)
        sampler_refine_obj = self._make_sampler(sampler_refine)
        sigmas_refine_obj = self._make_sigmas(sigmas_refine)

        self._progress(2, total_steps, "Building conditioning")
        positive, negative = self._build_conditioning(clip, prompt, frame_rate)

        if not hq:
            self._progress(3, total_steps, "Sampling main pass")
            video_latent, audio_latent, av_latent = self._build_initial_av_latent(
                target_width,
                target_height,
                frame_count,
                frame_rate,
                audio_vae,
                source_audio=source_audio,
            )

            if is_i2v:
                video_latent = self._inject_i2v_condition(
                    vae,
                    target_image,
                    video_latent,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                )
                av_latent = self._result_tuple(
                    LTXVConcatAVLatent.execute(
                        video_latent,
                        audio_latent,
                    )
                )[0]

            sampled = self._sample_latent(
                model=model,
                positive=positive,
                negative=negative,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_main_obj,
                sigmas=sigmas_main_obj,
                latent=av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated = self._result_tuple(LTXVSeparateAVLatent.execute(sampled))
            final_video_latent = separated[0]
            final_audio_latent = separated[1]
            self._progress(4, total_steps, "Decoding video/audio")
        else:
            self._progress(3, total_steps, "Sampling stage 1")
            if spatial_upscale_model is None:
                raise ValueError(
                    "HQ is enabled but spatial_upscale_model is missing in models_pipe."
                )

            stage1_width = max(64, target_width // 2)
            stage1_height = max(64, target_height // 2)

            stage1_video_latent, stage1_audio_latent, _ = self._build_initial_av_latent(
                stage1_width,
                stage1_height,
                frame_count,
                frame_rate,
                audio_vae,
                source_audio=source_audio,
            )

            if is_i2v:
                stage1_image = self._resize_image(
                    target_image, stage1_width, stage1_height
                )
                stage1_video_latent = self._inject_i2v_condition(
                    vae,
                    stage1_image,
                    stage1_video_latent,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                )

            stage1_av_latent = self._result_tuple(
                LTXVConcatAVLatent.execute(stage1_video_latent, stage1_audio_latent)
            )[0]

            first_sample = self._sample_latent(
                model=model,
                positive=positive,
                negative=negative,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_main_obj,
                sigmas=sigmas_main_obj,
                latent=stage1_av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated_stage1 = self._result_tuple(
                LTXVSeparateAVLatent.execute(first_sample)
            )
            stage1_video_out = separated_stage1[0]
            stage1_audio_out = separated_stage1[1]

            upsampled_video = LTXVLatentUpsampler().upsample_latent(
                stage1_video_out,
                spatial_upscale_model,
                vae,
            )[0]

            if is_i2v:
                upsampled_video = self._inject_i2v_condition(
                    vae,
                    target_image,
                    upsampled_video,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                )

            refine_audio_latent = stage1_audio_out
            if source_audio is not None:
                refine_audio_latent = stage1_audio_latent

            refined_av_latent = self._result_tuple(
                LTXVConcatAVLatent.execute(upsampled_video, refine_audio_latent)
            )[0]

            self._progress(5, total_steps, "Sampling stage 2")
            final_sample = self._sample_latent(
                model=model,
                positive=positive,
                negative=negative,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_refine_obj,
                sigmas=sigmas_refine_obj,
                latent=refined_av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated_final = self._result_tuple(
                LTXVSeparateAVLatent.execute(final_sample)
            )
            final_video_latent = separated_final[0]
            final_audio_latent = separated_final[1]
            self._progress(6, total_steps, "Decoding video/audio")

        images = self._decode_video_latent(
            final_video_latent,
            vae,
            use_tiled_decode=bool(use_tiled_vae_decode),
        )
        audio = self._decode_audio_latent(final_audio_latent, audio_vae)
        audio = self._apply_audio_gain(audio, generated_audio_gain_db)
        self._progress(total_steps, total_steps, f"{mode} complete")
        return images, audio

    def _run_v2v(
        self,
        hq,
        model,
        model_union_control,
        vae,
        audio_vae,
        clip,
        prompt,
        seed,
        cfg,
        frame_count_limit,
        frame_rate,
        sampler_main,
        sigmas_main,
        sampler_refine,
        sigmas_refine,
        megapixels_target,
        depth_megapixels,
        v2v_guide_strength,
        latent_downscale_factor,
        generated_audio_gain_db,
        spatial_upscale_model,
        da3_model,
        video,
        firstframe_image=None,
        source_audio=None,
        use_tiled_vae_decode=False,
        preview_enabled=True,
        is_outpaint_mode=False,
        v2v_aspect_ratio="16:9 (Landscape)",
        firstframe_strength=1.0,
        depth_mouth_mask=False,
        mouth_detect_megapixels=0.25,
        mouth_single_item=True,
        mouth_detect_chunk_size=10,
        mouth_mask_expand=8,
        mouth_mask_blur=8.0,
    ):
        if hq:
            self._log(
                "V2V HQ two-stage path is disabled; using single-pass IC-LoRA workflow.",
                level="warn",
            )
            hq = False

        total_steps = 6 if not hq else 8
        self._progress(1, total_steps, "Preparing V2V inputs")

        if video is None:
            raise ValueError(
                "V2V mode requires 'Video (V2V image batch)' connected in the Config node."
            )

        # Outpaint mode doesn't use depth or first frame
        if is_outpaint_mode:
            self._log(
                "V2V Outpaint mode: skipping depth estimation and first frame processing",
                level="ok",
            )
        else:
            if da3_model is None:
                raise ValueError(
                    "V2V Depth Control mode requires da3_model in models_pipe."
                )

        input_video_frames = int(video.shape[0])
        frame_count = input_video_frames
        self._log(
            f"V2V frame count from input video: {frame_count}",
            level="ok",
        )

        video_frames = video[:frame_count]

        noise_obj = self._make_noise(seed)
        sampler_main_obj = self._make_sampler(sampler_main)
        sigmas_main_obj = self._make_sigmas(sigmas_main)
        sampler_refine_obj = self._make_sampler(sampler_refine)
        sigmas_refine_obj = self._make_sigmas(sigmas_refine)

        positive_base, negative_base = self._build_conditioning(
            clip, prompt, frame_rate
        )

        if is_outpaint_mode:
            # Outpaint mode: pad video to target aspect ratio
            return self._run_v2v_outpaint(
                hq=hq,
                model=model,
                vae=vae,
                audio_vae=audio_vae,
                positive_base=positive_base,
                negative_base=negative_base,
                video_frames=video_frames,
                frame_count=frame_count,
                frame_rate=frame_rate,
                noise_obj=noise_obj,
                sampler_main_obj=sampler_main_obj,
                sigmas_main_obj=sigmas_main_obj,
                sampler_refine_obj=sampler_refine_obj,
                sigmas_refine_obj=sigmas_refine_obj,
                megapixels_target=megapixels_target,
                v2v_guide_strength=v2v_guide_strength,
                latent_downscale_factor=latent_downscale_factor,
                generated_audio_gain_db=generated_audio_gain_db,
                spatial_upscale_model=spatial_upscale_model,
                source_audio=source_audio,
                use_tiled_vae_decode=use_tiled_vae_decode,
                preview_enabled=preview_enabled,
                v2v_aspect_ratio=v2v_aspect_ratio,
                total_steps=total_steps,
            )

        # Original Depth Control mode
        use_firstframe = firstframe_image is not None

        target_image = self._scale_total_pixels(video_frames, megapixels_target)
        depth_input = self._scale_total_pixels(video_frames, depth_megapixels)

        mouth_masks = None
        if depth_mouth_mask and not is_outpaint_mode:
            self._log("[Mouth Mask] Detecting mouth region via SAM3.1...")
            _t0 = time.monotonic()
            mouth_masks = self._detect_mouth_masks(
                video_frames,
                detect_megapixels=mouth_detect_megapixels,
                single_item=mouth_single_item,
                detect_chunk_size=mouth_detect_chunk_size,
            )
            if mouth_masks is not None:
                self._log(f"[Mouth Mask] Done ({time.monotonic() - _t0:.1f}s)", level="ok")
            else:
                self._log(f"[Mouth Mask] Failed ({time.monotonic() - _t0:.1f}s), skipping", level="warn")

        self._progress(2, total_steps, "Estimating depth (quiet)")
        depth_result = self._run_external_node(
            "DepthAnything_V3",
            suppress_output=True,
            da3_model=da3_model,
            images=depth_input,
            normalization_mode="V2-Style",
            resize_method="resize",
            invert_depth=False,
            keep_model_size=False,
        )
        self._offload_da3_model(da3_model)
        depth_image = self._result_tuple(depth_result)[0]

        # Match the reference V2V graph: resize DA3 output back to the target
        # guide resolution using a plain nearest/stretch resize before snapping
        # to the latent multiple.
        depth_resized = self._resize_image(
            depth_image,
            int(target_image.shape[2]),
            int(target_image.shape[1]),
            method="nearest-exact",
        )

        if mouth_masks is not None:
            import torch.nn.functional as _F

            self._log("Applying mouth mask to depth guide...")
            _mask_device = torch.device("cuda") if torch.cuda.is_available() else depth_resized.device

            def _apply_depth_mouth_mask():
                dh, dw = depth_resized.shape[1], depth_resized.shape[2]
                batch = int(mouth_masks.shape[0])
                chunk_size = 8 if _mask_device.type == "cuda" else 1
                pbar = comfy.utils.ProgressBar(max(1, batch))
                alpha_chunks = []
                for start in range(0, batch, chunk_size):
                    end = min(batch, start + chunk_size)
                    m = mouth_masks[start:end].unsqueeze(1).float().to(_mask_device, non_blocking=True)
                    if mouth_mask_expand > 0:
                        m = _F.max_pool2d(m, kernel_size=2 * mouth_mask_expand + 1, stride=1, padding=mouth_mask_expand)
                    if mouth_mask_blur > 0.0:
                        _radius = max(1, int(math.ceil(mouth_mask_blur)))
                        _sigma = max(0.1, mouth_mask_blur / 3.0)
                        _k = 2 * _radius + 1
                        _x = torch.arange(_k, dtype=torch.float32, device=m.device) - _radius
                        _gauss = torch.exp(-_x.pow(2) / (2.0 * _sigma * _sigma))
                        _gauss = _gauss / _gauss.sum()
                        m = _F.conv2d(m, _gauss.view(1, 1, _k, 1), padding=(_radius, 0))
                        m = _F.conv2d(m, _gauss.view(1, 1, 1, _k), padding=(0, _radius))
                    masks_up = _F.interpolate(m, size=(dh, dw), mode="bilinear", align_corners=False).squeeze(1).clamp(0.0, 1.0)
                    alpha_chunk = ((1.0 - masks_up) if mouth_mask_blur > 0.0 else (masks_up <= 0.5).float()).unsqueeze(-1)
                    alpha_chunks.append(alpha_chunk.to(depth_resized.device))
                    pbar.update_absolute(end, max(1, batch), None)
                alpha = torch.cat(alpha_chunks, dim=0)
                self._log(f"[Mouth Mask] depth_resized shape={depth_resized.shape} masked pixels={(alpha < 0.999).sum().item()}")
                return depth_resized * alpha

            depth_resized = self._timed_log("Mouth mask applied to depth guide", _apply_depth_mouth_mask, level="ok")

        multiple = max(1, int(round(float(latent_downscale_factor) * 32.0)))
        guide_full = self._scale_to_multiple_cover(depth_resized, multiple, "lanczos")
        guide_full = self._fit_image_batch_frames(
            guide_full,
            frame_count,
            label="V2V guide",
        )

        try:
            global _ltx23_depth_preview
            preview_np = guide_full[0].detach().float().cpu().clamp(0.0, 1.0).numpy()
            if preview_np.ndim == 2:
                preview_np = np.repeat(preview_np[..., None], 3, axis=-1)
            elif preview_np.shape[-1] == 1:
                preview_np = np.repeat(preview_np, 3, axis=-1)
            _ltx23_depth_preview = Image.fromarray((preview_np[..., :3] * 255.0).astype(np.uint8))
        except Exception as preview_e:
            self._log(f"Depth preview update failed: {preview_e}", level="warn")

        self._progress(3, total_steps, "Building IC-LoRA guide")

        if not hq:
            self._progress(4, total_steps, "Running V2V main sampler")
            latent_video = self._result_tuple(
                EmptyLTXVLatentVideo.execute(
                    int(guide_full.shape[2]),
                    int(guide_full.shape[1]),
                    frame_count,
                    1,
                )
            )[0]

            if use_firstframe:
                ff = self._prepare_v2v_firstframe(
                    firstframe_image,
                    int(guide_full.shape[2]),
                    int(guide_full.shape[1]),
                )
                latent_video = self._inject_i2v_condition(
                    vae,
                    ff,
                    latent_video,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                    use_preprocess=False,
                )

            positive_guided, negative_guided, latent_video_guided = (
                self._apply_v2v_guide(
                    positive_base,
                    negative_base,
                    vae,
                    latent_video,
                    guide_full,
                    v2v_guide_strength,
                    latent_downscale_factor,
                    model_union_control,
                )
            )

            audio_latent = self._build_audio_latent(
                frame_count,
                frame_rate,
                audio_vae,
                source_audio=source_audio,
            )

            av_latent = self._result_tuple(
                LTXVConcatAVLatent.execute(latent_video_guided, audio_latent)
            )[0]

            sampled = self._sample_latent(
                model=model,
                positive=positive_guided,
                negative=negative_guided,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_main_obj,
                sigmas=sigmas_main_obj,
                latent=av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated = self._result_tuple(LTXVSeparateAVLatent.execute(sampled))
            sampled_video_latent = separated[0]
            sampled_audio_latent = separated[1]

            cropped = self._result_tuple(
                LTXVCropGuides.execute(
                    positive_guided,
                    negative_guided,
                    sampled_video_latent,
                )
            )
            final_video_latent = cropped[2]
            final_audio_latent = sampled_audio_latent
            self._progress(5, total_steps, "Decoding video/audio")
        else:
            self._progress(4, total_steps, "Running V2V stage 1")
            if spatial_upscale_model is None:
                raise ValueError(
                    "HQ is enabled but spatial_upscale_model is missing in models_pipe."
                )

            stage1_width = max(64, int(guide_full.shape[2]) // 2)
            stage1_height = max(64, int(guide_full.shape[1]) // 2)

            stage1_guide = self._resize_image(
                guide_full,
                stage1_width,
                stage1_height,
                method="lanczos",
            )
            stage1_video_latent = self._result_tuple(
                EmptyLTXVLatentVideo.execute(
                    stage1_width,
                    stage1_height,
                    frame_count,
                    1,
                )
            )[0]

            # Depth-only V2V path: no direct RGB conditioning in stage 1.

            if use_firstframe:
                ff_s1 = self._prepare_v2v_firstframe(
                    firstframe_image, stage1_width, stage1_height
                )
                stage1_video_latent = self._inject_i2v_condition(
                    vae,
                    ff_s1,
                    stage1_video_latent,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                    use_preprocess=False,
                )

            (
                positive_guided_stage1,
                negative_guided_stage1,
                latent_video_guided_stage1,
            ) = self._apply_v2v_guide(
                positive_base,
                negative_base,
                vae,
                stage1_video_latent,
                stage1_guide,
                v2v_guide_strength,
                latent_downscale_factor,
                model_union_control,
            )

            stage1_audio_latent = self._build_audio_latent(
                frame_count,
                frame_rate,
                audio_vae,
                source_audio=source_audio,
            )

            stage1_av_latent = self._result_tuple(
                LTXVConcatAVLatent.execute(
                    latent_video_guided_stage1, stage1_audio_latent
                )
            )[0]

            stage1_sampled = self._sample_latent(
                model=model,
                positive=positive_guided_stage1,
                negative=negative_guided_stage1,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_main_obj,
                sigmas=sigmas_main_obj,
                latent=stage1_av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated_stage1 = self._result_tuple(
                LTXVSeparateAVLatent.execute(stage1_sampled)
            )
            stage1_video_out = separated_stage1[0]
            stage1_audio_out = separated_stage1[1]

            cropped_stage1 = self._result_tuple(
                LTXVCropGuides.execute(
                    positive_guided_stage1,
                    negative_guided_stage1,
                    stage1_video_out,
                )
            )
            stage1_video_out = cropped_stage1[2]

            upsampled_video = LTXVLatentUpsampler().upsample_latent(
                stage1_video_out,
                spatial_upscale_model,
                vae,
            )[0]

            self._progress(5, total_steps, "Running V2V stage 2")

            # Depth-only V2V path: no direct RGB conditioning in stage 2.

            if use_firstframe:
                ff_s2 = self._prepare_v2v_firstframe(
                    firstframe_image,
                    int(guide_full.shape[2]),
                    int(guide_full.shape[1]),
                )
                upsampled_video = self._inject_i2v_condition(
                    vae,
                    ff_s2,
                    upsampled_video,
                    firstframe_strength,
                    self.I2V_PREPROCESS_COMPRESSION,
                    use_preprocess=False,
                )

            (
                positive_guided_stage2,
                negative_guided_stage2,
                latent_video_guided_stage2,
            ) = self._apply_v2v_guide(
                positive_base,
                negative_base,
                vae,
                upsampled_video,
                guide_full,
                v2v_guide_strength,
                latent_downscale_factor,
                model_union_control,
            )

            refine_audio_latent = stage1_audio_out
            if source_audio is not None:
                refine_audio_latent = stage1_audio_latent

            refined_av_latent = self._result_tuple(
                LTXVConcatAVLatent.execute(
                    latent_video_guided_stage2,
                    refine_audio_latent,
                )
            )[0]

            self._progress(6, total_steps, "Refining final latent")

            final_sample = self._sample_latent(
                model=model,
                positive=positive_guided_stage2,
                negative=negative_guided_stage2,
                cfg=cfg,
                noise=noise_obj,
                sampler=sampler_refine_obj,
                sigmas=sigmas_refine_obj,
                latent=refined_av_latent,
                frame_rate=frame_rate,
                preview_enabled=preview_enabled,
            )

            separated_final = self._result_tuple(
                LTXVSeparateAVLatent.execute(final_sample)
            )
            final_video_latent = separated_final[0]
            final_audio_latent = separated_final[1]

            cropped_final = self._result_tuple(
                LTXVCropGuides.execute(
                    positive_guided_stage2,
                    negative_guided_stage2,
                    final_video_latent,
                )
            )
            final_video_latent = cropped_final[2]
            self._progress(7, total_steps, "Decoding video/audio")

        images = self._decode_video_latent(
            final_video_latent,
            vae,
            use_tiled_decode=bool(use_tiled_vae_decode),
        )

        generated_audio = self._decode_audio_latent(final_audio_latent, audio_vae)
        generated_audio = self._apply_audio_gain(
            generated_audio,
            generated_audio_gain_db,
        )

        self._progress(total_steps, total_steps, "V2V complete")

        return images, generated_audio

    def _run_v2v_outpaint(
        self,
        hq,
        model,
        vae,
        audio_vae,
        positive_base,
        negative_base,
        video_frames,
        frame_count,
        frame_rate,
        noise_obj,
        sampler_main_obj,
        sigmas_main_obj,
        sampler_refine_obj,
        sigmas_refine_obj,
        megapixels_target,
        v2v_guide_strength,
        latent_downscale_factor,
        generated_audio_gain_db,
        spatial_upscale_model,
        source_audio,
        use_tiled_vae_decode,
        preview_enabled,
        v2v_aspect_ratio,
        total_steps,
    ):
        """Outpaint V2V workflow: pad video to target aspect ratio instead of using depth."""
        self._progress(1, total_steps, "Preparing Outpaint inputs")

        # Calculate target dimensions based on aspect ratio
        target_width, target_height = self._dims_from_megapixels_aspect(
            megapixels_target, v2v_aspect_ratio, divisible_by=16
        )

        self._log(
            f"Outpaint target dimensions: {target_width}x{target_height} "
            f"(aspect: {v2v_aspect_ratio})",
            level="ok",
        )

        # Scale input video to megapixels target
        scaled_video = self._scale_total_pixels(video_frames, megapixels_target)

        # Pad video to target aspect ratio using ResizeAndPadImage logic
        _, src_h, src_w, _ = scaled_video.shape

        # Calculate new size maintaining aspect ratio
        scale_w = target_width / float(src_w)
        scale_h = target_height / float(src_h)
        scale = min(scale_w, scale_h)

        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        # Resize video frames
        resized_video = self._resize_image(
            scaled_video,
            new_w,
            new_h,
            method="lanczos",
        )

        # Pad to target dimensions (centered)
        pad_left = (target_width - new_w) // 2
        pad_top = (target_height - new_h) // 2
        pad_right = target_width - new_w - pad_left
        pad_bottom = target_height - new_h - pad_top

        padded_video = torch.nn.functional.pad(
            resized_video,
            (0, 0, pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        self._log(
            f"Padded video from {src_w}x{src_h} to {target_width}x{target_height}",
            level="ok",
        )

        self._progress(2, total_steps, "Building IC-LoRA guide for outpaint")

        # Create empty latent for target size
        latent_video = self._result_tuple(
            EmptyLTXVLatentVideo.execute(
                target_width,
                target_height,
                frame_count,
                1,
            )
        )[0]

        # Apply IC-LoRA guide with padded video as guide
        cfg = float(self.CFG_DEFAULT)

        positive_guided, negative_guided, latent_video_guided = self._apply_v2v_guide(
            positive_base,
            negative_base,
            vae,
            latent_video,
            padded_video,
            v2v_guide_strength,
            latent_downscale_factor,
            model,
        )

        self._progress(3, total_steps, "Building audio latent")

        audio_latent = self._build_audio_latent(
            frame_count,
            frame_rate,
            audio_vae,
            source_audio=source_audio,
        )

        av_latent = self._result_tuple(
            LTXVConcatAVLatent.execute(latent_video_guided, audio_latent)
        )[0]

        self._progress(4, total_steps, "Running Outpaint main sampler")

        sampled = self._sample_latent(
            model=model,
            positive=positive_guided,
            negative=negative_guided,
            cfg=cfg,
            noise=noise_obj,
            sampler=sampler_main_obj,
            sigmas=sigmas_main_obj,
            latent=av_latent,
            frame_rate=frame_rate,
            preview_enabled=preview_enabled,
        )

        separated = self._result_tuple(LTXVSeparateAVLatent.execute(sampled))
        sampled_video_latent = separated[0]
        sampled_audio_latent = separated[1]

        cropped = self._result_tuple(
            LTXVCropGuides.execute(
                positive_guided,
                negative_guided,
                sampled_video_latent,
            )
        )
        final_video_latent = cropped[2]
        final_audio_latent = sampled_audio_latent

        self._progress(5, total_steps, "Decoding video/audio")

        images = self._decode_video_latent(
            final_video_latent,
            vae,
            use_tiled_decode=bool(use_tiled_vae_decode),
        )

        generated_audio = self._decode_audio_latent(final_audio_latent, audio_vae)
        generated_audio = self._apply_audio_gain(
            generated_audio,
            generated_audio_gain_db,
        )

        self._progress(total_steps, total_steps, "Outpaint complete")

        return images, generated_audio

    def sample(
        self,
        models_pipe,
        config_pipe,
        workflow_mode,
        hq,
        live_preview,
        frame_count_from_audio,
        vae_decode_tiled,
        megapixels_target,
        aspect_ratio,
        frame_count,
        v2v_mode,
        v2v_guide_strength,
        depth_megapixels,
        v2v_aspect_ratio,
        sampler_main,
        sampler_refine,
        steps,
        generated_audio_gain_db,
        firstframe_strength,
        depth_mouth_mask,
        mouth_detect_megapixels,
        mouth_single_item,
        mouth_detect_chunk_size,
        mouth_mask_expand,
        mouth_mask_blur,
    ):
        mode = str(workflow_mode)
        if mode not in ("I2V", "T2V", "V2V"):
            raise ValueError(f"Unknown workflow_mode: {mode}")
        mode_internal = mode

        self._log(f"Starting mode: {mode}")
        live_preview = bool(live_preview)
        if live_preview:
            ensure_crt_previewer()
            latent_preview.set_preview_method("taesd")
        else:
            latent_preview.set_preview_method("none")

        # UI semantics: HQ ON means single-pass (no spatial upscale).
        # Internal pipeline uses hq=True for the two-stage upscale path,
        # so invert once at entry to keep downstream logic unchanged.
        hq = not bool(hq)
        megapixels_target = round(float(megapixels_target) * 10.0) / 10.0
        frame_count = int(frame_count)
        steps = int(steps)

        models = self._unpack_models_pipe(models_pipe)
        config = self._unpack_config_pipe(config_pipe)

        model = models["model"]
        vae = models["vae"]
        audio_vae = models["audio_vae"]
        clip = models["clip"]
        spatial_upscale_model = models["spatial_upscale_model"]
        da3_model = models["da3_model"]
        model_union_control = models["model_union_control"]
        model_outpaint = models["model_outpaint"]
        latent_downscale_factor = float(models["latent_downscale_factor"])

        # V2V mode selection
        v2v_mode_str = str(v2v_mode)
        is_outpaint_mode = mode == "V2V" and v2v_mode_str == "Outpaint"

        if mode == "V2V":
            if is_outpaint_mode:
                model_for_v2v = model_outpaint
                if model_for_v2v is None:
                    raise ValueError(
                        "V2V Outpaint mode requires 'model_outpaint' connected in LTX 2.3 US Models Pipe (CRT)."
                    )
            else:
                model_for_v2v = model_union_control
                if model_for_v2v is None:
                    raise ValueError(
                        "V2V Depth Control mode requires 'model_union_control' connected in LTX 2.3 US Models Pipe (CRT)."
                    )
            if model_union_control is model:
                raise ValueError(
                    "V2V is wired with base model as model_union_control. "
                    "Connect the MODEL output from LTXICLoRALoaderModelOnly "
                    "to 'model_union_control' in LTX 2.3 US Models Pipe (CRT)."
                )

        prompt = config["prompt"]
        seed = int(config["seed"])
        cfg = float(self.CFG_DEFAULT)
        image = config["image"]
        video = config["video"]
        source_audio = config["source_audio"]
        target_fps = max(1.0, float(config["framerate"]))
        target_output_frames = int(max(1, frame_count))

        audio_connected = source_audio is not None
        effective_source_audio = source_audio if audio_connected else None
        if audio_connected:
            self._log(
                "source_audio is connected -> input audio enabled",
                level="ok",
            )

        if (
            bool(frame_count_from_audio)
            and effective_source_audio is not None
            and mode_internal
            in (
                "I2V",
                "T2V",
            )
        ):
            audio_frames = self._frame_count_from_audio(
                effective_source_audio, target_fps
            )
            if audio_frames is not None:
                frame_count = int(audio_frames)
                self._log(
                    f"Frame count overridden from source_audio length -> {frame_count} frames @ {target_fps:.2f} fps",
                    level="ok",
                )

        if mode_internal == "V2V" and video is not None:
            try:
                target_output_frames = int(video.shape[0])
            except Exception:
                target_output_frames = int(max(1, frame_count))
        else:
            target_output_frames = int(max(1, frame_count))

        sigmas_main, sigmas_refine = self._build_parametric_sigma_texts(steps)

        if mode_internal in ("I2V", "T2V"):
            images, audio = self._run_i2v_or_t2v(
                mode=mode_internal,
                hq=hq,
                model=model,
                vae=vae,
                audio_vae=audio_vae,
                clip=clip,
                prompt=prompt,
                seed=seed,
                cfg=cfg,
                megapixels_target=megapixels_target,
                aspect_ratio=aspect_ratio,
                frame_count=frame_count,
                frame_rate=target_fps,
                sampler_main=sampler_main,
                sigmas_main=sigmas_main,
                sampler_refine=sampler_refine,
                sigmas_refine=sigmas_refine,
                generated_audio_gain_db=float(generated_audio_gain_db),
                spatial_upscale_model=spatial_upscale_model,
                image=image,
                source_audio=effective_source_audio,
                use_tiled_vae_decode=bool(vae_decode_tiled),
                preview_enabled=live_preview,
                firstframe_strength=float(firstframe_strength),
            )
        else:
            if is_outpaint_mode:
                self._log(
                    "V2V using model_outpaint (outpaint mode - pad video to target aspect ratio).",
                    level="ok",
                )
            else:
                self._log(
                    "V2V using model_union_control (depth-guided IC-LoRA path).",
                    level="ok",
                )
            images, audio = self._run_v2v(
                hq=hq,
                model=model_for_v2v,
                model_union_control=model_union_control,
                vae=vae,
                audio_vae=audio_vae,
                clip=clip,
                prompt=prompt,
                seed=seed,
                cfg=cfg,
                frame_count_limit=frame_count,
                frame_rate=target_fps,
                sampler_main=sampler_main,
                sigmas_main=sigmas_main,
                sampler_refine=sampler_refine,
                sigmas_refine=sigmas_refine,
                megapixels_target=megapixels_target,
                depth_megapixels=float(depth_megapixels),
                v2v_guide_strength=1.0 if is_outpaint_mode else float(v2v_guide_strength),
                latent_downscale_factor=latent_downscale_factor,
                generated_audio_gain_db=float(generated_audio_gain_db),
                spatial_upscale_model=spatial_upscale_model,
                da3_model=da3_model,
                video=video,
                firstframe_image=image,
                source_audio=effective_source_audio,
                use_tiled_vae_decode=bool(vae_decode_tiled),
                preview_enabled=live_preview,
                is_outpaint_mode=is_outpaint_mode,
                v2v_aspect_ratio=v2v_aspect_ratio,
                firstframe_strength=float(firstframe_strength),
                depth_mouth_mask=bool(depth_mouth_mask),
                mouth_detect_megapixels=float(mouth_detect_megapixels),
                mouth_single_item=bool(mouth_single_item),
                mouth_detect_chunk_size=int(mouth_detect_chunk_size),
                mouth_mask_expand=int(mouth_mask_expand),
                mouth_mask_blur=float(mouth_mask_blur),
            )

        if effective_source_audio is not None:
            audio = self._fit_audio_to_frame_count(
                effective_source_audio,
                target_output_frames,
                target_fps,
            )
            self._log("Using source audio from config input (fitted to output duration)", level="ok")
        else:
            self._log("Using generated audio (no source audio connected)", level="ok")

        return (images, audio)

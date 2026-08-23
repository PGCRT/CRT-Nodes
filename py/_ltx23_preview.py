"""LTX 2.3 live-preview override for the CRT unified sampler.

KJNodes-ModelPreviewOverride-style transport, mirroring the MiniMax H3
implementation: instead of the core binary preview path (which the modern
frontend renders as a still per step), an OUTER_SAMPLE wrapper decodes a
temporally-coherent span of the latent every sampler step via the taeltx
TAEHV, encodes ONE animated WebP whose loop duration equals the clip's real
duration, and pushes it as base64 JSON over a custom ws message. The frontend
DOM widget plays it in a looping <img>, which reads as a real video at true
pace.

The taeltx2_3 checkpoint is auto-downloaded into models/vae_approx. Without
it there is no preview for that run (the download is kicked off at first node
use, so the following run has it); a core Latent2RGB projection is the
last-resort fallback.
"""

import base64
import io as _pyio
import logging
import math
import os
import queue as _queue
import threading
import time

import torch
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import latent_preview

_TAEHV_FILENAME = "taeltx2_3.safetensors"
_TAEHV_URL = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors"
_TAEHV_DOWNLOAD_LOCK = threading.Lock()
_TAEHV_DOWNLOAD_THREAD = None
_TAEHV_LAST_FAIL = 0.0
_TAEHV_RETRY_COOLDOWN = 600.0

_POV_EVENT = "crt_ltx23_preview"
_POV_MAX_RES = 768
_POV_QUALITY = 88
_POV_LATENT_FRAMES = 16
_POV_FRAMES = 18
_DEFAULT_FPS = 24.0


def _interior_picks(total, n):
    """Evenly spaced indices avoiding the absolute first/last frames, where
    causal temporal decoders leave their worst boundary artifacts."""
    if n >= total:
        return list(range(total))
    if n == 1:
        return [total // 2]
    return sorted({min(total - 1, max(0, round((i + 0.5) * total / n))) for i in range(n)})


def _preview_model_present():
    files = folder_paths.get_filename_list("vae_approx")
    return any(fn.lower().startswith(("taeltx2_3", "taeltx_2_3")) for fn in files)


def _find_preview_models():
    """Ordered candidate paths: standard arch before wide fallbacks."""
    files = folder_paths.get_filename_list("vae_approx")
    lower_files = [(fn, fn.lower()) for fn in files]
    prefixes = [
        "taeltx2_3",
        "taeltx_2_3",
        "taeltx2_3_wide",
        "taeltx_2_3_wide",
        "taeltx2",
        "taeltx_2",
    ]
    seen = set()
    result = []
    for prefix in prefixes:
        for fn, lower in lower_files:
            if lower.startswith(prefix):
                full = folder_paths.get_full_path("vae_approx", fn)
                if full and full not in seen:
                    seen.add(full)
                    result.append(full)
                break
    return result


def _download_preview_model():
    """Blocking worker: fetch the taeltx2_3 TAEHV preview decoder if missing."""
    global _TAEHV_LAST_FAIL
    target = folder_paths.get_full_path("vae_approx", _TAEHV_FILENAME)
    if target is not None:
        return target
    try:
        from .download_progress import download_url_with_progress

        target_dir = folder_paths.get_folder_paths("vae_approx")[0]
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, _TAEHV_FILENAME)
        print(f"[CRT LTX23] Downloading live-preview model {_TAEHV_FILENAME} ...")
        download_url_with_progress(
            _TAEHV_URL,
            target,
            label=_TAEHV_FILENAME,
            user_agent="CRT-Nodes",
            console_prefix="CRT LTX23",
        )
        print(f"[CRT LTX23] Live-preview model ready: {target}")
        return target
    except Exception as e:
        _TAEHV_LAST_FAIL = time.time()
        print(f"[CRT LTX23] Could not download {_TAEHV_FILENAME} ({e}); live preview unavailable until it succeeds.")
        return None


def kickoff_ltx23_preview_model_download():
    """Start the taeltx download once, in the background, at first node use."""
    global _TAEHV_DOWNLOAD_THREAD
    if _preview_model_present():
        return
    if _TAEHV_DOWNLOAD_THREAD is not None and _TAEHV_DOWNLOAD_THREAD.is_alive():
        return
    if time.time() - _TAEHV_LAST_FAIL < _TAEHV_RETRY_COOLDOWN:
        return
    with _TAEHV_DOWNLOAD_LOCK:
        if _preview_model_present():
            return
        if _TAEHV_DOWNLOAD_THREAD is not None and _TAEHV_DOWNLOAD_THREAD.is_alive():
            return
        if time.time() - _TAEHV_LAST_FAIL < _TAEHV_RETRY_COOLDOWN:
            return
        thread = threading.Thread(
            target=_download_preview_model,
            name="crt-ltx23-preview-download",
            daemon=True,
        )
        _TAEHV_DOWNLOAD_THREAD = thread
        thread.start()


class _TaeltxDecoder:
    """Batched temporal TAEHV decoder for LTX 2.3 latents (128 channels)."""

    def __init__(self, model_path):
        from comfy.taesd.taehv import TAEHV

        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        model = TAEHV(latent_channels=128)
        model.load_state_dict(sd, strict=True)
        del sd
        model.eval()
        self.device = comfy.model_management.vae_device()
        self.dtype = comfy.model_management.vae_dtype(self.device, [torch.float16, torch.bfloat16])
        model = model.to(device=self.device, dtype=self.dtype)
        if torch.device(self.device).type == "cuda":
            model.to(memory_format=torch.channels_last)
        self.model = model

    def decode_frames(self, video_bcthw):
        """[B, C, T, H, W] -> list of PIL frames, one batched temporal decode."""
        out = self.model.decode(video_bcthw.to(device=self.device, dtype=self.dtype))
        rgb = out[0].permute(1, 2, 3, 0).clamp(0, 1).float().cpu()
        return [latent_frame_to_image(frame) for frame in rgb]


def latent_frame_to_image(hwc):
    import latent_preview

    return latent_preview.preview_to_image(hwc, do_scale=False)


_DECODER_CACHE = {}
_DECODER_FAILED = set()
_WARNED_NO_DECODER = False


def _get_decoder():
    global _WARNED_NO_DECODER
    cached = _DECODER_CACHE.get("decoder")
    if cached is not None:
        return cached
    decoder = None
    for model_path in _find_preview_models():
        if model_path in _DECODER_FAILED:
            continue
        try:
            decoder = _TaeltxDecoder(model_path)
        except Exception as e:
            print(f"[CRT-preview] skipping {model_path}: {e}")
            _DECODER_FAILED.add(model_path)
            continue
        break
    if decoder is None:
        if not _WARNED_NO_DECODER:
            _WARNED_NO_DECODER = True
            print(
                "[CRT-preview] taeltx2_3 preview model unavailable; "
                "no LTX live preview this run (download runs in background; next run will have it)."
            )
        return None
    _DECODER_CACHE["decoder"] = decoder
    return decoder


def _num_keyframes(guider):
    try:
        positive = guider.conds.get("positive") if hasattr(guider, "conds") else None
        if positive and len(positive) > 0:
            kf = positive[0].get("keyframe_idxs")
            if kf is not None:
                return int(torch.unique(kf[0, 0, :, 0]).numel())
    except Exception:
        pass
    return 0


def _resolve_video_latent(x0, latent_shapes, num_keyframes):
    """Any LTX latent layout -> [B, C, T, H, W] video tensor (keyframes trimmed)."""
    if getattr(x0, "is_nested", False):
        try:
            x0 = x0.tensors[0]
        except Exception:
            return None

    fixed = None
    if torch.is_tensor(x0):
        if x0.ndim == 5 and x0.shape[0] > 0:
            fixed = x0
        elif x0.ndim == 4 and x0.shape[0] > 0:
            fixed = x0.unsqueeze(2)

    if fixed is None and latent_shapes and torch.is_tensor(x0):
        try:
            if int(sum(math.prod(s) for s in latent_shapes)) == int(x0.numel()):
                batch = x0.shape[0]
                target = latent_shapes[0]
                cut = math.prod(target[1:])
                if x0.ndim >= 2 and x0.numel() >= cut * batch:
                    remaining = x0.reshape(batch, -1)
                    fixed = remaining[:, :cut].reshape([batch] + list(target)[1:])
        except Exception:
            return None

    if fixed is None or fixed.ndim != 5 or fixed.shape[2] <= 0:
        return None

    if num_keyframes > 0 and fixed.shape[2] > num_keyframes:
        fixed = fixed[:, :, :-num_keyframes]
    if fixed.shape[2] <= 0:
        return None
    return fixed


def _encode_animated_webp(frames, duration_ms, quality=_POV_QUALITY, max_res=_POV_MAX_RES):
    if not frames:
        return None
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    duration_ms = int(max(20, min(500, duration_ms)))
    buf = _pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    except Exception as e:
        logging.warning(f"[CRT-preview] animated WebP encode failed: {e}")
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _AsyncPreviewEncoder:
    """Off-thread encoder. Bounded FIFO drops-on-full so sampling never blocks."""

    _STOP = object()

    def __init__(self, max_in_flight=2):
        self.q = _queue.Queue(maxsize=max_in_flight)
        self.thread = threading.Thread(target=self._run, name="crt_ltx23_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
            return True
        except _queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logging.exception("[CRT-preview] async encoder error")

    def shutdown(self, drain_timeout=5.0):
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except _queue.Full:
            pass
        self.thread.join(timeout=drain_timeout)


def _suppressed_preview_image(self_, preview_format, x0):
    return None


def _get_prompt_server():
    try:
        from server import PromptServer
        return PromptServer.instance, PromptServer.instance.client_id
    except Exception:
        return None, None


class _LTXPreviewOverrideWrapper:
    def __init__(self, node_id, fps=None):
        self.node_id = str(node_id) if node_id is not None else None
        self.fps = float(fps) if fps and float(fps) > 1 else _DEFAULT_FPS

    def _decode_video_frames(self, x0, latent_shapes, num_keyframes):
        """Returns (display_frames, total_pixel_frames) for true-pace timing."""
        fixed = _resolve_video_latent(x0, latent_shapes, num_keyframes)
        if fixed is None:
            return [], 0
        total = int(fixed.shape[2])
        count = min(_POV_LATENT_FRAMES, total)
        if count < total:
            indices = _interior_picks(total, count)
            selected = fixed[:, :, indices].contiguous()
        else:
            selected = fixed
        decoder = _get_decoder()
        if decoder is None:
            return [], 0
        try:
            pixel_frames = decoder.decode_frames(selected)
        except Exception as e:
            logging.warning(f"[CRT-preview] override decode failed: {e}")
            return [], 0
        if not pixel_frames:
            return [], 0
        n = min(_POV_FRAMES, len(pixel_frames))
        if n < len(pixel_frames):
            picks = _interior_picks(len(pixel_frames), n)
            display = [pixel_frames[i] for i in picks]
        else:
            display = pixel_frames
        return display, len(pixel_frames)

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        guider = executor.class_obj
        num_keyframes = _num_keyframes(guider)
        del guider

        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps = max(0, len(sigmas_list) - 1)

        server, client_id = _get_prompt_server()
        if server is not None and self.node_id is not None:
            try:
                server.send_sync(
                    _POV_EVENT,
                    {"node_id": self.node_id, "step": 0, "total": total_steps, "sigma": sigmas_list[0] if sigmas_list else None, "sigmas": sigmas_list},
                    client_id,
                )
            except Exception:
                pass

        encoder = _AsyncPreviewEncoder()

        def new_callback(step, x0, x, total_steps_):
            try:
                shapes = latent_shapes if latent_shapes else None
                frames, px_total = self._decode_video_frames(x0, shapes, num_keyframes)
                if not frames:
                    raise ValueError("no frames decoded")
                first = frames[0]
                w_, h_ = first.width, first.height
                b64 = None
                mime = "image/jpeg"
                duration_ms = 83
                if len(frames) > 1:
                    # TRUE PACE: one full loop lasts exactly the clip's real
                    # duration (decoded pixel length / native fps).
                    real_seconds = px_total / self.fps
                    duration_ms = int(round(real_seconds * 1000.0 / len(frames)))
                    b64 = _encode_animated_webp(frames, duration_ms)
                    mime = "image/webp"
                if not b64:
                    if first.width > _POV_MAX_RES or first.height > _POV_MAX_RES:
                        first = ImageOps.contain(first, (_POV_MAX_RES, _POV_MAX_RES), Image.LANCZOS)
                    buf = _pyio.BytesIO()
                    first.save(buf, format="JPEG", quality=_POV_QUALITY)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    mime = "image/jpeg"

                display_fps = round(1000.0 / max(1, duration_ms), 2)

                def send(b64=b64, mime=mime, w_=w_, h_=h_, step=step + 1, total=total_steps_, fps=display_fps):
                    if server is None or self.node_id is None:
                        return
                    server.send_sync(
                        _POV_EVENT,
                        {
                            "node_id": self.node_id,
                            "image": b64,
                            "mime": mime,
                            "w": int(w_),
                            "h": int(h_),
                            "step": int(step),
                            "total": int(total),
                            "fps": float(fps) if mime == "image/webp" else None,
                            "sigmas": None,
                        },
                        client_id,
                    )

                encoder.submit(send)
            except Exception as e:
                logging.warning(f"[CRT-preview] override preview failed: {e}")
            if callback is not None:
                callback(step, x0, x, total_steps_)

        # Suppress core binary previews for the duration of this run (KJ-style
        # scoped class patch): they render as a still per step in the modern
        # frontend and would fight the animated override. Restored afterwards,
        # so the user's global preview method is untouched outside the run.
        prev_methods = []
        targets = [latent_preview.LatentPreviewer]
        stack = list(latent_preview.LatentPreviewer.__subclasses__())
        while stack:
            cls = stack.pop()
            targets.append(cls)
            stack.extend(cls.__subclasses__())
        for cls in targets:
            if "decode_latent_to_preview_image" in cls.__dict__:
                prev_methods.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
                cls.decode_latent_to_preview_image = _suppressed_preview_image

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            for cls, prev in prev_methods:
                cls.decode_latent_to_preview_image = prev


def apply_ltx23_preview_override(model, node_id, fps=None):
    """Attach the animated-WebP live preview wrapper to the sampling model."""
    import comfy.patcher_extension

    m = model.clone()
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "crt_ltx23_preview",
        _LTXPreviewOverrideWrapper(node_id, fps),
    )
    return m

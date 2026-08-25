"""MiniMax H3 live-preview decoder for the CRT unified sampler.

True video-stream preview like the LTX Unified Sampler: every sampler tick
decodes the next latent chunk into a staging buffer, a completed sweep is
promoted atomically, and a sender thread loops the published sweep at true
temporal pace. The first frame of each tick is also returned as the standard
per-step anchor image so the preview starts immediately (pre-sweep).

Decoder is the temporal TAEHV taeh3 checkpoint (24 latent channels,
patch_size 2) auto-downloaded into models/vae_approx. Without it the
previewer falls back to the core MiniMaxH3 latent-to-RGB factor projection.
"""

import logging
import math
import os
import threading
import time
import weakref

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import comfy.latent_formats
import comfy.model_management
import comfy.utils
import folder_paths
import latent_preview

# --- Spectrum offline-replay preview fix ---------------------------------
# Spectrum's offline_smoothing_replay runs two passes: capture (no preview)
# then replay (preview on second half). That makes the CRT video preview
# start at 20/20. Patch it so both passes emit previews — the video sweep
# then starts at step 1 and loops at natural pace (8fps sample, native fps
# for timing), matching LTX behaviour.
try:
    # Prefer the embedded Spectrum engine; fall back to an external install.
    try:
        from .usopt_spectrum_h3 import sampling as _spectrum_sampling
    except Exception:
        import comfyui_spectrum_h3.sampling as _spectrum_sampling

    _orig_offline_progress_callbacks = _spectrum_sampling._offline_progress_callbacks

    def _patched_offline_progress_callbacks(callback, total_steps: int):
        if callback is None or total_steps <= 0:
            return None, callback, None
        import comfy.utils as _cu
        total_work = total_steps * 2
        progress = _cu.ProgressBar(total_work)
        replay_finished = False

        def capture_callback(step, x0, x, _pass_steps):
            # Was progress-only; now also forwards preview so the CRT sweep
            # starts immediately (first pass = steps 0..total-1 / total_work).
            progress.update_absolute(step + 1, total_work)
            try:
                callback(step, x0, x, total_work)
            except Exception:
                pass

        def replay_callback(step, x0, x, _pass_steps):
            nonlocal replay_finished
            callback(total_steps + step, x0, x, total_work)
            replay_finished = step + 1 >= total_steps

        def complete_progress():
            if not replay_finished:
                progress.update_absolute(total_work, total_work)

        return capture_callback, replay_callback, complete_progress

    _spectrum_sampling._offline_progress_callbacks = _patched_offline_progress_callbacks
except Exception:
    pass

from .download_progress import download_url_with_progress

_TAEH3_FILENAME = "taeh3.safetensors"
_TAEH3_URL = "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors"
_TAEH3_DOWNLOAD_LOCK = threading.Lock()
_TAEH3_DOWNLOAD_THREAD = None
_TAEH3_LAST_FAIL = 0.0
_TAEH3_RETRY_COOLDOWN = 600.0

_PREVIEW_FPS = 8.0
_DECODE_LATENT_PER_TICK = 8
_MAX_PREVIEW_EDGE = 512
_DEFAULT_NATIVE_FPS = 24.0
_SENDER_IDLE_SECONDS = 12.0

_H3_LATENT_FORMATS = (comfy.latent_formats.MiniMaxH3AV, comfy.latent_formats.MiniMaxH3Video)

_LIVE_PREVIEWERS = weakref.WeakSet()


def reset_all_timelines():
    for p in list(_LIVE_PREVIEWERS):
        try:
            p.reset_timeline()
        except Exception:
            pass


def wipe_all_caches():
    """Hard wipe: drop every buffered frame and force fresh previewer instances
    on the next run. This is intentionally aggressive — the user explicitly
    requested that no decoded frame from a previous run ever appears in the
    next run."""
    for p in list(_LIVE_PREVIEWERS):
        try:
            p.reset_timeline()
        except Exception:
            pass
    # Drop cached previewer objects themselves so the next
    # get_previewer() builds a brand-new instance with zero history.
    _PREVIEWER_CACHE.clear()
    _LAST_CORE_PREVIEW_META.clear()


def is_h3_latent_format(latent_format):
    return isinstance(latent_format, _H3_LATENT_FORMATS)


def taeh3_model_present():
    files = folder_paths.get_filename_list("vae_approx")
    return any(fn.lower().startswith("taeh3") for fn in files)


def _download_taeh3():
    global _TAEH3_LAST_FAIL
    target = folder_paths.get_full_path("vae_approx", _TAEH3_FILENAME)
    if target is not None:
        return target
    try:
        target_dir = folder_paths.get_folder_paths("vae_approx")[0]
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, _TAEH3_FILENAME)
        print(f"[CRT MiniMaxH3] Downloading live-preview model {_TAEH3_FILENAME} ...")
        download_url_with_progress(_TAEH3_URL, target, label=_TAEH3_FILENAME, user_agent="CRT-Nodes", console_prefix="CRT MiniMaxH3")
        print(f"[CRT MiniMaxH3] Live-preview model ready: {target}")
        return target
    except Exception as e:
        _TAEH3_LAST_FAIL = time.time()
        print(f"[CRT MiniMaxH3] Could not download {_TAEH3_FILENAME} ({e}); live preview falls back to RGB factors.")
        return None


def kickoff_taeh3_download():
    global _TAEH3_DOWNLOAD_THREAD
    if taeh3_model_present():
        return
    if _TAEH3_DOWNLOAD_THREAD is not None and _TAEH3_DOWNLOAD_THREAD.is_alive():
        return
    if time.time() - _TAEH3_LAST_FAIL < _TAEH3_RETRY_COOLDOWN:
        return
    with _TAEH3_DOWNLOAD_LOCK:
        if taeh3_model_present():
            return
        if _TAEH3_DOWNLOAD_THREAD is not None and _TAEH3_DOWNLOAD_THREAD.is_alive():
            return
        if time.time() - _TAEH3_LAST_FAIL < _TAEH3_RETRY_COOLDOWN:
            return
        thread = threading.Thread(target=_download_taeh3, name="crt-minimaxh3-preview-download", daemon=True)
        _TAEH3_DOWNLOAD_THREAD = thread
        thread.start()


def _find_taeh3_path():
    files = folder_paths.get_filename_list("vae_approx")
    for fn in files:
        if fn.lower().startswith("taeh3"):
            return folder_paths.get_full_path("vae_approx", fn)
    return None


def _is_taehv_state_dict(sd):
    return "decoder.1.weight" in sd and "decoder.22.bias" in sd


class _TAEH3Decoder:
    """Decode-only temporal TAEHV sized for H3 (24ch, patch 2)."""

    def __init__(self, sd):
        from comfy.taesd.taehv import TAEHV, conv
        latent_channels = sd["decoder.1.weight"].shape[1]
        patch_size = max(1, int(round((sd["decoder.22.bias"].shape[0] / 3) ** 0.5)))
        model = TAEHV(latent_channels=latent_channels)
        if model.patch_size != patch_size:
            model.patch_size = patch_size
            model.encoder[0] = conv(3 * patch_size ** 2, model.encoder[0].out_channels)
            model.decoder[-1] = conv(model.decoder[-1].in_channels, 3 * patch_size ** 2)
        model.load_state_dict(sd)
        del model.encoder
        self.device = comfy.model_management.vae_device()
        self.dtype = comfy.model_management.vae_dtype(self.device, [torch.float16, torch.bfloat16])
        model = model.eval().to(device=self.device, dtype=self.dtype)
        if torch.device(self.device).type == "cuda":
            model.to(memory_format=torch.channels_last)
        self.model = model

    def decode_frames(self, latent_bcthw, frame_indices=None):
        """[B, C, T, H, W] -> list of PIL frames."""
        if frame_indices is not None and len(frame_indices) == 1:
            out = self.model.decode(latent_bcthw[:, :, frame_indices[0]:frame_indices[0]+1].to(device=self.device, dtype=self.dtype))
            # Single latent time: take the last temporal output frame.
            rgb = out[0, :, -1].movedim(0, -1).clamp(0, 1).float()
            return [latent_preview.preview_to_image(rgb, do_scale=False)]
        # Batched temporal decode: TAEHV needs neighbouring latent frames for
        # correct motion/quality, so the selected span is decoded in ONE call.
        out = self.model.decode(latent_bcthw.to(device=self.device, dtype=self.dtype))
        rgb = out[0].permute(1, 2, 3, 0).clamp(0, 1).float().cpu()
        return [latent_preview.preview_to_image(frame, do_scale=False) for frame in rgb]


_SNIFF_LOCK = threading.Lock()
_LAST_CORE_PREVIEW_META = {}
_SNIFFED_SERVERS = weakref.WeakSet()


def _install_send_sniffer(server):
    """Record the metadata core attaches to its own tagged preview sends.

    The modern frontend renders previews exclusively from
    PREVIEW_IMAGE_WITH_METADATA and keys them by display_node_id/prompt_id,
    so replaying core's exact attribution is the only reliable way for the
    sender thread to stay visible. Scraping server attributes (last_node_id,
    client_id) is not enough — there is no last_prompt_id attribute.
    """
    if server in _SNIFFED_SERVERS:
        return
    _SNIFFED_SERVERS.add(server)
    original = server.send_sync

    def send_sync(event, data, sid=None):
        try:
            if event == 4 and isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
                with _SNIFF_LOCK:
                    snapshot = dict(data[1])
                    snapshot["_sid"] = sid
                    _LAST_CORE_PREVIEW_META.clear()
                    _LAST_CORE_PREVIEW_META.update(snapshot)
        except Exception:
            pass
        return original(event, data, sid)

    server.send_sync = send_sync


def _latest_core_preview_meta():
    with _SNIFF_LOCK:
        return dict(_LAST_CORE_PREVIEW_META)


def _get_server():
    try:
        from protocol import BinaryEventTypes
        from server import PromptServer
        _install_send_sniffer(PromptServer.instance)
        return PromptServer.instance, BinaryEventTypes.UNENCODED_PREVIEW_IMAGE
    except Exception:
        return None, None


class CRTMiniMaxH3WrappedPreviewer(latent_preview.LatentPreviewer):
    """LTX-style sweep preview for H3: per-tick chunk + sender video loop + anchor."""

    _blank = Image.new("RGB", (256, 256), (8, 8, 8))

    def __init__(self, model_path=None, preview_state=None):
        self.taesd = None
        self._model_path = model_path
        self._model = None
        self._model_failed = False
        self._preview_state = preview_state
        fmt = comfy.latent_formats.MiniMaxH3Video()
        self._factors = torch.tensor(fmt.latent_rgb_factors).transpose(0, 1)
        self._bias = torch.tensor(fmt.latent_rgb_factors_bias)
        self._cursor = 0
        self._frames_key = None
        self._stream_logged = False
        self._lock = threading.Lock()
        self._display_px = []
        self._staging_px = []
        self._staging_filled = 0
        self._staging_total = 0
        self._publish_ready = False
        self._meta = None
        self._epoch = 0
        self._run_token = None
        self._last_enqueue = 0.0
        self._sender_thread = None
        _LIVE_PREVIEWERS.add(self)

    def reset_timeline(self):
        with self._lock:
            self._epoch += 1
            self._cursor = 0
            self._frames_key = None
            self._display_px = []
            self._staging_px = []
            self._staging_filled = 0
            self._staging_total = 0
            self._publish_ready = False
            self._meta = None
            self._stream_logged = False
            self._last_enqueue = 0.0

    def _get_model(self, device=None):
        if self._model is not None or self._model_path is None or self._model_failed:
            return self._model
        try:
            sd = comfy.utils.load_torch_file(self._model_path, safe_load=True)
            if not _is_taehv_state_dict(sd):
                raise ValueError("not a temporal TAEHV state dict")
            self._model = _TAEH3Decoder(sd)
        except Exception as error:
            self._model_failed = True
            logging.warning("[CRT MiniMaxH3] taeh3 preview model unusable (%s); using RGB factors", error)
        return self._model

    @staticmethod
    def _first_frame_shape(x0, last_shapes):
        if getattr(x0, "is_nested", False):
            try:
                x0 = x0.tensors[0]
            except Exception:
                return None
        if torch.is_tensor(x0) and x0.ndim == 5 and x0.shape[0] > 0 and x0.shape[2] > 0:
            return x0
        # Fallback: try unpacking flattened latents like LTX does
        if last_shapes and hasattr(comfy.utils, "unpack_latents"):
            try:
                if int(sum(math.prod(s) for s in last_shapes)) == int(x0.numel()):
                    unpacked = None
                    try:
                        unpacked = comfy.utils.unpack_latents(x0, last_shapes)
                    except Exception:
                        unpacked = None
                    # Core unpack only reshapes when multiple shapes are packed;
                    # single-entry packs come back unreshaped and odd layouts
                    # can raise outright — redo the first entry explicitly.
                    if not (unpacked and unpacked[0].ndim == 5):
                        batch = x0.shape[0]
                        remaining = x0.reshape(batch, -1)
                        target = last_shapes[0]
                        cut = math.prod(target[1:])
                        unpacked = [remaining[:, :cut].reshape([batch] + list(target)[1:])]
                    if unpacked and unpacked[0].ndim == 5:
                        return unpacked[0]
            except Exception:
                pass
        return None

    def _decode_frames(self, slice_bcthw):
        """Decode [B, C, Tchunk, H, W] chunk to list[PIL]."""
        device = slice_bcthw.device
        model = self._get_model(device)
        if model is not None:
            try:
                with torch.no_grad():
                    indices = list(range(slice_bcthw.shape[2]))
                    return model.decode_frames(slice_bcthw, frame_indices=indices)
            except Exception as error:
                logging.warning("[CRT MiniMaxH3] taeh3 decode failed (%s); using RGB factors", error)
                self._model_failed = True
        # RGB factors fallback per latent time
        factors = self._factors.to(dtype=slice_bcthw.dtype, device=device)
        bias = self._bias.to(dtype=slice_bcthw.dtype, device=device)
        images = []
        for t in range(slice_bcthw.shape[2]):
            chw = slice_bcthw[0, :, t]
            rgb = torch.nn.functional.linear(chw.movedim(0, -1), factors, bias=bias)
            images.append(latent_preview.preview_to_image(rgb))
        return images

    def decode_latent_to_preview_image(self, preview_format, x0):
        server, preview_event = _get_server()
        last_shapes = getattr(self._preview_state, "last_latent_shapes", None) if self._preview_state else None
        # Non-server path (e.g. headless test) — single middle frame anchor
        if server is None:
            fixed = self._first_frame_shape(x0, last_shapes)
            if fixed is None:
                return ("JPEG", self._blank, _MAX_PREVIEW_EDGE)
            mid = fixed.shape[2] // 2
            return ("JPEG", self._decode_frames(fixed[:, :, mid:mid+1])[0], _MAX_PREVIEW_EDGE)

        fixed = self._first_frame_shape(x0, last_shapes)
        if fixed is None:
            return ("JPEG", self._blank, _MAX_PREVIEW_EDGE)

        with self._lock:
            run_token = getattr(self._preview_state, "run_token", 0)
            stale = (
                run_token != self._run_token
                or self._meta is None
                or self._meta.get("node_id") != getattr(server, "last_node_id", None)
            )
            if stale:
                self._run_token = run_token
                self._meta = {"sid": getattr(server, "client_id", None), "node_id": getattr(server, "last_node_id", None), "prompt_id": getattr(server, "last_prompt_id", None)}
                self._epoch += 1
                self._cursor = 0
                self._frames_key = None
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False
                self._display_px = []

        key = tuple(fixed.shape)
        if key != self._frames_key:
            self._frames_key = key
            self._cursor = 0
            with self._lock:
                self._epoch += 1
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False
                self._display_px = []

        total = int(fixed.shape[2])
        start = self._cursor % total
        if start == 0:
            with self._lock:
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False
        count = min(_DECODE_LATENT_PER_TICK, total - start)
        end = start + count
        chunk = fixed[:, :, start:end]
        self._cursor = end % total
        images = self._decode_frames(chunk)
        with self._lock:
            base_px = start * max(1, len(images) // max(1, count))
            needed = base_px + len(images)
            if len(self._staging_px) < needed:
                self._staging_px.extend([None] * (needed - len(self._staging_px)))
            for i, img in enumerate(images):
                self._staging_px[base_px + i] = img
            self._staging_filled += count
            self._staging_total = total
            # Publish a looping video from the very first chunk so the UI shows
            # a video immediately, not a still per step. Full-sweep videos
            # replace the partial ones once the sweep completes.
            if self._staging_filled >= total:
                self._publish_ready = True
            elif self._staging_filled > 0:
                self._publish_ready = True
        anchor = images[0]
        self._last_enqueue = time.monotonic()
        self._ensure_sender()
        if not self._stream_logged:
            self._stream_logged = True
            dec = "taeh3" if self._model is not None and not self._model_failed else "RGB factors"
            print(f"[CRT-preview] H3 streaming @ {_PREVIEW_FPS:.0f}fps display ({dec} decoder)")

        # Anchor through standard path — shows immediately, even pre-sweep.
        return (preview_format or "JPEG", anchor, _MAX_PREVIEW_EDGE)

    def _sender_loop(self):
        display_px = []
        samples = []
        epoch = -1
        while True:
            with self._lock:
                meta = self._meta
                if epoch != self._epoch:
                    epoch = self._epoch
                    display_px = []
                    samples = []
                if self._publish_ready and self._staging_filled > 0:
                    is_full = self._staging_total > 0 and self._staging_filled >= self._staging_total
                    if is_full:
                        display_px = self._staging_px
                        self._display_px = display_px
                        self._staging_px = []
                        self._staging_filled = 0
                        self._staging_total = 0
                        self._publish_ready = False
                    else:
                        # Partial sweep: show whatever we have so far as a
                        # looping video immediately, keep staging to grow to
                        # the full video.
                        valid = [f for f in self._staging_px if f is not None]
                        if valid:
                            display_px = valid
                            self._display_px = display_px
                        self._publish_ready = False
                        if not valid:
                            continue
                    fps = getattr(self._preview_state, "fps_override", None) if self._preview_state else None
                    try:
                        fps = float(fps)
                    except Exception:
                        fps = _DEFAULT_NATIVE_FPS
                    if not (fps > 1):
                        fps = _DEFAULT_NATIVE_FPS
                    stride = max(1, round(fps / _PREVIEW_FPS))
                    samples = [f for f in (display_px[i] for i in range(0, len(display_px), stride)) if f is not None]
                    if is_full:
                        print(f"[CRT-preview] H3 sweep published: {len(display_px)} frames, {len(samples)} sampled")
                    else:
                        print(f"[CRT-preview] H3 partial: {len(display_px)}/{self._staging_total or '?'} frames, {len(samples)} sampled")
            if not samples or meta is None:
                time.sleep(0.25)
                continue
            if time.monotonic() - self._last_enqueue > _SENDER_IDLE_SECONDS:
                time.sleep(0.5)
                continue
            fps = getattr(self._preview_state, "fps_override", None) if self._preview_state else None
            try:
                fps = float(fps)
            except Exception:
                fps = _DEFAULT_NATIVE_FPS
            if not (fps > 1):
                fps = _DEFAULT_NATIVE_FPS
            real_seconds = len(display_px) / fps
            interval = max(0.01, real_seconds / max(1, len(samples)))
            for image in samples:
                if self._epoch != epoch:
                    break
                started = time.monotonic()
                try:
                    server, preview_event = _get_server()
                    if server is not None:
                        self._send_frame(server, preview_event, image, meta)
                except Exception as e:
                    logging.warning("[CRT-preview] send failed: %s", e)
                if self._epoch != epoch:
                    break
                elapsed = time.monotonic() - started
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    @staticmethod
    def _send_frame(server, preview_event, image, meta):
        """Deliver one frame attributed exactly like core attributes anchors.

        Mirrors comfy_execution.progress.update_handler: the modern frontend
        only renders previews bound to the executing node via
        PREVIEW_IMAGE_WITH_METADATA, keyed by display_node_id and prompt_id.
        Core's own anchor sends carry the authoritative metadata — reuse it.
        """
        snap = _latest_core_preview_meta()
        sid = snap.get("_sid") or meta.get("sid")
        payload = ("JPEG", image, _MAX_PREVIEW_EDGE)
        metadata = {
            "node_id": snap.get("node_id", meta.get("node_id")),
            "prompt_id": snap.get("prompt_id", meta.get("prompt_id")),
            "display_node_id": snap.get("display_node_id", meta.get("node_id")),
            "parent_node_id": snap.get("parent_node_id"),
            "real_node_id": snap.get("real_node_id", meta.get("node_id")),
        }
        try:
            from comfy_api.feature_flags import supports_feature
            tagged = supports_feature(getattr(server, "sockets_metadata", {}) or {}, sid, "supports_preview_metadata")
        except Exception:
            tagged = False
        if tagged:
            server.send_sync(4, (payload, metadata), sid)
        else:
            server.send_sync(preview_event, payload, sid)

    def _ensure_sender(self):
        if self._sender_thread is not None and self._sender_thread.is_alive():
            return
        self._sender_thread = threading.Thread(target=self._sender_loop, name="crt-minimaxh3-preview-sender", daemon=True)
        self._sender_thread.start()


_PREVIEWER_CACHE = {}
_ORIG_GET_PREVIEWER = latent_preview.get_previewer
_OUTER_MARKER = "_crt_minimaxh3_preview_wrapper"
_OUTER_CURRENT = None

class _PreviewState:
    last_latent_shapes = None
    fps_override = _DEFAULT_NATIVE_FPS
    run_token = 0
_PREVIEW_STATE = _PreviewState()

class _PreviewFixGuider:
    def __init__(self, guider, fps_override=None):
        self._guider = guider
        self._fps_override = fps_override
    def __getattr__(self, k):
        return getattr(self._guider, k)
    def sample(self, noise, latent_image, *a, **kw):
        shapes = (tuple(latent_image.shape),) if not getattr(latent_image, "is_nested", False) else tuple(tuple(t.shape) for t in latent_image.unbind())
        _PREVIEW_STATE.last_latent_shapes = shapes
        _PREVIEW_STATE.fps_override = float(self._fps_override) if self._fps_override else _DEFAULT_NATIVE_FPS
        # A new guider.sample invocation is a new denoise run: bump the token
        # so live previewers drop all buffers from the previous run instead of
        # mixing old decodes into the new sweep.
        _PREVIEW_STATE.run_token += 1
        try:
            return self._guider.sample(noise, latent_image, *a, **kw)
        finally:
            _PREVIEW_STATE.last_latent_shapes = None

def _get_previewer_instance(model_path):
    cached = _PREVIEWER_CACHE.get(("previewer", model_path))
    if cached is None:
        cached = CRTMiniMaxH3WrappedPreviewer(model_path=model_path, preview_state=_PREVIEW_STATE)
        _PREVIEWER_CACHE[("previewer", model_path)] = cached
    return cached

def _h3_get_previewer(device, latent_format, *args, **kwargs):
    if is_h3_latent_format(latent_format):
        model_path = _find_taeh3_path()
        if model_path is None:
            model_path = _download_taeh3()
        if model_path is not None:
            try:
                p = _get_previewer_instance(model_path)
                if p._get_model(device) is not None:
                    return p
            except Exception as e:
                print(f"[CRT-preview] skipping {model_path}: {e}")
        return _get_previewer_instance(None)
    delegate = _OUTER_CURRENT or _ORIG_GET_PREVIEWER
    return delegate(device, latent_format, *args, **kwargs)

def ensure_h3_previewer():
    global _OUTER_CURRENT
    current = latent_preview.get_previewer
    if getattr(current, _OUTER_MARKER, False):
        return
    _OUTER_CURRENT = current
    wrapper = _h3_get_previewer
    setattr(wrapper, _OUTER_MARKER, True)
    latent_preview.get_previewer = wrapper

try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web
    @PromptServer.instance.routes.get("/crt/minimaxh3/ensure_preview_model")
    async def _minimaxh3_ensure_preview_model_route(request):
        if taeh3_model_present():
            return _aiohttp_web.json_response({"status": "ready"})
        path = await request.app.loop.run_in_executor(None, _download_taeh3)
        status = "ready" if path else "unavailable"
        return _aiohttp_web.json_response({"status": status})
except Exception as _route_e:
    print(f"[CRT MiniMaxH3] preview route not registered: {_route_e}")


# --- KJNodes-style preview override ---------------------------------------
# Ported from ComfyUI-KJNodes ModelPreviewOverride: instead of the core binary
# preview path (which the modern frontend replaces per message, i.e. a still
# per step), every sampler step decodes several latent frames, encodes ONE
# animated WebP and pushes it as base64 JSON over a custom ws message. The
# frontend DOM widget plays it in a looping <img>, which is a real video.

import base64
import io as _pyio
import queue as _queue

_PREVIEW_OVERRIDE_EVENT = "crt_minimaxh3_preview"
_POV_MAX_RES = 512
_POV_QUALITY = 80
_POV_LATENT_FRAMES = 16
_POV_FRAMES = 18


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
        self.thread = threading.Thread(target=self._run, name="crt_h3_preview_encoder", daemon=True)
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


def _get_prompt_server():
    try:
        from server import PromptServer
        return PromptServer.instance, PromptServer.instance.client_id
    except Exception:
        return None, None


class _H3PreviewOverrideWrapper:
    def __init__(self, node_id):
        self.node_id = str(node_id) if node_id is not None else None

    def _decode_video_frames(self, x0, latent_shapes=None):
        """Decode a temporally-coherent span of the clip.

        Returns (display_frames, total_pixel_frames): display_frames is an
        evenly-spaced subset for the animation; total_pixel_frames is the
        clip's full decoded pixel length, used to bake TRUE-PACE timing.
        """
        fixed = CRTMiniMaxH3WrappedPreviewer._first_frame_shape(x0, latent_shapes)
        if fixed is None:
            return [], 0
        total = int(fixed.shape[2])
        if total <= 0:
            return [], 0
        count = min(_POV_LATENT_FRAMES, total)
        if count < total:
            if count == 1:
                indices = [0]
            else:
                indices = sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})
            selected = fixed[:, :, indices].contiguous()
        else:
            selected = fixed
        previewer = _get_previewer_instance(_find_taeh3_path())
        try:
            pixel_frames = previewer._decode_frames(selected)
        except Exception as e:
            logging.warning(f"[CRT-preview] override decode failed: {e}")
            return [], 0
        if not pixel_frames:
            return [], 0
        # Evenly subsample the decoded pixel frames for the animation.
        n = min(_POV_FRAMES, len(pixel_frames))
        if n < len(pixel_frames):
            if n == 1:
                picks = [0]
            else:
                picks = sorted({round(i * (len(pixel_frames) - 1) / (n - 1)) for i in range(n)})
            display = [pixel_frames[i] for i in picks]
        else:
            display = pixel_frames
        return display, len(pixel_frames)

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        guider = executor.class_obj
        model_patcher = guider.model_patcher
        del guider, model_patcher  # attribution flows through the ws payload only

        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps = max(0, len(sigmas_list) - 1)

        # Boundary-0 message: tells the widget a new run started (sigmas reset).
        server, client_id = _get_prompt_server()
        if server is not None and self.node_id is not None:
            try:
                server.send_sync(
                    _PREVIEW_OVERRIDE_EVENT,
                    {"node_id": self.node_id, "step": 0, "total": total_steps, "sigma": sigmas_list[0] if sigmas_list else None, "sigmas": sigmas_list},
                    client_id,
                )
            except Exception:
                pass

        encoder = _AsyncPreviewEncoder()

        def new_callback(step, x0, x, total_steps_):
            try:
                shapes = latent_shapes if latent_shapes else getattr(_PREVIEW_STATE, "last_latent_shapes", None)
                frames, px_total = self._decode_video_frames(x0, shapes)
                if not frames:
                    raise ValueError("no frames decoded")
                first = frames[0]
                w_, h_ = first.width, first.height
                b64 = None
                mime = "image/jpeg"
                duration_ms = 83
                if len(frames) > 1:
                    # TRUE PACE: one full loop lasts exactly the clip's real
                    # duration (decoded pixel length / native fps), regardless
                    # of how many preview frames are shown.
                    native_fps = getattr(_PREVIEW_STATE, "fps_override", None)
                    try:
                        native_fps = float(native_fps)
                    except Exception:
                        native_fps = 0.0
                    if not (native_fps > 1):
                        native_fps = _DEFAULT_NATIVE_FPS
                    real_seconds = px_total / native_fps
                    duration_ms = int(round(real_seconds * 1000.0 / len(frames)))
                    b64 = _encode_animated_webp(frames, duration_ms)
                    mime = "image/webp"
                if not b64:
                    if max_res and (first.width > _POV_MAX_RES or first.height > _POV_MAX_RES):
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
                        _PREVIEW_OVERRIDE_EVENT,
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

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.shutdown(drain_timeout=5.0)


def apply_h3_preview_override(model, node_id):
    """Attach the animated-WebP live preview wrapper to the sampling model.

    The wrapper key is 'kj_preview_override' so the (embedded) Spectrum engine
    recognizes it as the observational preview and bypasses it during the
    transformer-free offline replay pass.
    """
    import comfy.patcher_extension

    m = model.clone()
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "kj_preview_override",
        _H3PreviewOverrideWrapper(node_id),
    )
    return m

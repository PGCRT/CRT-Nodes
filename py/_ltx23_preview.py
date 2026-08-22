"""Native LTX2 live-preview decoder for the CRT unified sampler.

Ported from the KJNodes LTX preview implementation (WrappedPreviewer):
prefers a taeltx TAEHV decode when the checkpoint is available and falls
back to the LTX2 latent-to-RGB factor projection otherwise.

Playback model (sweep double-buffering):
  * Every sampler tick decodes the next latent chunk into a STAGING buffer.
  * A sweep is one full pass over all latent frames of the clip. A sweep's
    frames share the same denoise progress, so a sweep is only publishable
    once complete.
  * The sender thread plays the last PUBLISHED sweep, looping it at true
    temporal pace. Incomplete sweeps are never displayed; fresh decodes
    appear on the pass after their sweep completed.

No KJNodes imports; the projection tables live in _ltx23_preview_tables.
"""

import logging
import math
import threading
import time
import weakref

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.utils
import latent_preview

from ._ltx23_preview_tables import _LTX2_RGB_BIAS, _LTX2_RGB_FACTORS

# Display rate cap and decode budget. Playback interval is derived from the
# video's real duration, so these shape cost/smoothness, not temporal pace.
_PREVIEW_FPS = 8.0
_DECODE_LATENT_PER_TICK = 4
_MAX_PREVIEW_EDGE = 512
_DEFAULT_NATIVE_FPS = 24.0
_SENDER_IDLE_SECONDS = 12.0

# Every live previewer, so runs can wipe all timelines at execution start.
_LIVE_PREVIEWERS = weakref.WeakSet()


def reset_all_timelines():
    """Wipe every cached timeline. Called when a new run starts."""
    for previewer in list(_LIVE_PREVIEWERS):
        try:
            previewer.reset_timeline()
        except Exception:
            pass


def _get_server():
    """Return (PromptServer.instance, UNENCODED_PREVIEW_IMAGE event id)."""
    try:
        from protocol import BinaryEventTypes
        from server import PromptServer

        return PromptServer.instance, BinaryEventTypes.UNENCODED_PREVIEW_IMAGE
    except Exception:
        return None, None


def build_taehv(model_path):
    """Build a standard TAEHV(latent_channels=128) and load weights directly."""
    from comfy.taesd.taehv import TAEHV

    taehv = TAEHV(latent_channels=128)
    sd = comfy.utils.load_torch_file(model_path)
    taehv.load_state_dict(sd, strict=True)
    taehv.eval()
    taehv.show_progress_bar = False
    return taehv


class CRTLTXWrappedPreviewer(latent_preview.TAEHVPreviewerImpl):
    """True-pace looping preview with atomic sweep snapshots."""

    _blank = Image.new("RGB", (256, 256), (8, 8, 8))

    def __init__(self, model_path=None, preview_state=None):
        # Bypass TAEHVPreviewerImpl.__init__: we manage weights ourselves.
        self.taesd = None
        self._model_path = model_path
        self._model = None
        self._model_failed = False
        self._preview_state = preview_state
        self._factors = torch.tensor(_LTX2_RGB_FACTORS).transpose(0, 1)
        self._bias = torch.tensor(_LTX2_RGB_BIAS)
        self._cursor = 0
        self._frames_key = None
        self._stream_logged = False

        # Sweep state (shared with sender thread under _lock).
        self._lock = threading.Lock()
        self._display_px = []      # complete published sweep being played
        self._native_fps = _DEFAULT_NATIVE_FPS
        self._staging_px = []      # accumulating current sweep
        self._staging_filled = 0   # latent frames staged in current sweep
        self._publish_ready = False

        # Run attribution + lifecycle.
        self._meta = None
        self._epoch = 0           # bumped on every reset; sender drops its playlist
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
            self._publish_ready = False
            self._meta = None
            self._stream_logged = False

    def _get_model(self, device):
        if self._model is not None or self._model_path is None or self._model_failed:
            return self._model
        try:
            self._model = build_taehv(self._model_path).to(device)
        except Exception as error:
            self._model_failed = True
            logging.warning(
                "[CRT LTX23] taeltx preview model unusable (%s); using RGB factors",
                error,
            )
        return self._model

    @staticmethod
    def _ensure_video_latent_shape(x0, last_shapes):
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

    def _flatten_to_frames(self, x0):
        """Return (F, C, H, W) frame list from any supported latent shape."""
        last_shapes = (
            self._preview_state.last_latent_shapes if self._preview_state else None
        )
        fixed = self._ensure_video_latent_shape(x0, last_shapes)
        if fixed is None:
            return None
        if fixed.ndim == 5:
            batch, channels, frames, height, width = fixed.shape
            frames = fixed[0].transpose(0, 1).reshape(-1, channels, height, width)
        else:
            frames = fixed
        return frames

    def _resolve_native_fps(self):
        fps = (
            getattr(self._preview_state, "fps_override", None)
            if self._preview_state
            else None
        )
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 0.0
        return fps if fps > 1.0 else _DEFAULT_NATIVE_FPS

    def _decode_frames(self, slice_fchw):
        """Decode (F,C,H,W) latents to a list of PIL images."""
        device = slice_fchw.device
        model = self._get_model(device)
        if model is not None:
            try:
                with torch.no_grad():
                    out = model.decode(slice_fchw.unsqueeze(0))
                    out = out[0].permute(1, 2, 3, 0).float().cpu()
                return [
                    latent_preview.preview_to_image(frame, do_scale=False)
                    for frame in out
                ]
            except Exception as error:
                logging.warning(
                    "[CRT LTX23] taeltx preview decode failed (%s); using RGB factors",
                    error,
                )
                self._model_failed = True

        images = []
        factors = self._factors.to(dtype=slice_fchw.dtype, device=device)
        bias = self._bias.to(dtype=slice_fchw.dtype, device=device)
        for frame in slice_fchw:
            chw = frame.movedim(0, -1)
            rgb = torch.sigmoid(F.linear(chw, factors, bias))
            array = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            images.append(Image.fromarray(array))
        return images

    def decode_latent_to_preview_image(self, preview_format, x0):
        server, preview_event = _get_server()
        if server is None:
            frames = self._flatten_to_frames(x0)
            if frames is None:
                return self._blank
            mid = frames.size(0) // 2
            return self._decode_frames(frames[mid : mid + 1])[0]

        frames = self._flatten_to_frames(x0)
        if frames is None:
            return self._blank

        # Freeze attribution to THIS run while our node executes; an unknown
        # node id means a new run that bypassed reset (safety net).
        with self._lock:
            stale = self._meta is None or self._meta.get("node_id") != getattr(
                server, "last_node_id", None
            )
            if stale:
                self._meta = {
                    "sid": getattr(server, "client_id", None),
                    "node_id": getattr(server, "last_node_id", None),
                    "prompt_id": getattr(server, "last_prompt_id", None),
                }
                self._epoch += 1
                self._cursor = 0
                self._frames_key = None
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False
                self._display_px = []

        key = tuple(frames.shape)
        if key != self._frames_key:
            # New stage / new resolution: restart the sweep and stop any
            # playback of the old shape.
            self._frames_key = key
            self._cursor = 0
            with self._lock:
                self._epoch += 1
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False
                self._display_px = []

        total = frames.size(0)

        # A tick starting at 0 begins a NEW sweep: discard whatever partial
        # staging existed so sweeps never mix generations.
        start = self._cursor % total
        if start == 0:
            with self._lock:
                self._staging_px = []
                self._staging_filled = 0
                self._publish_ready = False

        count = min(_DECODE_LATENT_PER_TICK, total - start)
        end = start + count
        chunk = frames[start:end]
        self._cursor = end % total

        images = self._decode_frames(chunk)

        with self._lock:
            base_px = start * max(1, len(images) // max(1, count))
            needed = base_px + len(images)
            if len(self._staging_px) < needed:
                self._staging_px.extend([None] * (needed - len(self._staging_px)))
            for i, image in enumerate(images):
                self._staging_px[base_px + i] = image
            self._staging_filled += count
            if self._staging_filled >= total:
                self._publish_ready = True
        anchor = images[0]
        self._last_enqueue = time.monotonic()
        self._ensure_sender()

        if not self._stream_logged:
            self._stream_logged = True
            decoder = (
                "taeltx"
                if self._model is not None and not self._model_failed
                else "RGB factors"
            )
            print(
                f"[CRT-preview] true-pace streaming @ {_PREVIEW_FPS:.0f}fps display "
                f"({decoder} decoder)"
            )

        # Anchor frame through the standard path per step (this one MAY be
        # partial-sweep by design: it reflects live denoise progress).
        return (preview_format or "JPEG", anchor, _MAX_PREVIEW_EDGE)

    def _sender_loop(self):
        display_px = []
        samples = []
        native_fps = _DEFAULT_NATIVE_FPS
        epoch = -1
        while True:
            with self._lock:
                meta = self._meta
                if epoch != self._epoch:
                    # Run was reset: drop the previous run's playlist
                    # immediately instead of looping stale frames.
                    epoch = self._epoch
                    display_px = []
                    samples = []
                if self._publish_ready and self._staging_filled > 0:
                    # Atomic swap at loop boundary: promote the completed
                    # sweep wholesale. Never merge partial staging into a
                    # playing loop.
                    display_px = self._staging_px
                    self._display_px = display_px
                    self._staging_px = []
                    self._staging_filled = 0
                    self._publish_ready = False
                    native_fps = self._resolve_native_fps()

                    stride = max(1, round(native_fps / _PREVIEW_FPS))
                    samples = [
                        f
                        for f in (
                            display_px[i] for i in range(0, len(display_px), stride)
                        )
                        if f is not None
                    ]
                    logging.info(
                        "[CRT-preview] sweep published: %d px frames, %d sampled",
                        len(display_px),
                        len(samples),
                    )

            if not samples or meta is None:
                time.sleep(0.25)
                continue

            # Go silent between runs so unrelated executions stay untouched.
            if time.monotonic() - self._last_enqueue > _SENDER_IDLE_SECONDS:
                time.sleep(0.5)
                continue

            # True temporal pace: one full loop lasts exactly the clip's
            # real duration no matter how many frames are displayed.
            real_seconds = len(display_px) / native_fps
            interval = max(0.01, real_seconds / max(1, len(samples)))

            for image in samples:
                started = time.monotonic()
                try:
                    server, preview_event = _get_server()
                    if server is not None:
                        self._send_frame(server, preview_event, image, meta)
                except Exception as error:
                    logging.warning("[CRT-preview] preview send failed: %s", error)
                if self._epoch != epoch:
                    break
                elapsed = time.monotonic() - started
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    @staticmethod
    def _send_frame(server, preview_event, image, meta):
        """Deliver one frame attributed to the run that decoded it.

        Mirrors comfy_execution.progress.update_handler: modern frontends only
        render previews bound to the executing node via
        PREVIEW_IMAGE_WITH_METADATA; plain UNENCODED events are ignored.
        """
        sid = meta.get("sid")
        payload = ("JPEG", image, _MAX_PREVIEW_EDGE)
        try:
            from comfy_api.feature_flags import supports_feature

            tagged = supports_feature(
                getattr(server, "sockets_metadata", {}) or {},
                sid,
                "supports_preview_metadata",
            )
        except Exception:
            tagged = False

        if tagged:
            metadata = {
                "node_id": meta.get("node_id"),
                "prompt_id": meta.get("prompt_id"),
                "display_node_id": meta.get("node_id"),
                "real_node_id": meta.get("node_id"),
            }
            # BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA
            server.send_sync(4, (payload, metadata), sid)
        else:
            server.send_sync(preview_event, payload, sid)

    def _ensure_sender(self):
        if self._sender_thread is not None and self._sender_thread.is_alive():
            return
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="crt-ltx23-preview-sender",
            daemon=True,
        )
        self._sender_thread.start()

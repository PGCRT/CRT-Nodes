import gc
import math
import weakref

import torch

import comfy.model_management as mm
import comfy.model_sampling
import comfy.utils
import latent_preview
import nodes

from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    align_frame_count,
)

from comfy_extras.nodes_audio import vae_decode_audio

from ._cache_fingerprint import stable_fingerprint
from ._minimaxh3_preview import _PreviewFixGuider, _PREVIEW_STATE, apply_h3_preview_override, kickoff_taeh3_download, wipe_all_caches
from . import MiniMaxUSOpt

MODE_T2V = "T2V"
MODE_FL2VA = "I2V"
MODE_REF2VA = "R2V"
WORKFLOW_MODES = (MODE_T2V, MODE_FL2VA, MODE_REF2VA)

FL_ASPECT_MODES = ("Preserve First", "Preserve Last", "Optimal")

FPS = 24.0
SAMPLER_NAME = "res_multistep"
SCHEDULER_NAME = "simple"
STEPS_FULL_DEFAULT = 20
STEPS_TURBO_DEFAULT = 4

SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

# Official Ref2VA limits: <=9 images, <=3 videos, <=3 standalone audios,
# <=12 files mixed. The native MiniMaxH3ReferenceToVideo node matches these.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

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


def _active_family(workflow_mode):
    """T2V and FL2VA share the FL2VA checkpoint family."""
    return "ref2va" if str(workflow_mode) == MODE_REF2VA else "fl2va"


class CRT_MiniMaxH3USModelsPipe:
    SAMPLER_CLASS = "CRT_MiniMaxH3UnifiedSampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE. Decodes generated audio; REF2VA also uses it to encode reference soundtracks."}),
                "clip": ("CLIP", {"tooltip": "Qwen3-VL MiniMax text/image encoder."}),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True, "tooltip": "FL2VA diffusion model. Used by T2V and FL2VA modes; loaded only when those modes run."}),
                "fl2va_turbo_model": ("MODEL", {"lazy": True, "tooltip": "FL2VA base merged with the FL2VA Turbo LoRA. Loaded only when Turbo is enabled in an FL2VA-family mode."}),
                "ref2va_model": ("MODEL", {"lazy": True, "tooltip": "REF2VA diffusion model. Used by the REF2VA mode; loaded only when REF2VA runs."}),
                "ref2va_turbo_model": ("MODEL", {"lazy": True, "tooltip": "REF2VA base merged with the REF2VA Turbo LoRA. Loaded only when Turbo is enabled in REF2VA."}),
            },
            "hidden": {
                "minimax_h3_us_prompt": "DYNPROMPT",
                "minimax_h3_us_unique": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MINIMAXH3_US_MODELS_PIPE",)
    RETURN_NAMES = ("models_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Bundles the MiniMax H3 model variants, video/audio VAEs, and CLIP. Reads workflow mode and Turbo state from the connected Unified Sampler(s) and lazy-loads - or downloads - only the variant the current run actually uses."

    @classmethod
    def _iter_prompt_nodes(cls, prompt):
        """Yield (node_id, node) over plain dicts and DynamicPrompts alike.

        DYNPROMPT is required so subgraph-expanded nodes ("24:30"-style ids)
        are visible; the original prompt only contains subgraph placeholders.
        """
        if prompt is None:
            return
        get_node = getattr(prompt, "get_node", None)
        all_ids = getattr(prompt, "all_node_ids", None)
        if callable(get_node) and callable(all_ids):
            for node_id in all_ids():
                try:
                    node = get_node(node_id)
                except Exception:
                    continue
                if isinstance(node, dict):
                    yield str(node_id), node
            return
        if isinstance(prompt, dict):
            for node_id, node in prompt.items():
                yield str(node_id), node

    @classmethod
    def _sampler_needs(cls, inputs, unique_id, exact_only=False):
        link = inputs.get("models_pipe")
        if not (isinstance(link, (list, tuple)) and len(link) >= 1):
            return None
        if exact_only and str(link[0]) != str(unique_id):
            return None
        mode = inputs.get("workflow_mode", MODE_FL2VA)
        turbo = bool(inputs.get("turbo", False))
        family = _active_family(mode)
        return f"{family}_turbo_model" if turbo else f"{family}_model"

    @classmethod
    def _required_model_keys(cls, prompt, unique_id):
        """Find Unified Samplers consuming THIS pipe and their needed variants.

        Primary match: sampler links pointing at this pipe's unique id. If the
        graph aliases ids (subgraph expansion variants), fall back to the union
        of every sampler's needs so the run still gets the right weights.
        """
        needs = set()
        fallback_needs = set()
        found_exact = False
        for node_id, node in cls._iter_prompt_nodes(prompt):
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != cls.SAMPLER_CLASS:
                continue
            needed = cls._sampler_needs(node.get("inputs", {}), unique_id, exact_only=True)
            if needed is not None:
                found_exact = True
                needs.add(needed)
                continue
            needed = cls._sampler_needs(node.get("inputs", {}), unique_id, exact_only=False)
            if needed is not None:
                fallback_needs.add(needed)
        if not found_exact:
            if fallback_needs:
                print(
                    "[CRT MiniMaxH3] Models Pipe id not matched exactly; "
                    f"using union of all Unified Samplers in this run: {sorted(fallback_needs)}"
                )
            return fallback_needs
        return needs

    _MODEL_INPUT_KEYS = ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
    # Bump when pipe-building semantics change: older cached outputs (which may
    # have baked None into unevaluated sockets) must invalidate exactly once.
    _PIPE_CACHE_VERSION = "v3-all-sockets"

    @classmethod
    def _connected_model_keys(cls, prompt, unique_id):
        """Which of the four model sockets have links on THIS pipe node.

        Reads the prompt graph: connected optional inputs appear as
        [node_id, slot] link lists; unconnected ones are absent/None.
        Matching is defensive because subgraph expansion mangles ids: exact
        uid, then uid-suffix, then the single pipe, else the union across
        every pipe node (over-requesting a connected loader is harmless).
        """
        connected = set()
        pipe_nodes = []
        for node_id, node in cls._iter_prompt_nodes(prompt):
            if not isinstance(node, dict):
                continue
            ctype = str(node.get("class_type", ""))
            if ctype != "CRT_MiniMaxH3USModelsPipe" and not ctype.endswith("CRT_MiniMaxH3USModelsPipe"):
                continue
            inputs = node.get("inputs", {})
            pipe_nodes.append((str(node_id), inputs))
            for k in cls._MODEL_INPUT_KEYS:
                link = inputs.get(k)
                if isinstance(link, (list, tuple)) and len(link) >= 1:
                    connected.add(k)
        if not pipe_nodes:
            return set()
        uid = str(unique_id)
        for node_id, inputs in pipe_nodes:
            if node_id == uid or node_id.endswith(":" + uid) or uid.endswith(":" + node_id):
                connected = {
                    k
                    for k in cls._MODEL_INPUT_KEYS
                    if isinstance(inputs.get(k), (list, tuple)) and len(inputs.get(k)) >= 1
                }
                break
        return connected

    @classmethod
    def check_lazy_status(
        cls,
        minimax_h3_us_prompt=None,
        minimax_h3_us_unique=None,
        fl2va_model=None,
        fl2va_turbo_model=None,
        ref2va_model=None,
        ref2va_turbo_model=None,
        **_other_inputs,
    ):
        # Connected-but-unevaluated lazy inputs arrive as None, so requesting a
        # key forces the engine to evaluate that loader branch; unconnected ones
        # surface as a clear missing-input error naming the exact socket.
        available = {
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        needed = cls._required_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
        # Request EVERY connected loader, not just the active variant: the pipe
        # must carry all four models so mode/turbo switches never see a stale
        # None baked into an unevaluated socket.
        connected = cls._connected_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
        needed = needed | connected
        sampler_ids = [
            nid
            for nid, node in cls._iter_prompt_nodes(minimax_h3_us_prompt)
            if isinstance(node, dict) and node.get("class_type") == cls.SAMPLER_CLASS
        ]
        if not needed:
            # No sampler matched this pipe's uid (common when the pipe lives
            # inside a subgraph and the sampler is outside, or vice versa).
            # Fall back to evaluating every connected loader so the sampler
            # can at least fall back to an available family instead of
            # crashing with "no model connected".
            print(
                "[CRT MiniMaxH3] Models Pipe: no exact Unified Sampler match; "
                f"evaluating all connected loaders for fallback."
            )
            needed = {k for k, v in available.items() if v is None}
            # If nothing is connected yet, still request all 4 to surface a
            # clear "connect a model" error from build_pipe.
            if not needed:
                needed = set(available.keys())
        return sorted(key for key in needed if available[key] is None)

    @classmethod
    def IS_CHANGED(cls, minimax_h3_us_prompt=None, minimax_h3_us_unique=None, **kwargs):
        # Bust the cache when the sampler's required variant changes OR when
        # the evaluation state of any of the four model inputs changes (a
        # newly-connected loader must never keep serving a stale pipe that
        # baked None into that socket).
        try:
            needed = cls._required_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
            presence = tuple(
                kwargs.get(k) is not None
                for k in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
            )
            return (
                hash(frozenset(needed))
                ^ hash(presence)
                ^ hash(str(minimax_h3_us_unique))
                ^ hash(cls._PIPE_CACHE_VERSION)
            )
        except Exception:
            return float("nan")

    def build_pipe(
        self,
        vae,
        audio_vae,
        clip,
        fl2va_model=None,
        fl2va_turbo_model=None,
        ref2va_model=None,
        ref2va_turbo_model=None,
        **_hidden,
    ):
        # Fail loudly instead of emitting a None-filled pipe: a returned output
        # here would be cached and served forever without re-running the lazy
        # resolution, permanently starving downstream samplers.
        available = {
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        needed = self._required_model_keys(
            _hidden.get("minimax_h3_us_prompt"), _hidden.get("minimax_h3_us_unique")
        )
        if not needed:
            # Never return a None-filled pipe from here: it would be cached and
            # served forever without re-running lazy resolution. Fail loudly so
            # this is visible instead of starving downstream samplers silently.
            raise ValueError(
                "MiniMax H3 US Models Pipe: could not find a Unified Sampler for "
                f"this run (pipe uid={_hidden.get('minimax_h3_us_unique')!r}). "
                "Check that this pipe output feeds a MiniMax H3 Unified Sampler (CRT)."
            )
        missing = sorted(key for key in needed if available[key] is None)
        if missing:
            raise ValueError(
                "MiniMax H3 US Models Pipe: "
                + ", ".join(f"'{k}'" for k in missing)
                + " is required by the connected Unified Sampler but nothing is "
                "connected to that socket."
            )
        pipe = {
            "vae": vae,
            "audio_vae": audio_vae,
            "clip": clip,
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        return (pipe,)


class CRT_MiniMaxH3USConfig:
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
                        "tooltip": "Official prompt structure: 'integrated_multimodal_description:' (shots/motion), 'overall_soundscape:' (ambient/dialogue/SFX) and, optionally, 'non_diegetic_music:'. In R2V address references as <Picture i> / <Video k> / <Audio j>.",
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
                "First Frame (I2V)": (
                    "IMAGE",
                    {"tooltip": "FL2VA starting keyframe. The first frame is a geometry anchor stretched to the canvas."},
                ),
                "Last Frame (I2V)": (
                    "IMAGE",
                    {"tooltip": "FL2VA ending keyframe. Aspect-preserving cover-crop; motion is generated between both frames."},
                ),
                **{
                    f"Ref Image {i} (REF2VA)": (
                        "IMAGE",
                        {"tooltip": f"REF2VA reference image {i}, addressed as <Picture {i}> in the prompt."},
                    )
                    for i in range(1, MAX_REF_IMAGES + 1)
                },
                **{
                    f"Ref Video {i} (REF2VA)": (
                        "IMAGE",
                        {"tooltip": f"REF2VA reference video {i} as an IMAGE batch at 24 fps (2-15s at 24 fps, 48+ frames recommended), addressed as <Video {i}> in the prompt."},
                    )
                    for i in range(1, MAX_REF_VIDEOS + 1)
                },
                **{
                    f"Ref Video Audio {i} (REF2VA)": (
                        "AUDIO",
                        {"tooltip": f"Soundtrack paired with Ref Video {i}; addressed as its own <Audio> tag before <Video {i}>."},
                    )
                    for i in range(1, MAX_REF_VIDEOS + 1)
                },
                **{
                    f"Ref Audio {i} (REF2VA)": (
                        "AUDIO",
                        {"tooltip": f"Standalone REF2VA reference audio {i}, addressed as <Audio j> in the prompt."},
                    )
                    for i in range(1, MAX_REF_AUDIOS + 1)
                },
                "Frames (override)": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1, "forceInput": True, "tooltip": "Values above 0 override the sampler frame count; snapped up to the 17n+5 grid. 0 keeps the sampler setting."},
                ),
                "MegaPixels (override)": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.02, "forceInput": True, "tooltip": "Values above 0 override the sampler megapixels_target; 0 keeps the sampler setting."},
                ),
            },
        }

    RETURN_TYPES = ("MINIMAXH3_US_CONFIG_PIPE",)
    RETURN_NAMES = ("config_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Collects the prompt, seed, optional keyframes and REF2VA reference media, plus per-workflow overrides for the unified sampler."

    def build_pipe(self, prompt, seed, **kwargs):
        first_frame = kwargs.get("First Frame (I2V)", None)
        last_frame = kwargs.get("Last Frame (I2V)", None)
        override_frames = kwargs.get("Frames (override)", None)
        override_megapixels = kwargs.get("MegaPixels (override)", None)

        ref_images = {}
        for i in range(1, MAX_REF_IMAGES + 1):
            img = kwargs.get(f"Ref Image {i} (REF2VA)", None)
            if img is not None:
                ref_images[f"ref_image_{i - 1}"] = img
        ref_videos = {}
        ref_video_audios = {}
        for i in range(1, MAX_REF_VIDEOS + 1):
            vid = kwargs.get(f"Ref Video {i} (REF2VA)", None)
            if vid is not None:
                ref_videos[f"ref_video_{i - 1}"] = vid
                aud = kwargs.get(f"Ref Video Audio {i} (REF2VA)", None)
                if aud is not None:
                    ref_video_audios[f"ref_video_audio_{i - 1}"] = aud
        ref_audios = {}
        for i in range(1, MAX_REF_AUDIOS + 1):
            aud = kwargs.get(f"Ref Audio {i} (REF2VA)", None)
            if aud is not None:
                ref_audios[f"ref_audio_{i - 1}"] = aud

        def _optional_int(v):
            if v is None:
                return None
            try:
                iv = int(v)
            except Exception:
                return None
            return iv if iv > 0 else None

        def _optional_float(v):
            if v is None:
                return None
            try:
                fv = float(v)
            except Exception:
                return None
            return fv if fv > 0 else None

        pipe = {
            "prompt": str(prompt),
            "seed": int(seed),
            "first_frame": first_frame if first_frame is None else first_frame[:1],
            "last_frame": last_frame if last_frame is None else last_frame[:1],
            "ref_images": ref_images,
            "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios,
            "override_frames": _optional_int(override_frames),
            "override_megapixels": _optional_float(override_megapixels),
        }
        return (pipe,)


def _apply_sigma_shift(model, shift_video, shift_audio):
    """Mirror of the native MiniMaxH3SigmaShift operation."""
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)

    to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    return m


# --- Per-token prompt weighting (Krea2PromptWeight port) --------------------
# H3's Qwen3-VL presentation is NOT chat-templated (raw prompt tokens), and the
# tokenizer runs with disable_weights=True, so ComfyUI's native (word:weight)
# does nothing. Same trick as KJNodes Krea2PromptWeight: scale the weighted
# tokens' attention VALUE (de-emphasis / removal at weight<1) and bias their
# attention LOGIT (emphasis at weight>1), patched into every DiT block's attn.

_PROMPT_WEIGHT_PATTERN = None


def _h3_prompt_weight_pattern():
    global _PROMPT_WEIGHT_PATTERN
    if _PROMPT_WEIGHT_PATTERN is None:
        import re
        _PROMPT_WEIGHT_PATTERN = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
    return _PROMPT_WEIGHT_PATTERN


def _h3_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] for t in tok[key][0]]


def _h3_find_subsequence(seq, sub):
    n = len(sub)
    out = []
    if n == 0:
        return out
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def _h3_parse_prompt_weights(clip, prompt, log):
    """Return (weight_pairs, cleaned_prompt). pairs = (pos, v_factor, k_bias)."""
    pattern = _h3_prompt_weight_pattern()
    terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(prompt)]
    if not terms:
        return [], prompt
    clean = pattern.sub(lambda m: m.group(1), prompt)
    ids = _h3_token_ids(clip, clean)
    pairs = []
    for phrase, w in terms:
        if w > 1.0:
            v_factor, k_bias = 1.0, (w - 1.0) * 2.0   # emphasis via attention boost
        else:
            v_factor, k_bias = 1.0 + (w - 1.0), 0.0   # de-emphasis / removal via value scaling
        positions = []
        for variant in (" " + phrase, phrase):  # words usually carry a leading-space token
            sub = _h3_token_ids(clip, variant)
            matches = _h3_find_subsequence(ids, sub)
            if matches:
                for mi in matches:
                    positions.extend(mi + off for off in range(len(sub)))
                break
        if not positions:
            log(f"Prompt weight: phrase '{phrase}' not found in prompt; skipped.", level="warn")
            continue
        for cp in positions:
            pairs.append((cp, v_factor, k_bias))
    return pairs, clean


class _H3WeightPatch:
    """Descriptor binding the weighting attention forward onto H3's Attention."""

    def __get__(self, obj, objtype=None):
        import types
        return types.MethodType(_h3_attn_forward_weight, obj)


def _h3_attn_forward_weight(self, x, rope_freqs=None, transformer_options={}):
    import comfy.model_management
    from comfy.ldm.modules.attention import (
        AttentionTensorContainer,
        attention_pytorch,
        optimized_attention,
    )

    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    v = v.clone()
    weights = transformer_options.get("minimax_h3_token_weights")
    if weights:
        for pos, v_factor, _ in weights:
            if v_factor != 1.0 and pos < s:
                v[pos] = v[pos] * v_factor
    bias = None
    if weights and any(kb != 0.0 for _, _, kb in weights):
        bias = q.new_zeros(1, s)
        for pos, _, kb in weights:
            if kb != 0.0 and pos < s:
                bias[:, pos] = kb
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    if bias is not None:
        # per-key logit bias needs the raw sdpa path; the optimized dispatcher
        # only forwards its own mask conventions
        out = attention_pytorch(q, k, v, self.heads, mask=bias, skip_reshape=True)
    else:
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))


class CRT_MiniMaxH3UnifiedSampler:
    COLOR_INFO = "\033[38;5;117m"
    COLOR_WARN = "\033[38;5;208m"
    COLOR_OK = "\033[38;5;120m"
    COLOR_RESET = "\033[0m"

    # Speed-optimization patch defaults follow the tuned reference chain.
    SOL_DEFAULTS = dict(
        tau_start=1.15,
        tau_end=0.8,
        curve="smoothstep",
        min_tokens=4096,
        strict=False,
        dense_percent=0.0,
        thresh_type="diag",
        int8_qk=False,
        int8_pv=False,
        sink_conditioning="exact_kv",
        dense_blocks="",
    )
    CHUNK_FF_DEFAULTS = dict(chunks=2, min_tokens=8192)
    SPECTRUM_DEFAULTS = dict(
        blend_weight=0.5,
        degree=1,
        ridge_lambda=0.1,
        window_size=2,
        flex_window=0.75,
        warmup_steps=1,
        tail_actual_steps=1,
        max_history=8,
        debug=False,
        history_storage="system_ram",
        bootstrap_first_forecast=True,
        anchor_residual_feedback=False,
        selective_rollback_correction=False,
        offline_smoothing_replay=True,
        audio_blend_weight=0.0,
        offline_archive_storage="system_ram",
        model_aware_mode="off",
        model_aware_risk_threshold=0.65,
        model_aware_trust_shrinkage=False,
        model_aware_replay_generic_correction=False,
        generic_correction_mode="coordinate_rls",
        generic_correction_limiter="hard_clip",
        generic_correction_limit=0.4,
        generic_correction_attenuation="no_attenuation",
    )

    # Single-entry prompt-only conditioning cache (the expensive Qwen encode).
    # Guarded by weakrefs so a freed/reallocated model can never satisfy a hit,
    # and only used when NO media is attached - media paths always rebuild.
    _TEXT_COND_CACHE = {
        "key": None,
        "clip": None,
        "vae": None,
        "audio_vae": None,
        "value": None,
    }

    @classmethod
    def _log(cls, message, level="info"):
        color = cls.COLOR_INFO
        if level == "warn":
            color = cls.COLOR_WARN
        elif level == "ok":
            color = cls.COLOR_OK
        print(f"{color}[CRT MiniMaxH3]{cls.COLOR_RESET} {message}")

    @classmethod
    def _progress(cls, step, total, label):
        width = 18
        filled = max(0, min(width, int(round((step / float(max(1, total))) * width))))
        bar = "#" * filled + "-" * (width - filled)
        cls._log(f"[{step}/{total}] {bar} {label}")

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

    @classmethod
    def IS_CHANGED(
        cls,
        models_pipe,
        config_pipe,
        workflow_mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        live_preview,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db,
        **_other_inputs,
    ):
        return stable_fingerprint(
            models_pipe,
            config_pipe,
            workflow_mode,
            int(steps),
            int(steps_turbo),
            bool(turbo),
            bool(enable_sol_attn),
            bool(enable_chunk_ff),
            bool(enable_spectrum),
            bool(live_preview),
            bool(vae_decode_tiled),
            bool(unload_before_decode),
            bool(low_vram),
            float(megapixels_target),
            str(aspect_ratio),
            str(fl_aspect_mode),
            int(length_frames),
            bool(audio_frames_override),
            bool(video_frames_override),
            float(generated_audio_gain_db),
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_pipe": ("MINIMAXH3_US_MODELS_PIPE",),
                "config_pipe": ("MINIMAXH3_US_CONFIG_PIPE",),
                "workflow_mode": (
                    WORKFLOW_MODES,
                    {"default": MODE_FL2VA, "tooltip": "T2V: pure text (FL2VA checkpoint). FL2VA (I2V): first/last-frame keyframes. REF2VA (R2V): reference images/videos/audio. Selects which model variant the Models Pipe loads."},
                ),
                "steps": (
                    "INT",
                    {"default": STEPS_FULL_DEFAULT, "min": 1, "max": 60, "step": 1, "tooltip": "Steps used when Turbo is OFF."},
                ),
                "steps_turbo": (
                    "INT",
                    {"default": STEPS_TURBO_DEFAULT, "min": 1, "max": 60, "step": 1, "tooltip": "Steps used when Turbo is ON (the official Turbo LoRAs are trained for 4)."},
                ),
                "turbo": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use the Turbo-LoRA variant of the active mode's family from the Models Pipe, together with the Steps Turbo count."},
                ),
                "enable_sol_attn": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Scheduled Sol attention: sparsifies self-attention on high-noise steps (tau ramp), denser near the end. Applied after the sigma shift so its schedule matches."},
                ),
                "enable_chunk_ff": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Chunked feed-forward: splits each MLP pass to reduce peak VRAM. Composable with Sol attention and Spectrum."},
                ),
                "enable_spectrum": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Spectrum forecast: predicts solver steps from past denoised anchors to cut sampling NFEs. Applied last so it wraps the final patched model."},
                ),
                "live_preview": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Decode intermediate video previews during sampling via the auto-downloaded taeh3 approximation (RGB-factor fallback when offline). Adds per-step decode overhead - disabled by default."},
                ),
                "vae_decode_tiled": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode video latents in tiles to reduce peak VRAM at the cost of additional processing time."},
                ),
                "unload_before_decode": (
                    "BOOLEAN",
                    {"default": True, "advanced": True, "tooltip": "Unload the diffusion model after sampling and before VAE decode to reduce decode-time VRAM."},
                ),
                "low_vram": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Unload CLIP after conditioning and the VAEs before sampling, then reload the VAEs for decode."},
                ),
                "megapixels_target": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.05,
                        "max": 2.0,
                        "step": 0.02,
                        "tooltip": "Target canvas area in megapixels, ceiled to the model's 32px grid (0.98 at 16:9 is the official 1344x768 768p canvas).",
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {"default": "16:9 (Landscape)", "tooltip": "Canvas aspect. Ignored in I2V when a First/Last frame is connected - the frame sets the canvas."},
                ),
                "fl_aspect_mode": (
                    FL_ASPECT_MODES,
                    {
                        "default": "Preserve First",
                        "tooltip": "I2V with BOTH frames connected: how to reconcile different aspect ratios. Preserve First/Last: that frame defines the canvas and the other is cover-cropped to it. Optimal: a middle-ground canvas (geometric mean of both ratios) at the megapixel target; both frames are cover-cropped to it.",
                    },
                ),
                "length_frames": (
                    "INT",
                    {
                        "default": 124,
                        "min": 5,
                        "max": 362,
                        "step": 1,
                        "tooltip": "Clip length in frames at 24 fps, snapped to the model's 17n+5 grid (124 frames = ~5s; trained range is 124-362).",
                    },
                ),
                "audio_frames_override": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "REF2VA only. OFF (default): Duration (frames) is a hard cap - the output length equals it and longer references are trimmed, never stretched. ON: the output length derives from the longest Ref Audio instead (no cap - long inputs risk OOM). Output is always snapped to the 17n+5 grid at 24 fps."},
                ),
                "video_frames_override": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "REF2VA only. OFF (default): Duration (frames) is a hard cap - the output length equals it and longer references are trimmed, never stretched. ON: the output length derives from the longest Ref Video instead, taking priority over the audio override (no cap - long inputs risk OOM). Output is always snapped to the 17n+5 grid at 24 fps."},
                ),
                "generated_audio_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1, "tooltip": "Gain applied to the generated audio after decode, in decibels."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "sample"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Unified MiniMax H3 sampler: orchestrates conditioning, sigma shift, speed patches, scheduling, and AV decode for T2V / FL2VA / REF2VA."

    @staticmethod
    def _require_pipe_dict(pipe, pipe_name):
        if not isinstance(pipe, dict):
            raise ValueError(f"{pipe_name} is not a valid pipe object.")
        return pipe

    @classmethod
    def _unpack_models_pipe(cls, models_pipe, workflow_mode):
        pipe = cls._require_pipe_dict(models_pipe, "models_pipe")
        for key in ("vae", "audio_vae", "clip"):
            if pipe.get(key, None) is None:
                raise ValueError(f"models_pipe is missing {key}.")

        family = _active_family(workflow_mode)
        base = pipe.get(f"{family}_model", None)
        turbo_model = pipe.get(f"{family}_turbo_model", None)
        return {
            "vae": pipe["vae"],
            "audio_vae": pipe["audio_vae"],
            "clip": pipe["clip"],
            "family": family,
            "base_model": base,
            "turbo_model": turbo_model,
        }

    @staticmethod
    def _unpack_config_pipe(config_pipe):
        pipe = CRT_MiniMaxH3UnifiedSampler._require_pipe_dict(config_pipe, "config_pipe")
        return {
            "prompt": str(pipe.get("prompt", "")),
            "seed": int(pipe.get("seed", 0)),
            "first_frame": pipe.get("first_frame", None),
            "last_frame": pipe.get("last_frame", None),
            "ref_images": pipe.get("ref_images", {}) or {},
            "ref_videos": pipe.get("ref_videos", {}) or {},
            "ref_video_audios": pipe.get("ref_video_audios", {}) or {},
            "ref_audios": pipe.get("ref_audios", {}) or {},
            "override_frames": pipe.get("override_frames", None),
            "override_megapixels": pipe.get("override_megapixels", None),
        }

    @staticmethod
    def _offload_clip(clip):
        try:
            if clip is None:
                return
            target = mm.unet_offload_device()
            for obj in (
                getattr(clip, "cond_stage_model", None),
                getattr(getattr(clip, "patcher", None), "model", None),
            ):
                if obj is not None and hasattr(obj, "to"):
                    try:
                        obj.to(target)
                    except Exception:
                        pass
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"CLIP offload failed: {e}", level="warn")

    @staticmethod
    def _offload_vae(vae):
        try:
            if vae is None:
                return
            target = mm.unet_offload_device()
            obj = getattr(vae, "first_stage_model", None) or getattr(vae, "model", None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(target)
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"VAE offload failed: {e}", level="warn")

    @staticmethod
    def _reload_vae(vae):
        try:
            if vae is None:
                return
            device = mm.get_torch_device()
            obj = getattr(vae, "first_stage_model", None) or getattr(vae, "model", None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(device)
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"VAE reload failed: {e}", level="warn")

    @staticmethod
    def _unload_sampling_model(model):
        errors = []
        try:
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(mm.unet_offload_device())
        except Exception as e:
            errors.append(f"model.to(offload) failed: {e}")
        try:
            mm.unload_all_models()
        except Exception as e:
            errors.append(f"unload_all_models failed: {e}")
        try:
            mm.cleanup_models_gc()
        except Exception as e:
            errors.append(f"cleanup_models_gc failed: {e}")
        try:
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            errors.append(f"soft_empty_cache failed: {e}")
        return errors

    @classmethod
    def _apply_speed_patches(cls, model, enable_sol_attn, enable_chunk_ff, enable_spectrum):
        """Apply the optimization chain in dependency order.

        Sigma shift must already be on the model: the Sol patch reads
        model_sampling.percent_to_sigma at patch time. Sol and ChunkFF install
        object patches that adopt earlier patches as fallbacks, and Spectrum's
        sampler wrappers go on last so they run the fully patched model.
        Everything is embedded (MiniMaxUSOpt) — no external custom nodes.
        """
        return MiniMaxUSOpt.apply_us_opt(
            model,
            enable_sol=bool(enable_sol_attn),
            sol_params=dict(cls.SOL_DEFAULTS),
            enable_chunk_ff=bool(enable_chunk_ff),
            chunk_params=dict(cls.CHUNK_FF_DEFAULTS),
            enable_spectrum=bool(enable_spectrum),
            spectrum_params=dict(cls.SPECTRUM_DEFAULTS),
            log_fn=cls._log,
        )

    @staticmethod
    def _dims_from_megapixels_aspect(megapixels, aspect_ratio):
        """Official ResolutionSelector recipe: aspect * megapixels, ceiled to 32."""
        ratio_str = str(aspect_ratio).split(" ")[0]
        try:
            width_ratio, height_ratio = map(int, ratio_str.split(":"))
        except Exception:
            width_ratio, height_ratio = 16, 9

        ratio = float(width_ratio) / float(max(height_ratio, 1))
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        width = math.sqrt(total_pixels * ratio)
        height = math.sqrt(total_pixels / max(ratio, 1e-8))

        multiple = 32
        width = max(multiple, int(math.ceil(width / multiple)) * multiple)
        height = max(multiple, int(math.ceil(height / multiple)) * multiple)
        return int(width), int(height)

    @staticmethod
    def _resize_crop_cover(image, width, height):
        """Aspect-preserving cover resize, then center-crop to exactly
        width x height - never stretches."""
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
            image.movedim(-1, 1), scaled_w, scaled_h, "lanczos", "disabled"
        ).movedim(1, -1)

        x0 = max(0, (scaled_w - width) // 2)
        y0 = max(0, (scaled_h - height) // 2)
        return (
            upscaled[:, y0:y0 + height, x0:x0 + width, :].contiguous(),
        )[0]

    @staticmethod
    def _frame_canvas(image, megapixels_target):
        """Aspect-preserving resize to the closest resolution above the
        megapixel target, then center-crop the few pixels down to a 32-multiple.
        Returns (cropped_image, width, height)."""
        _, src_h, src_w, _ = image.shape
        scale = math.sqrt(
            max(1.0, float(megapixels_target) * 1_000_000)
            / float(max(1, src_w * src_h))
        )
        rw = max(32, int(math.ceil(src_w * scale)))
        rh = max(32, int(math.ceil(src_h * scale)))
        resized = comfy.utils.common_upscale(
            image.movedim(-1, 1), rw, rh, "lanczos", "disabled"
        ).movedim(1, -1)
        width = max(32, (rw // 32) * 32)
        height = max(32, (rh // 32) * 32)
        x0 = (rw - width) // 2
        y0 = (rh - height) // 2
        cropped = resized[:, y0:y0 + height, x0:x0 + width, :].contiguous()
        return cropped, width, height

    @classmethod
    def _resolve_i2v_canvas(cls, first, last, megapixels_target, fl_aspect_mode):
        """I2V canvas: the connected keyframes override the aspect widget.

        Exactly one frame -> that frame's aspect defines the canvas. Both
        frames -> fl_aspect_mode picks which one is preserved; Optimal builds a
        middle-ground canvas from both ratios. The non-preserved frame is
        cover-cropped (never stretched) onto the canvas."""
        if last is None:
            first, width, height = cls._frame_canvas(first, megapixels_target)
            cls._log(f"I2V canvas from first frame: {width}x{height}", level="ok")
            return width, height, first, None
        if first is None:
            last, width, height = cls._frame_canvas(last, megapixels_target)
            cls._log(f"I2V canvas from last frame: {width}x{height}", level="ok")
            return width, height, None, last

        _, fh, fw, _ = first.shape
        _, lh, lw, _ = last.shape

        if str(fl_aspect_mode) == "Optimal":
            ratio = math.sqrt((fw / float(fh)) * (lw / float(lh)))
            target_area = max(1.0, float(megapixels_target) * 1_000_000)
            rw = max(32, int(math.ceil(math.sqrt(target_area * ratio))))
            rh = max(32, int(math.ceil(math.sqrt(target_area / max(ratio, 1e-8)))))
            width = max(32, (rw // 32) * 32)
            height = max(32, (rh // 32) * 32)
            first = cls._resize_crop_cover(first, width, height)
            last = cls._resize_crop_cover(last, width, height)
            cls._log(f"I2V optimal F/L canvas: {width}x{height}", level="ok")
            return width, height, first, last

        if str(fl_aspect_mode) == "Preserve Last":
            base_img, width, height = cls._frame_canvas(last, megapixels_target)
            first = cls._resize_crop_cover(first, width, height)
            cls._log(f"I2V canvas from preserved last frame: {width}x{height}", level="ok")
            return width, height, first, base_img

        base_img, width, height = cls._frame_canvas(first, megapixels_target)
        last = cls._resize_crop_cover(last, width, height)
        cls._log(f"I2V canvas from preserved first frame: {width}x{height}", level="ok")
        return width, height, base_img, last

    @staticmethod
    def _length_from_frames(frames):
        return align_frame_count(max(5, int(frames)))

    @staticmethod
    def _weak(obj):
        try:
            return weakref.ref(obj)
        except TypeError:
            return None

    @classmethod
    def _get_text_conditioning_cache(cls, cache_key, clip, vae, audio_vae):
        cached = cls._TEXT_COND_CACHE
        if cached["key"] != cache_key:
            return None
        for field, obj in (("clip", clip), ("vae", vae), ("audio_vae", audio_vae)):
            ref = cached[field]
            if ref is None or ref() is not obj:
                return None
        cls._log("Conditioning cache HIT - skipping CLIP/VAE encode", level="ok")
        return cached["value"]

    @classmethod
    def _store_text_conditioning_cache(cls, cache_key, clip, vae, audio_vae, value):
        try:
            refs = {
                "clip": cls._weak(clip),
                "vae": cls._weak(vae),
                "audio_vae": cls._weak(audio_vae),
            }
            if any(r is None for r in refs.values()):
                return
            cached = cls._TEXT_COND_CACHE
            cached["key"] = cache_key
            cached.update(refs)
            cached["value"] = value
        except Exception:
            pass

    @classmethod
    def _build_conditioning_and_latent(cls, mode, clip, vae, audio_vae, prompt, config, width, height, length):
        # Tensors must not be evaluated with `or` / `bool()` (ambiguous for
        # multi-value tensors). Check `is not None` for frames, len for dicts.
        has_media = bool(
            (config.get("first_frame") is not None)
            or (config.get("last_frame") is not None)
            or bool(config.get("ref_images"))
            or bool(config.get("ref_videos"))
            or bool(config.get("ref_video_audios"))
            or bool(config.get("ref_audios"))
        )

        cache_key = None
        if not has_media:
            cache_key = (
                "minimaxh3-cond-text-v2",
                str(prompt),
                mode,
                int(width),
                int(height),
                int(length),
            )
            cached = cls._get_text_conditioning_cache(cache_key, clip, vae, audio_vae)
            if cached is not None:
                return cached

        if mode == MODE_REF2VA:
            total_refs = (
                len(config["ref_images"])
                + len(config["ref_videos"])
                + len(config["ref_video_audios"])
                + len(config["ref_audios"])
            )
            if total_refs > 12:
                cls._log(
                    f"{total_refs} reference files connected; the official Ref2VA limit is 12 mixed files - quality may degrade.",
                    level="warn",
                )
            has_refs = bool(
                config["ref_images"] or config["ref_videos"] or config["ref_audios"]
            )
            # AIToolkit ref-video treatment (always on): snap the ref's own
            # duration DOWN to 17n+5 and trim the paired audio to the same
            # real-time window. Matches ai-toolkit's minimax_h3_ref2va recipe.
            def _snap_down(n):
                return ((max(5, int(n)) - 5) // 17) * 17 + 5

            for k in list((config.get("ref_videos") or {}).keys()):
                vid = config["ref_videos"][k]
                if vid is not None and hasattr(vid, "shape"):
                    try:
                        total = int(vid.shape[0])
                        n = _snap_down(total)
                        if n < total:
                            config["ref_videos"][k] = vid[:n]
                            audio_key = k.replace("ref_video_", "ref_video_audio_")
                            if audio_key in (config.get("ref_video_audios") or {}):
                                aud = config["ref_video_audios"][audio_key]
                                if isinstance(aud, dict) and aud.get("waveform") is not None:
                                    sr = int(aud.get("sample_rate", 44100))
                                    keep = int(round(n / 24 * sr))
                                    aud["waveform"] = aud["waveform"][..., :keep]
                    except Exception:
                        pass
            # Resize REF2VA visuals to the target megapixel area, quantized to 32,
            # aspect-preserving cover resize + center-crop (never stretch).
            def _resize_ref_to_target(img_batch):
                _, h, w, _ = img_batch.shape
                ratio = float(w) / float(max(1, h))
                total_pixels = int(width) * int(height)
                tw = math.sqrt(total_pixels * ratio)
                th = math.sqrt(total_pixels / max(ratio, 1e-8))
                tw = max(32, int(round(tw / 32) * 32))
                th = max(32, int(round(th / 32) * 32))
                if tw == w and th == h:
                    return img_batch
                return cls._resize_crop_cover(img_batch, tw, th)

            ref_images_resized = {}
            for k, img in (config["ref_images"] or {}).items():
                try:
                    ref_images_resized[k] = _resize_ref_to_target(img[:1]) if img is not None else img
                except Exception:
                    ref_images_resized[k] = img
            ref_videos_resized = {}
            for k, vid in (config["ref_videos"] or {}).items():
                try:
                    ref_videos_resized[k] = _resize_ref_to_target(vid) if vid is not None else vid
                except Exception:
                    ref_videos_resized[k] = vid

            # Patch adapt_canvas for this call so ref videos also respect the
            # target MP area instead of the fixed 768 short edge. The native
            # ref-video path resizes with crop="disabled" (stretch) when the
            # aspect differs, so pre-cover-crop the frames to the patched
            # canvas aspect and hand the native node an exact-aspect input.
            import comfy_extras.nodes_minimax_h3 as _h3_nodes
            _orig_adapt = _h3_nodes.adapt_canvas
            def _patched_adapt(vw, vh):
                ratio = float(vw) / float(max(1, vh))
                total_pixels = int(width) * int(height)
                cw = math.sqrt(total_pixels * ratio)
                ch = math.sqrt(total_pixels / max(ratio, 1e-8))
                cw = max(32, int(round(cw / 32) * 32))
                ch = max(32, int(round(ch / 32) * 32))
                return int(cw), int(ch)
            _h3_nodes.adapt_canvas = _patched_adapt
            # Pre-cover-crop each ref video to its patched canvas so the
            # native resize never stretches (crop="disabled" on exact aspect
            # is a no-op).
            for k in list(ref_videos_resized.keys()):
                vid = ref_videos_resized[k]
                if vid is None or not hasattr(vid, "shape"):
                    continue
                try:
                    cw, ch = _patched_adapt(int(vid.shape[2]), int(vid.shape[1]))
                    if (int(vid.shape[2]), int(vid.shape[1])) != (cw, ch):
                        ref_videos_resized[k] = cls._resize_crop_cover(vid, cw, ch)
                except Exception:
                    pass
            try:
                outputs = cls._result_tuple(
                    MiniMaxH3ReferenceToVideo.execute(
                        clip=clip,
                        vae=vae,
                        audio_vae=audio_vae,
                        prompt=prompt,
                        width=int(width),
                        height=int(height),
                        length=int(length),
                        ref_image_size="match",
                        ref_images=ref_images_resized,
                        ref_videos=ref_videos_resized,
                        ref_video_audios=config["ref_video_audios"],
                        ref_audios=config["ref_audios"],
                    )
                )
            finally:
                _h3_nodes.adapt_canvas = _orig_adapt
            positive, latent = outputs[0], outputs[1]
            if has_refs:
                cls._log(
                    "REF2VA refs: "
                    f"{len(config['ref_images'])} image(s), {len(config['ref_videos'])} video(s), "
                    f"{len(config['ref_audios'])} standalone audio(s)",
                    level="ok",
                )
            else:
                cls._log("REF2VA without any reference media; running prompt-only.", level="warn")
        else:
            first_frame = config["first_frame"] if mode == MODE_FL2VA else None
            last_frame = config["last_frame"] if mode == MODE_FL2VA else None
            if mode == MODE_FL2VA and first_frame is None and last_frame is None:
                cls._log(
                    "FL2VA selected without keyframes; falling back to prompt-only T2V conditioning.",
                    level="warn",
                )
            outputs = cls._result_tuple(
                MiniMaxH3ImageToVideo.execute(
                    clip=clip,
                    vae=vae,
                    prompt=prompt,
                    width=int(width),
                    height=int(height),
                    length=int(length),
                    first_frame=first_frame,
                    last_frame=last_frame,
                )
            )
            positive, latent = outputs[0], outputs[1]

        if cache_key is not None:
            cls._store_text_conditioning_cache(cache_key, clip, vae, audio_vae, (positive, latent))
        return positive, latent

    def sample(
        self,
        models_pipe,
        config_pipe,
        workflow_mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        live_preview,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db=0.0,
        **_other_inputs,
    ):
        mode = str(workflow_mode)
        if mode not in WORKFLOW_MODES:
            raise ValueError(f"Unknown workflow_mode: {mode}")
        total_steps = 6

        kickoff_taeh3_download()
        wipe_all_caches()

        self._log(f"Starting mode: {mode}")
        live_preview = bool(live_preview)
        previous_preview_method = latent_preview.args.preview_method
        try:
            # Core binary previews render as a still-per-step in the modern
            # frontend; live preview is delivered by the animated-WebP override
            # wrapper instead, so the core path stays off entirely.
            latent_preview.set_preview_method("none")

            images, audio = self._sample_inner(
                models_pipe,
                config_pipe,
                mode,
                steps,
                steps_turbo,
                turbo,
                enable_sol_attn,
                enable_chunk_ff,
                enable_spectrum,
                vae_decode_tiled,
                unload_before_decode,
                low_vram,
                megapixels_target,
                aspect_ratio,
                fl_aspect_mode,
                length_frames,
                audio_frames_override,
                video_frames_override,
                generated_audio_gain_db,
                live_preview=bool(live_preview),
                unique_id=_other_inputs.get("unique_id"),
            )
        finally:
            # Restore the user's global preview method; our override must not
            # leak into other workflows after the run.
            latent_preview.args.preview_method = previous_preview_method
        return images, audio

    def _sample_inner(
        self,
        models_pipe,
        config_pipe,
        mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db=0.0,
        live_preview=False,
        unique_id=None,
    ):
        total_steps = 6

        models = self._unpack_models_pipe(models_pipe, mode)
        config = self._unpack_config_pipe(config_pipe)

        use_turbo = bool(turbo) and models["turbo_model"] is not None
        if bool(turbo) and models["turbo_model"] is None:
            if models["base_model"] is not None:
                self._log(
                    "Turbo enabled but the Turbo variant of the active family is not connected; "
                    "using the base model with full steps.",
                    level="warn",
                )
        if use_turbo:
            step_count = int(steps_turbo)
        else:
            step_count = int(steps)
        base_model = None
        if models["base_model"] is None or models["turbo_model"] is None:
            # Socket visibility: shows exactly which variants the pipe actually
            # delivered. A ✓ on the active family's missing variant means the
            # pipe served a stale cache — the IS_CHANGED presence hash now
            # busts that, so this should only appear on genuine wiring gaps.
            received = {
                k: ("✓" if (models_pipe.get(k) if isinstance(models_pipe, dict) else None) is not None else "✗")
                for k in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
            }
            self._log(
                "Models Pipe sockets received: "
                + " ".join(f"{k}={v}" for k, v in received.items()),
                level="warn",
            )
        if models["base_model"] is None and models["turbo_model"] is None:
            # Last resort: try any model wired to the pipe, even from the
            # other family, so a graph with only FL2VA wired can still run
            # an R2V request instead of hard-crashing.
            for fallback_key in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model"):
                fb = models_pipe.get(fallback_key) if isinstance(models_pipe, dict) else None
                if fb is not None:
                    self._log(
                        f"Required {models['family']} model missing; falling back to {fallback_key} (may be suboptimal for {mode}).",
                        level="warn",
                    )
                    base_model = fb
                    if fallback_key.endswith("_turbo_model") and not use_turbo:
                        step_count = int(steps)
                    break
            if base_model is None:
                raise ValueError(
                    f"models_pipe has no '{models['family']}_model' (or '{models['family']}_turbo_model') "
                    f"connected, required by {mode}. Connect the matching AutoDL model loader output to "
                    "that socket on the MiniMax H3 US Models Pipe (CRT)."
                )
        else:
            base_model = models["turbo_model"] if use_turbo else models["base_model"]
        if base_model is None and not use_turbo and models["turbo_model"] is not None:
            self._log(
                "Turbo is OFF but the base model is not connected; falling back to the Turbo model with full steps.",
                level="warn",
            )
            base_model = models["turbo_model"]
        elif base_model is None and use_turbo and models["turbo_model"] is None and models["base_model"] is not None:
            self._log(
                "Turbo is ON but the Turbo model is not connected; falling back to the base model.",
                level="warn",
            )
            base_model = models["base_model"]
        if base_model is None:
            raise ValueError(
                f"Selected model variant for {mode} is not connected (turbo={bool(turbo)}). "
                f"Connect '{models['family']}_{'turbo_' if use_turbo else ''}model' on the Pipe or toggle Turbo."
            )
        self._log(
            f"Family: {models['family']} | variant: {'Turbo LoRA' if use_turbo else 'base'} | steps: {step_count}",
            level="ok",
        )

        self._progress(1, total_steps, "Applying sigma shift and speed patches")
        model = _apply_sigma_shift(base_model, SHIFT_VIDEO, SHIFT_AUDIO)
        model = self._apply_speed_patches(model, enable_sol_attn, enable_chunk_ff, enable_spectrum)
        if live_preview:
            model = apply_h3_preview_override(model, unique_id)

        # Per-token prompt weighting: (word:1.5) emphasize, (word:-1) remove.
        # The tokenizer runs with disable_weights=True so the syntax must be
        # stripped from the prompt before conditioning and applied as an
        # attention patch instead.
        weight_pairs, clean_prompt = _h3_parse_prompt_weights(models["clip"], config["prompt"], self._log)
        if weight_pairs:
            config["prompt"] = clean_prompt
            to = model.model_options.get("transformer_options", {}).copy()
            to["minimax_h3_token_weights"] = weight_pairs
            model.model_options["transformer_options"] = to
            dm = model.get_model_object("diffusion_model")
            patch = _H3WeightPatch()
            for idx, block in enumerate(dm.blocks):
                model.add_object_patch(
                    f"diffusion_model.blocks.{idx}.attn.forward",
                    patch.__get__(block.attn, block.attn.__class__),
                )
            self._log(f"Prompt weighting active on {len(weight_pairs)} token(s).", level="ok")

        override_megapixels = config["override_megapixels"]
        if override_megapixels is not None:
            megapixels_target = float(override_megapixels)
            self._log(f"Overriding megapixels_target from US Config -> {megapixels_target}", level="ok")
        override_frames = config["override_frames"]
        if override_frames is not None:
            length_frames = int(override_frames)
            self._log(f"Overriding frame count from US Config -> {length_frames}", level="ok")

        # REF2VA frame-count overrides: longest reference video wins over the
        # longest reference audio; both snap to the 17n+5 grid afterwards.
        if mode == MODE_REF2VA:
            for socket, video_frames in sorted(config["ref_videos"].items()):
                if len(video_frames.shape) != 4:
                    raise ValueError(
                        f"'{socket} (REF2VA)' must be an IMAGE batch of video frames."
                    )
                if int(video_frames.shape[0]) < 5:
                    raise ValueError(
                        f"'{socket} (REF2VA)' needs at least 5 frames (~0.2s at 24 fps); got {int(video_frames.shape[0])}."
                    )
            for socket, ref_audio in sorted(config["ref_audios"].items()):
                waveform = ref_audio.get("waveform", None) if isinstance(ref_audio, dict) else None
                if waveform is None or not isinstance(waveform, torch.Tensor):
                    raise ValueError(f"'{socket} (REF2VA)' carries no audio waveform.")

        if mode == MODE_REF2VA and override_frames is None:
            if bool(video_frames_override) and config["ref_videos"]:
                longest_video = max(
                    int(v.shape[0]) for v in config["ref_videos"].values()
                )
                length_frames = max(5, longest_video)
                self._log(
                    f"Ref video length overrides frame count -> {length_frames} frames",
                    level="ok",
                )
            elif bool(audio_frames_override) and config["ref_audios"]:
                longest_audio_frames = 0
                for ref_audio in config["ref_audios"].values():
                    waveform = ref_audio.get("waveform", None) if isinstance(ref_audio, dict) else None
                    sample_rate = ref_audio.get("sample_rate", None) if isinstance(ref_audio, dict) else None
                    if waveform is None or not sample_rate:
                        continue
                    seconds = float(waveform.shape[-1]) / float(sample_rate)
                    longest_audio_frames = max(
                        longest_audio_frames, int(round(seconds * FPS))
                    )
                if longest_audio_frames > 0:
                    length_frames = max(5, longest_audio_frames)
                    self._log(
                        f"Ref audio length overrides frame count -> {length_frames} frames",
                        level="ok",
                    )

        if mode == MODE_FL2VA and (
            config["first_frame"] is not None or config["last_frame"] is not None
        ):
            width, height, config["first_frame"], config["last_frame"] = (
                self._resolve_i2v_canvas(
                    config["first_frame"],
                    config["last_frame"],
                    megapixels_target,
                    fl_aspect_mode,
                )
            )
        else:
            width, height = self._dims_from_megapixels_aspect(megapixels_target, aspect_ratio)
        length = self._length_from_frames(length_frames)
        if length > 362:
            self._log(
                f"{length} frames exceeds the trained range (362); generation quality may degrade.",
                level="warn",
            )
        self._log(f"Canvas: {width}x{height} | length: {length} frames @ {FPS:.0f} fps", level="ok")

        if mode != MODE_FL2VA and (config["first_frame"] is not None or config["last_frame"] is not None):
            self._log("First/Last Frame inputs are I2V-only; ignoring them in this mode.", level="warn")
        if mode != MODE_REF2VA and (
            config["ref_images"] or config["ref_videos"] or config["ref_video_audios"] or config["ref_audios"]
        ):
            self._log("Ref * inputs are R2V-only; ignoring them in this mode.", level="warn")

        self._progress(2, total_steps, "Building conditioning and AV latent")
        positive, latent = self._build_conditioning_and_latent(
            mode,
            models["clip"],
            models["vae"],
            models["audio_vae"],
            config["prompt"],
            config,
            width,
            height,
            length,
        )

        if low_vram:
            self._log("Low VRAM: unloading CLIP after conditioning", level="ok")
            self._offload_clip(models["clip"])

        noise_obj = self._result_tuple(RandomNoise.execute(config["seed"]))[0]
        sampler_name = "euler" if use_turbo else SAMPLER_NAME
        sampler_obj = self._result_tuple(KSamplerSelect.execute(sampler_name))[0]
        sigmas_obj = self._result_tuple(
            BasicScheduler.execute(model, SCHEDULER_NAME, step_count, 1.0)
        )[0]
        guider_obj = self._result_tuple(BasicGuider.execute(model, positive))[0]
        # Wrap guider so the previewer sees correct latent shapes / fps for video sweep
        if live_preview:
            try:
                guider_obj = _PreviewFixGuider(guider_obj, fps_override=FPS)
                _PREVIEW_STATE.fps_override = float(FPS)
            except Exception:
                pass

        if low_vram:
            self._log("Low VRAM: unloading VAEs before sampling", level="ok")
            self._offload_vae(models["vae"])
            self._offload_vae(models["audio_vae"])
            mm.soft_empty_cache()

        self._progress(3, total_steps, f"Sampling ({sampler_name}/{SCHEDULER_NAME}, {step_count} steps)")
        sampled = self._result_tuple(
            SamplerCustomAdvanced.execute(noise_obj, guider_obj, sampler_obj, sigmas_obj, latent)
        )
        output_latent = sampled[0] if len(sampled) > 0 else latent

        samples = output_latent.get("samples") if isinstance(output_latent, dict) else None
        if samples is None:
            raise RuntimeError("Sampler returned no packed latent output.")
        streams = samples.unbind() if getattr(samples, "is_nested", False) else (samples,)
        video_stream = streams[0]
        audio_stream = streams[-1] if len(streams) > 1 else None

        if unload_before_decode:
            self._log("Unload-before-decode enabled: unloading diffusion model", level="ok")
            unload_errors = self._unload_sampling_model(model)
            if unload_errors:
                self._log("Unload-before-decode warnings: " + " | ".join(unload_errors), level="warn")

        # unload_before_decode evicts every loaded model (VAEs included) via
        # unload_all_models, so the VAEs must be pulled back whenever that ran,
        # not only in the low_vram flow.
        if low_vram or unload_before_decode:
            self._log("Reloading VAEs for decode", level="ok")
            self._reload_vae(models["vae"])
            self._reload_vae(models["audio_vae"])

        self._progress(4, total_steps, "Decoding video/audio")
        video_latent = {"samples": video_stream}
        if vae_decode_tiled:
            images = nodes.VAEDecodeTiled().decode(
                models["vae"],
                video_latent,
                512,
                64,
                64,
                8,
            )[0]
        else:
            images = nodes.VAEDecode().decode(models["vae"], video_latent)[0]

        audio = None
        if audio_stream is not None:
            audio = vae_decode_audio(models["audio_vae"], {"samples": audio_stream})
            gain = 10.0 ** (float(generated_audio_gain_db) / 20.0)
            waveform = audio.get("waveform", None)
            if waveform is not None:
                audio = dict(audio)
                audio["waveform"] = waveform * gain

        self._progress(total_steps, total_steps, f"{mode} complete")
        if audio is None:
            raise RuntimeError("Packed latent contained no audio stream to decode.")
        return (images, audio)

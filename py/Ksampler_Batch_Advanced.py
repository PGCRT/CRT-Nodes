import torch
import threading
import numpy as np
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
import node_helpers
import os
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo


# ── Reference latent helper ───────────────────────────────────────────────────

def _attach_reference_latent(conditioning, ref_samples):
    if conditioning is None or isinstance(conditioning, str):
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [ref_samples]},
        append=False,
    )


# ── Sigma helpers ─────────────────────────────────────────────────────────────

def _split_sigmas_at_injection(sigmas, injection_point):
    """Split sigmas into two stages at the injection_point ratio (0..1)."""
    steps = max(sigmas.shape[-1] - 1, 0)
    total_steps = round(steps * injection_point)
    if total_steps <= 0:
        return sigmas[:1], sigmas
    high = sigmas[:-(total_steps)]
    low  = sigmas[-(total_steps + 1):]
    return high, low


def _multiply_sigmas(sigmas, factor, start, end):
    out = sigmas.clone()
    total = len(out)
    start_idx = int(start * total)
    end_idx   = int(end   * total)
    for i in range(start_idx, end_idx):
        out[i] *= factor
    return out


# ── Detail Daemon helpers ─────────────────────────────────────────────────────

def _make_detail_daemon_schedule(
    steps, start, end, bias, amount, exponent,
    start_offset, end_offset, fade, smooth,
):
    start = min(start, end)
    mid   = start + bias * (end - start)
    multipliers = np.zeros(steps)
    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values ** exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values ** exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx  : end_idx  + 1] = end_values
    multipliers[:start_idx]    = start_offset
    multipliers[end_idx + 1:]  = end_offset
    multipliers *= 1 - fade
    return multipliers


def _get_dd_schedule_value(sigma, sigmas, dd_schedule):
    sched_len = len(dd_schedule)
    if (
        sched_len < 1
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    if sched_len == 1:
        return dd_schedule[0].item()

    deltas = (sigmas[:-1] - sigma).abs()
    idx    = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        return dd_schedule[idx].item()

    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        return dd_schedule[idxlow]
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def _detail_daemon_sampler(
    model, x, sigmas,
    *, dds_wrapped_sampler, dds_make_schedule, dds_cfg_scale_override,
    **kwargs,
):
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0

    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max  = float(sigmas_cpu[0])
    sigma_min  = float(sigmas_cpu[-1]) + 1e-5

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = _get_dd_schedule_value(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-6, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in ("inner_model", "sigmas"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))

    return dds_wrapped_sampler.sampler_function(
        model_wrapper, x, sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


def _build_dd_sampler(
    base_sampler, amount, start, end, bias,
    exponent, start_offset, end_offset, fade, smooth, cfg_scale_override,
):
    def dds_make_schedule(steps):
        return _make_detail_daemon_schedule(
            steps, start, end, bias, amount, exponent,
            start_offset, end_offset, fade, smooth,
        )
    return comfy.samplers.KSAMPLER(
        _detail_daemon_sampler,
        extra_options={
            "dds_wrapped_sampler": base_sampler,
            "dds_make_schedule":   dds_make_schedule,
            "dds_cfg_scale_override": cfg_scale_override,
        },
    )


# ── Node ──────────────────────────────────────────────────────────────────────

class CRT_KSamplerBatchAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Core (same as KSampler Batch) ─────────────────────────────
                "model":          ("MODEL",),
                "vae":            ("VAE",),
                "positive":       ("CONDITIONING",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "increment_seed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Each batch item gets seed+1, seed+2, etc. Disable to use same seed for all.",
                    },
                ),
                "steps":   ("INT",   {"default": 8,   "min": 1,   "max": 10000}),
                "cfg":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["Batch (Parallel)", "Sequential"], {"default": "Batch (Parallel)"}),
                "image_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Target resolution in megapixels. Images are resized (lanczos) preserving aspect ratio.",
                    },
                ),
                "edit_model_flux2klein": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable ReferenceLatent-style conditioning for edit/fill models (flux2klein).",
                    },
                ),
                "reference_mode": (
                    ["per_item", "shared"],
                    {
                        "default": "per_item",
                        "tooltip": "per_item: each image is its own reference. shared: image[0] is reference for all.",
                    },
                ),
                "enable_vae_decode": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode latents to images using the VAE."},
                ),
                "create_comparison_grid": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Create a horizontal side-by-side comparison grid."},
                ),
                "save_images":          ("BOOLEAN", {"default": True}),
                "save_folder_path":     ("STRING",  {"default": ".\\ComfyUI\\output"}),
                "save_subfolder_name":  ("STRING",  {"default": "KSAMPLER_BATCH_ADV"}),
                "save_filename_prefix": ("STRING",  {"default": "output"}),
                # ── Sigma settings ────────────────────────────────────────────
                "stage1_sigma_factor": (
                    "FLOAT",
                    {"default": 1.005, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage2_sigma_factor": (
                    "FLOAT",
                    {"default": 0.995, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage1_sigma_start": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage1_sigma_end": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage2_sigma_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage2_sigma_end": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                # ── Detail Daemon ─────────────────────────────────────────────
                "details_amount_stage1": (
                    "FLOAT",
                    {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "details_amount_stage2": (
                    "FLOAT",
                    {"default": 0.15, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                # ── Noise Injection ───────────────────────────────────────────
                "enable_noise_injection": (
                    ["disable", "enable"],
                    {"default": "enable"},
                ),
                "injection_point": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "injection_seed_offset": (
                    "INT",
                    {"default": 1, "min": -100, "max": 100, "step": 1},
                ),
                "injection_strength": (
                    "FLOAT",
                    {"default": 0.10, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "normalize_injected_noise": (
                    ["enable", "disable"],
                    {"default": "enable"},
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "Starting latent. If image is also connected, image takes priority. "
                                   "If neither is connected, a dummy empty latent is auto-generated.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image batch. Resized to image_megapixels before encoding. "
                                   "In edit_model_flux2klein mode: used as reference latent.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES  = ("images", "comparison_grid", "latents")
    FUNCTION      = "sample_batch"
    CATEGORY      = "CRT/Sampling"

    # ── Helpers (identical to KSampler Batch) ─────────────────────────────────

    def _resize_to_megapixels(self, images, megapixels, quantize_to):
        B, H, W, C = images.shape
        target_pixels = megapixels * 1_000_000
        scale  = (target_pixels / (H * W)) ** 0.5
        new_H  = max(quantize_to, round(H * scale / quantize_to) * quantize_to)
        new_W  = max(quantize_to, round(W * scale / quantize_to) * quantize_to)
        if new_H == H and new_W == W:
            return images
        resized = comfy.utils.common_upscale(
            images.movedim(-1, 1), new_W, new_H, "lanczos", "disabled"
        )
        return resized.movedim(1, -1)

    def _fix_conditioning(self, conditioning, target_batch_size):
        if not conditioning:
            return conditioning
        fixed = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = cond_dict.copy()
            B = cond_tensor.shape[0]
            if B != target_batch_size:
                idx = [i % B for i in range(target_batch_size)]
                cond_tensor = cond_tensor[idx]
            if "pooled_output" not in new_dict or new_dict["pooled_output"] is None:
                hidden = cond_tensor.shape[-1]
                pooled = torch.zeros(target_batch_size, hidden, device=cond_tensor.device, dtype=cond_tensor.dtype)
            else:
                pooled = new_dict["pooled_output"]
                if pooled.shape[0] != target_batch_size:
                    Bp    = pooled.shape[0]
                    idx_p = [i % Bp for i in range(target_batch_size)]
                    pooled = pooled[idx_p]
            new_dict["pooled_output"] = pooled
            fixed.append([cond_tensor, new_dict])
        return fixed

    def _save_image(self, image_tensor, folder_path, subfolder_name, filename_prefix, seed, prompt=None, extra_pnginfo=None):
        try:
            cleaned_subfolder = subfolder_name.strip().lstrip("/\\")
            final_dir = os.path.join(folder_path, cleaned_subfolder)
            os.makedirs(final_dir, exist_ok=True)
            cleaned_prefix = filename_prefix.strip().lstrip("/\\")
            base  = f"{cleaned_prefix}_{seed}"
            path  = os.path.join(final_dir, f"{base}.png")
            counter = 1
            while os.path.exists(path):
                path = os.path.join(final_dir, f"{base}_{counter}.png")
                counter += 1
            pil_image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            meta = PngInfo()
            if prompt is not None:
                meta.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                meta.add_text("workflow", json.dumps(extra_pnginfo["workflow"]))
            pil_image.save(path, "PNG", pnginfo=meta)
            print(f"[CRT KSampler Batch Adv] Saved: {os.path.basename(path)}")
        except Exception as e:
            print(f"[CRT KSampler Batch Adv] Failed to save image (seed {seed}): {e}")

    # ── Noise injection helper ─────────────────────────────────────────────────

    def _inject_noise(self, latent_tensor, seed, strength, normalize):
        """Return a new tensor with scaled random noise blended in."""
        out = latent_tensor.clone()
        torch.manual_seed(seed)
        new_noise = torch.randn_like(out)
        if normalize == "enable":
            orig_std  = out.std().item()
            orig_mean = out.mean().item()
            if orig_std > 1e-6:
                new_noise = new_noise * orig_std + orig_mean
        return out + new_noise * strength

    # ── Main ──────────────────────────────────────────────────────────────────

    def sample_batch(
        self,
        model,
        vae,
        positive,
        seed,
        increment_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode,
        image_megapixels,
        edit_model_flux2klein,
        reference_mode,
        enable_vae_decode,
        create_comparison_grid,
        save_images,
        save_folder_path,
        save_subfolder_name,
        save_filename_prefix,
        stage1_sigma_factor,
        stage2_sigma_factor,
        stage1_sigma_start,
        stage1_sigma_end,
        stage2_sigma_start,
        stage2_sigma_end,
        details_amount_stage1,
        details_amount_stage2,
        enable_noise_injection,
        injection_point,
        injection_seed_offset,
        injection_strength,
        normalize_injected_noise,
        latent_image=None,
        image=None,
        negative=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # Detail Daemon parameters — fixed defaults, not exposed in UI
        dd_start, dd_end, dd_bias = 0.0, 1.0, 0.5
        dd_exponent               = 1.0
        dd_start_offset           = 0.0
        dd_end_offset             = 0.0
        dd_fade_stage1            = 0.0
        dd_fade_stage2            = 0.5
        dd_smooth                 = True
        dd_cfg_scale_override     = 1.0

        # ── VAE geometry ──────────────────────────────────────────────────────
        vae_channels  = getattr(vae, "latent_channels", 4)
        vae_downscale = getattr(vae, "downscale_ratio",  8)

        # ── Resolve samples & reference (identical to KSampler Batch) ────────
        ref_latents = None
        samples     = None
        B_cond      = positive[0][0].shape[0] if positive else 1

        if image is not None:
            imgs = self._resize_to_megapixels(image, image_megapixels, vae_downscale)
            B_img, H, W, _ = imgs.shape
            latH, latW     = H // vae_downscale, W // vae_downscale
            target_batch   = max(B_img, B_cond)
            comfy.model_management.load_model_gpu(vae.patcher)

            if edit_model_flux2klein:
                ref_latents = vae.encode(imgs[:, :, :, :3])
                samples = torch.zeros(
                    [target_batch, vae_channels, latH, latW],
                    device=comfy.model_management.intermediate_device(),
                )
                samples = comfy.sample.fix_empty_latent_channels(model, samples)
                denoise = 1.0
                print("[CRT KSampler Batch Adv] edit_model_flux2klein: denoise forced to 1.0")
            else:
                samples = vae.encode(imgs[:, :, :, :3])

        elif latent_image is not None:
            samples      = latent_image["samples"]
            samples      = comfy.sample.fix_empty_latent_channels(model, samples)
            target_batch = max(samples.shape[0], B_cond)

        else:
            print(
                "[CRT KSampler Batch Adv] WARNING: no image or latent_image connected — "
                "generating dummy 64x64 empty latent"
            )
            target_batch = B_cond
            dummy_side   = max(1, 64 // vae_downscale)
            samples = torch.zeros(
                [target_batch, vae_channels, dummy_side, dummy_side],
                device=comfy.model_management.intermediate_device(),
            )
            samples = comfy.sample.fix_empty_latent_channels(model, samples)

        # Cycle to target_batch
        if samples.shape[0] != target_batch:
            idx     = [i % samples.shape[0] for i in range(target_batch)]
            samples = samples[idx]
        if ref_latents is not None and ref_latents.shape[0] != target_batch:
            idx_r       = [i % ref_latents.shape[0] for i in range(target_batch)]
            ref_latents = ref_latents[idx_r]

        seeds = [seed + i if increment_seed else seed for i in range(target_batch)]

        # ── Conditioning ──────────────────────────────────────────────────────
        positive = self._fix_conditioning(positive, target_batch)
        if negative:
            negative = self._fix_conditioning(negative, target_batch)
        else:
            ref         = positive[0][0]
            zero_cond   = torch.zeros_like(ref)
            zero_pooled = torch.zeros(target_batch, ref.shape[-1], device=ref.device, dtype=ref.dtype)
            negative    = [[zero_cond, {"pooled_output": zero_pooled}]]

        # ── Build sigma schedules ─────────────────────────────────────────────
        ksampler_plan = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sigmas_full = ksampler_plan.sigmas.clone()

        stage1_sigmas, stage2_sigmas = _split_sigmas_at_injection(sigmas_full, injection_point)
        stage1_sigmas = _multiply_sigmas(stage1_sigmas, stage1_sigma_factor, stage1_sigma_start, stage1_sigma_end)
        stage2_sigmas = _multiply_sigmas(stage2_sigmas, stage2_sigma_factor, stage2_sigma_start, stage2_sigma_end)
        # Used when noise injection is disabled (single-pass with stage1 sigma shaping)
        full_sigmas   = _multiply_sigmas(sigmas_full, stage1_sigma_factor, stage1_sigma_start, stage1_sigma_end)

        # ── Build Detail Daemon samplers ──────────────────────────────────────
        base_sampler   = comfy.samplers.sampler_object(sampler_name)
        sampler_stage1 = _build_dd_sampler(
            base_sampler, details_amount_stage1,
            dd_start, dd_end, dd_bias, dd_exponent,
            dd_start_offset, dd_end_offset, dd_fade_stage1, dd_smooth, dd_cfg_scale_override,
        )
        sampler_stage2 = _build_dd_sampler(
            base_sampler, details_amount_stage2,
            dd_start, dd_end, dd_bias, dd_exponent,
            dd_start_offset, dd_end_offset, dd_fade_stage2, dd_smooth, dd_cfg_scale_override,
        )

        # ── Reference conditioning for parallel edit mode ─────────────────────
        pos_for_parallel = positive
        neg_for_parallel = negative
        if edit_model_flux2klein and ref_latents is not None and mode == "Batch (Parallel)":
            ref_pos          = ref_latents[0:1] if reference_mode == "shared" else ref_latents
            ref_neg          = torch.zeros_like(ref_pos)
            pos_for_parallel = _attach_reference_latent(positive, ref_pos)
            neg_for_parallel = _attach_reference_latent(negative, ref_neg)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ── Sampling ──────────────────────────────────────────────────────────
        if mode == "Sequential":
            latents_out = []
            for i in range(target_batch):
                pos_i = [[positive[j][0][i:i+1], positive[j][1]] for j in range(len(positive))]
                neg_i = [[negative[j][0][i:i+1], negative[j][1]] for j in range(len(negative))]

                if edit_model_flux2klein and ref_latents is not None:
                    ref_i = ref_latents[i:i+1] if reference_mode == "per_item" else ref_latents[0:1]
                    pos_i = _attach_reference_latent(pos_i, ref_i)
                    neg_i = _attach_reference_latent(neg_i, torch.zeros_like(ref_i))

                lat_i    = samples[i:i+1]
                cur_seed = seeds[i]

                if enable_noise_injection == "enable":
                    cb1 = latent_preview.prepare_callback(model, max(0, stage1_sigmas.shape[-1] - 1))
                    cb2 = latent_preview.prepare_callback(model, max(0, stage2_sigmas.shape[-1] - 1))

                    noise_i = comfy.sample.prepare_noise(lat_i, cur_seed, None)
                    out1 = comfy.sample.sample_custom(
                        model, noise_i, cfg, sampler_stage1, stage1_sigmas,
                        pos_i, neg_i, lat_i,
                        noise_mask=None, callback=cb1,
                        disable_pbar=disable_pbar, seed=cur_seed,
                    )
                    injected = self._inject_noise(
                        out1, cur_seed + injection_seed_offset,
                        injection_strength, normalize_injected_noise,
                    )
                    zeros_noise = torch.zeros_like(injected)
                    out2 = comfy.sample.sample_custom(
                        model, zeros_noise, cfg, sampler_stage2, stage2_sigmas,
                        pos_i, neg_i, injected,
                        noise_mask=None, callback=cb2,
                        disable_pbar=disable_pbar, seed=cur_seed,
                    )
                    latents_out.append(out2)

                else:
                    cb_full = latent_preview.prepare_callback(model, max(0, full_sigmas.shape[-1] - 1))
                    noise_i = comfy.sample.prepare_noise(lat_i, cur_seed, None)
                    out = comfy.sample.sample_custom(
                        model, noise_i, cfg, sampler_stage1, full_sigmas,
                        pos_i, neg_i, lat_i,
                        noise_mask=None, callback=cb_full,
                        disable_pbar=disable_pbar, seed=cur_seed,
                    )
                    latents_out.append(out)

            samples_out = torch.cat(latents_out, dim=0)

        else:
            # Parallel: one batched GPU call per stage + background carousel preview
            noise = torch.cat(
                [comfy.sample.prepare_noise(samples[i:i+1], seeds[i], None) for i in range(target_batch)],
                dim=0,
            )

            try:
                previewer = latent_preview.get_previewer(
                    comfy.model_management.get_torch_device(),
                    model.model.latent_format,
                )
            except Exception:
                previewer = None

            pbar   = comfy.utils.ProgressBar(steps)
            _state = {"previews": [None] * target_batch, "step": 0, "total": steps}
            _lock  = threading.Lock()
            _stop  = threading.Event()

            def _carousel():
                idx = 0
                while not _stop.is_set():
                    with _lock:
                        preview = _state["previews"][idx % target_batch]
                        step    = _state["step"]
                        total   = _state["total"]
                    if preview is not None:
                        try:
                            pbar.update_absolute(step, total, preview)
                        except Exception:
                            pass
                    idx = (idx + 1) % target_batch
                    _stop.wait(0.3)

            _thread = threading.Thread(target=_carousel, daemon=True)
            _thread.start()

            def batch_callback(step, x0, x, total_steps):
                if previewer and x0 is not None:
                    n       = min(target_batch, x0.shape[0])
                    decoded = []
                    for k in range(n):
                        try:
                            decoded.append(previewer.decode_latent_to_preview_image("JPEG", x0[k:k+1]))
                        except Exception:
                            decoded.append(None)
                    with _lock:
                        _state["previews"][:n] = decoded
                        _state["step"]  = step + 1
                        _state["total"] = total_steps
                else:
                    pbar.update_absolute(step + 1, total_steps, None)

            if enable_noise_injection == "enable":
                out1 = comfy.sample.sample_custom(
                    model, noise, cfg, sampler_stage1, stage1_sigmas,
                    pos_for_parallel, neg_for_parallel, samples,
                    noise_mask=None, callback=batch_callback,
                    disable_pbar=disable_pbar, seed=seeds[0],
                )

                # Inject per-item noise into the stage-1 output
                injected = out1.clone()
                for i in range(target_batch):
                    injected[i:i+1] = self._inject_noise(
                        injected[i:i+1],
                        seeds[i] + injection_seed_offset,
                        injection_strength,
                        normalize_injected_noise,
                    )

                zeros_noise = torch.zeros_like(injected)
                cb_s2 = latent_preview.prepare_callback(model, max(0, stage2_sigmas.shape[-1] - 1))
                samples_out = comfy.sample.sample_custom(
                    model, zeros_noise, cfg, sampler_stage2, stage2_sigmas,
                    pos_for_parallel, neg_for_parallel, injected,
                    noise_mask=None, callback=cb_s2,
                    disable_pbar=disable_pbar, seed=seeds[0],
                )

            else:
                samples_out = comfy.sample.sample_custom(
                    model, noise, cfg, sampler_stage1, full_sigmas,
                    pos_for_parallel, neg_for_parallel, samples,
                    noise_mask=None, callback=batch_callback,
                    disable_pbar=disable_pbar, seed=seeds[0],
                )

            _stop.set()
            _thread.join(timeout=1.0)

        # ── VAE Decode ────────────────────────────────────────────────────────
        images_out = None
        grid_out   = None

        if enable_vae_decode:
            decoded    = vae.decode(samples_out)
            images_out = decoded

            if save_images:
                for i in range(target_batch):
                    self._save_image(
                        decoded[i], save_folder_path, save_subfolder_name,
                        save_filename_prefix, seeds[i], prompt, extra_pnginfo,
                    )

            if create_comparison_grid and target_batch > 1:
                grid_out = torch.cat(list(decoded.unbind(0)), dim=1).unsqueeze(0)

        result = {"samples": samples_out}
        if latent_image is not None and "noise_mask" in latent_image:
            result["noise_mask"] = latent_image["noise_mask"]

        return (images_out, grid_out, result)


NODE_CLASS_MAPPINGS      = {"CRT_KSamplerBatchAdvanced": CRT_KSamplerBatchAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"CRT_KSamplerBatchAdvanced": "KSampler Batch Advanced (CRT)"}

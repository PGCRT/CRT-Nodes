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


def _attach_reference_latent(conditioning, ref_samples):
    if conditioning is None or isinstance(conditioning, str):
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [ref_samples]},
        append=False,
    )


class CRT_KSamplerBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "increment_seed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, each batch item gets seed+1, seed+2, etc. If disabled, all items use the same seed.",
                    },
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["Batch (Parallel)", "Sequential"], {"default": "Batch (Parallel)"}),
                "image_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Target resolution in megapixels. Images are resized (lanczos) preserving aspect ratio and quantized to VAE boundaries before encoding.",
                    },
                ),
                "edit_model_flux2klein": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable ReferenceLatent-style conditioning for edit/fill models (flux2klein). Forces denoise=1 and samples from an empty latent. Requires image input.",
                    },
                ),
                "reference_mode": (
                    ["per_item", "shared"],
                    {
                        "default": "per_item",
                        "tooltip": "per_item: each image in the batch is used as its own reference latent. shared: image[0] is used as the reference for every batch item.",
                    },
                ),
                "enable_vae_decode": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode latents to images using the VAE."},
                ),
                "create_comparison_grid": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Create a horizontal side-by-side comparison grid of all batch outputs."},
                ),
                "save_images": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save output images to disk."},
                ),
                "save_folder_path": (
                    "STRING",
                    {"default": ".\\ComfyUI\\output", "tooltip": "Root folder where outputs will be saved."},
                ),
                "save_subfolder_name": (
                    "STRING",
                    {"default": "KSAMPLER_BATCH", "tooltip": "Subfolder inside the root folder for organizing outputs."},
                ),
                "save_filename_prefix": (
                    "STRING",
                    {"default": "output", "tooltip": "Prefix for output filenames. Seed is appended automatically."},
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "Starting latent. If image is also connected, image takes priority. "
                                   "If neither input is connected, a dummy empty latent is auto-generated using VAE channel info.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image batch. Resized to image_megapixels before encoding. "
                                   "In normal mode: VAE-encoded as the starting latent. "
                                   "In edit_model_flux2klein mode: used as reference latent; sampling starts from an empty latent.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("images", "comparison_grid", "latents")
    FUNCTION = "sample_batch"
    CATEGORY = "CRT/Sampling"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resize_to_megapixels(self, images, megapixels, quantize_to):
        """Resize [B, H, W, C] to target megapixels preserving AR, quantized to quantize_to."""
        B, H, W, C = images.shape
        target_pixels = megapixels * 1_000_000
        scale = (target_pixels / (H * W)) ** 0.5
        new_H = max(quantize_to, round(H * scale / quantize_to) * quantize_to)
        new_W = max(quantize_to, round(W * scale / quantize_to) * quantize_to)
        if new_H == H and new_W == W:
            return images
        # common_upscale expects [B, C, H, W]
        resized = comfy.utils.common_upscale(
            images.movedim(-1, 1), new_W, new_H, "lanczos", "disabled"
        )
        return resized.movedim(1, -1)

    def _fix_conditioning(self, conditioning, target_batch_size):
        """Cycle-broadcast every conditioning entry to target_batch_size using modulo indexing."""
        if not conditioning:
            return conditioning
        fixed = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = cond_dict.copy()

            # Cycle main conditioning tensor
            B = cond_tensor.shape[0]
            if B != target_batch_size:
                idx = [i % B for i in range(target_batch_size)]
                cond_tensor = cond_tensor[idx]

            # Cycle / create pooled_output
            if "pooled_output" not in new_dict or new_dict["pooled_output"] is None:
                hidden = cond_tensor.shape[-1]
                pooled = torch.zeros(target_batch_size, hidden, device=cond_tensor.device, dtype=cond_tensor.dtype)
            else:
                pooled = new_dict["pooled_output"]
                if pooled.shape[0] != target_batch_size:
                    Bp = pooled.shape[0]
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
            base = f"{cleaned_prefix}_{seed}"
            path = os.path.join(final_dir, f"{base}.png")
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
            print(f"[CRT KSampler Batch] Saved: {os.path.basename(path)}")
        except Exception as e:
            print(f"[CRT KSampler Batch] Failed to save image (seed {seed}): {e}")

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
        latent_image=None,
        image=None,
        negative=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if model is None:
            raise RuntimeError(
                "[CRT KSampler Batch] MODEL input is None. Connect a valid diffusion MODEL node "
                "(for ERNIE_Turbo use CRTAutoDLErnieTurboModel), not a CLIP/text encoder node."
            )

        # VAE geometry — used to create correctly-shaped latents
        vae_channels  = getattr(vae, "latent_channels", 4)
        vae_downscale = getattr(vae, "downscale_ratio", 8)

        # ── Resolve samples & reference ───────────────────────────────────────
        ref_latents = None   # [B, C, latH, latW] — only set in edit mode
        samples     = None   # [B, C, latH, latW] — starting latent for sampling

        # target_batch = max(input_B, cond_B) — both cycle to fill the other
        B_cond = positive[0][0].shape[0] if positive else 1

        if image is not None:
            imgs = self._resize_to_megapixels(image, image_megapixels, vae_downscale)
            B_img, H, W, _ = imgs.shape
            latH, latW = H // vae_downscale, W // vae_downscale
            target_batch = max(B_img, B_cond)

            comfy.model_management.load_model_gpu(vae.patcher)

            if edit_model_flux2klein:
                # Encode images as reference latents, then cycle to target_batch
                ref_latents = vae.encode(imgs[:, :, :, :3])  # [B_img, C, latH, latW]

                # Empty latent: model generates from noise, guided by reference
                samples = torch.zeros(
                    [target_batch, vae_channels, latH, latW],
                    device=comfy.model_management.intermediate_device(),
                )
                samples = comfy.sample.fix_empty_latent_channels(model, samples)
                denoise = 1.0  # force full denoising — edit model fills from scratch
                print("[CRT KSampler Batch] edit_model_flux2klein: denoise forced to 1.0")

            else:
                # Normal img2img: encode image as starting latent
                samples = vae.encode(imgs[:, :, :, :3])  # [B_img, C, latH, latW]

        elif latent_image is not None:
            samples = latent_image["samples"]
            samples = comfy.sample.fix_empty_latent_channels(model, samples)
            target_batch = max(samples.shape[0], B_cond)

        else:
            # No input — generate a dummy empty latent and warn
            print(
                "[CRT KSampler Batch] WARNING: no image or latent_image connected — "
                "generating dummy 64x64 empty latent"
            )
            target_batch = B_cond
            dummy_side   = max(1, 64 // vae_downscale)
            samples = torch.zeros(
                [target_batch, vae_channels, dummy_side, dummy_side],
                device=comfy.model_management.intermediate_device(),
            )
            samples = comfy.sample.fix_empty_latent_channels(model, samples)

        # Cycle samples to target_batch using modulo (handles any ratio cleanly)
        if samples.shape[0] != target_batch:
            idx = [i % samples.shape[0] for i in range(target_batch)]
            samples = samples[idx]

        # Cycle ref_latents to target_batch using modulo
        if ref_latents is not None and ref_latents.shape[0] != target_batch:
            idx_r = [i % ref_latents.shape[0] for i in range(target_batch)]
            ref_latents = ref_latents[idx_r]

        seeds = [seed + i if increment_seed else seed for i in range(target_batch)]

        # ── Conditioning ──────────────────────────────────────────────────────
        positive = self._fix_conditioning(positive, target_batch)

        if negative:
            negative = self._fix_conditioning(negative, target_batch)
        else:
            ref      = positive[0][0]
            zero_cond   = torch.zeros_like(ref)
            zero_pooled = torch.zeros(target_batch, ref.shape[-1], device=ref.device, dtype=ref.dtype)
            negative = [[zero_cond, {"pooled_output": zero_pooled}]]

        # Pre-attach reference conditioning for PARALLEL mode only.
        # Sequential mode attaches per-item inside the loop.
        pos_for_parallel = positive
        neg_for_parallel = negative

        if edit_model_flux2klein and ref_latents is not None and mode == "Batch (Parallel)":
            if reference_mode == "shared":
                ref_pos = ref_latents[0:1]                    # broadcast to whole batch
            else:
                ref_pos = ref_latents                         # [B, C, H, W] — one per item
            ref_neg = torch.zeros_like(ref_pos)
            pos_for_parallel = _attach_reference_latent(positive, ref_pos)
            neg_for_parallel = _attach_reference_latent(negative, ref_neg)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ── Sampling ──────────────────────────────────────────────────────────
        if mode == "Sequential":
            latents_out = []
            for i in range(target_batch):
                pos_i = [[positive[j][0][i:i+1], positive[j][1]] for j in range(len(positive))]
                neg_i = [[negative[j][0][i:i+1], negative[j][1]] for j in range(len(negative))]

                # Per-item reference attachment in sequential edit mode
                if edit_model_flux2klein and ref_latents is not None:
                    ref_i = ref_latents[i:i+1] if reference_mode == "per_item" else ref_latents[0:1]
                    pos_i = _attach_reference_latent(pos_i, ref_i)
                    neg_i = _attach_reference_latent(neg_i, torch.zeros_like(ref_i))

                latent_i = samples[i:i+1]
                cur_seed = seeds[i]
                noise_i  = comfy.sample.prepare_noise(latent_i, cur_seed, None)

                out = comfy.sample.sample(
                    model, noise_i, steps, cfg, sampler_name, scheduler,
                    pos_i, neg_i, latent_i,
                    denoise=denoise,
                    seed=cur_seed,
                    callback=latent_preview.prepare_callback(model, steps),
                    disable_pbar=disable_pbar,
                )
                latents_out.append(out)
            samples_out = torch.cat(latents_out, dim=0)

        else:
            # Parallel: one batched GPU call + background carousel preview
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
                    n = min(target_batch, x0.shape[0])
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

            samples_out = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                pos_for_parallel, neg_for_parallel, samples,
                denoise=denoise,
                seed=seeds[0],
                callback=batch_callback,
                disable_pbar=disable_pbar,
            )

            _stop.set()
            _thread.join(timeout=1.0)

        # ── VAE Decode ────────────────────────────────────────────────────────
        images_out = None
        grid_out   = None

        if enable_vae_decode:
            decoded    = vae.decode(samples_out)   # [B, H, W, C]
            images_out = decoded

            if save_images:
                for i in range(target_batch):
                    self._save_image(
                        decoded[i], save_folder_path, save_subfolder_name,
                        save_filename_prefix, seeds[i], prompt, extra_pnginfo,
                    )

            if create_comparison_grid and target_batch > 1:
                # Concat horizontally: [B, H, W, C] → [1, H, B*W, C]
                grid_out = torch.cat(list(decoded.unbind(0)), dim=1).unsqueeze(0)

        # Preserve noise_mask if latent_image had one
        result = {"samples": samples_out}
        if latent_image is not None and "noise_mask" in latent_image:
            result["noise_mask"] = latent_image["noise_mask"]

        return (images_out, grid_out, result)


NODE_CLASS_MAPPINGS      = {"CRT_KSamplerBatch": CRT_KSamplerBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"CRT_KSamplerBatch": "KSampler Batch (CRT)"}

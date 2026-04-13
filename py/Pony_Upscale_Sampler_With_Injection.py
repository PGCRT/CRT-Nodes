import torch
import torch.nn.functional as F
import numpy as np
import logging
import math
import comfy.utils
import comfy.sd
import comfy.sample
import comfy.model_management as mm
import comfy.samplers
import node_helpers
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
import folder_paths
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled
from .Latent_Injection_Sampler import (
    split_sigmas_denoise_stateless,
    multiply_sigmas_stateless,
    build_dd_sampler,
    run_custom_sample,
)


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def colored_print(message, color=Colors.ENDC):
    print(f"{color}{message}{Colors.ENDC}")


def attach_reference_latent(conditioning, ref_samples):
    if conditioning is None or isinstance(conditioning, str):
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [ref_samples]},
        append=False,
    )


class ColorMatch:
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        colored_print(
            f"🎨 Applying color matching (method: {method}, strength: {strength:.2f})...",
            Colors.CYAN,
        )

        try:
            from color_matcher import ColorMatcher
        except Exception:
            colored_print("❌ ERROR: 'color-matcher' library not found.", Colors.RED)
            raise Exception(
                "ColorMatch requires 'color-matcher'. Please 'pip install color-matcher'"
            )

        cm, out = ColorMatcher(), []
        image_ref, image_target = image_ref.cpu(), image_target.cpu()

        batch_size = image_target.size(0)
        colored_print(f"   🔄 Processing {batch_size} image(s)...", Colors.BLUE)

        for i in range(batch_size):
            target_np, ref_np = (
                image_target[i].numpy(),
                image_ref[
                    i if image_ref.size(0) == image_target.size(0) else 0
                ].numpy(),
            )
            target_mean = np.mean(target_np, axis=(0, 1))
            ref_mean = np.mean(ref_np, axis=(0, 1))

            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            result = target_np + strength * (result - target_np)

            final_mean = np.mean(result, axis=(0, 1))
            colored_print(
                f"   📊 Image {i + 1}: Target RGB({target_mean[0]:.3f},{target_mean[1]:.3f},{target_mean[2]:.3f}) → Final RGB({final_mean[0]:.3f},{final_mean[1]:.3f},{final_mean[2]:.3f})",
                Colors.BLUE,
            )

            out.append(torch.from_numpy(result))

        colored_print("✅ Color matching completed!", Colors.GREEN)
        return (torch.stack(out, dim=0).to(torch.float32).clamp_(0, 1),)


class PonyUpscaleSamplerWithInjection:
    @classmethod
    def INPUT_TYPES(s):
        upscale_models = folder_paths.get_filename_list("upscale_models")
        default_upscale_model = upscale_models[0] if len(upscale_models) > 0 else ""
        upscale_model_choices = upscale_models if len(upscale_models) > 0 else [""]
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "euler"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Classifier Free Guidance scale. Higher values follow the prompt more closely.",
                    },
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.33,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Amount of denoising to apply. 1.0 = full denoising (txt2img), 0.5-0.8 typical for img2img.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_shift": (
                    "INT",
                    {
                        "default": 0,
                        "min": -100000,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "Offset added to the main seed for variation",
                    },
                ),
                "edit_model_flux2klein": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable per-tile ReferenceLatent-style conditioning for edit models",
                    },
                ),
                "enable_upscale_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable AI model upscaling before sampling",
                    },
                ),
                "upscale_model_name": (
                    upscale_model_choices,
                    {
                        "default": default_upscale_model,
                        "tooltip": "Select upscale model from models/upscale_models folder",
                    },
                ),
                "tile_size_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target megapixels per tile used to derive upscale factor",
                    },
                ),
                "tile_grid": (
                    ["2x2", "3x3", "4x4", "5x5", "6x6", "7x7", "8x8"],
                    {
                        "default": "4x4",
                        "tooltip": "Grid size for tiling (e.g., 4x4 = 16 tiles)",
                    },
                ),
                "tile_padding": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 25.0,
                        "step": 0.5,
                        "tooltip": "Padding around each tile as percentage of tile size",
                    },
                ),
                "mask_blur": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "Blur radius for tile blending as percentage of tile size",
                    },
                ),
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
                "details_amount_stage1": (
                    "FLOAT",
                    {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "details_amount_stage2": (
                    "FLOAT",
                    {"default": 0.15, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "enable_noise_injection": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                        "tooltip": "Enable noise injection during sampling",
                    },
                ),
                "injection_point": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Percentage of steps after which to inject noise",
                    },
                ),
                "injection_seed_offset": (
                    "INT",
                    {
                        "default": 1,
                        "min": -100,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Offset added to main seed for injection noise",
                    },
                ),
                "injection_strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.01,
                        "tooltip": "Strength of injected noise",
                    },
                ),
                "normalize_injected_noise": (
                    ["enable", "disable"],
                    {
                        "default": "enable",
                        "tooltip": "Normalize injected noise to match latent statistics",
                    },
                ),
                "vae_decode_tiled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use tiled VAE decode. Tile size and overlap are derived from tile size megapixels and tile padding.",
                    },
                ),
                "color_match_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength of color matching between original and enhanced result. 0.0 = disabled, 1.0 = full matching",
                    },
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    CATEGORY = "CRT/Sampling"

    def __init__(self):
        self.upscale_loader = UpscaleModelLoader()
        self.image_upscaler = ImageUpscaleWithModel()

    def _get_model_scale_factor(self, model_name):
        """Detect model scale factor from filename."""
        model_name_lower = model_name.lower()
        if "4x" in model_name_lower or "x4" in model_name_lower:
            return 4
        elif "2x" in model_name_lower or "x2" in model_name_lower:
            return 2
        elif "8x" in model_name_lower or "x8" in model_name_lower:
            return 8
        else:
            return 4

    def _upscale_image(self, image, method, model_name, target_scale):
        """Upscale image using AI model or simple resize."""
        colored_print(f"🚀 Starting image upscaling ({method})...", Colors.CYAN)

        input_h, input_w = image.shape[1], image.shape[2]
        target_h = int(input_h * target_scale)
        target_w = int(input_w * target_scale)

        colored_print(
            f"   📐 Input: {input_w}x{input_h} → Target: {target_w}x{target_h} ({target_scale:.1f}x)",
            Colors.BLUE,
        )

        if method == "simple_resize":
            colored_print("   🔄 Using simple Lanczos resize...", Colors.CYAN)
            upscaled_image = comfy.utils.lanczos(
                image.permute(0, 3, 1, 2), target_w, target_h
            ).permute(0, 2, 3, 1)
            colored_print(
                f"   ✅ Simple resize completed: {upscaled_image.shape[2]}x{upscaled_image.shape[1]}",
                Colors.GREEN,
            )
            return upscaled_image

        else:  # method == "model"
            try:
                upscale_model = self.upscale_loader.load_model(model_name)[0]
                colored_print(
                    f"   ✅ Upscale model '{model_name}' loaded successfully",
                    Colors.GREEN,
                )
            except Exception as e:
                colored_print(f"   ❌ Failed to load upscale model: {e}", Colors.RED)
                raise
            model_scale = self._get_model_scale_factor(model_name)
            colored_print(f"   🔍 Model scale factor: {model_scale}x", Colors.BLUE)
            device = mm.get_torch_device()
            upscale_model.to(device)

            with torch.no_grad():
                upscaled_image = self.image_upscaler.upscale(upscale_model, image)[0]

            current_h, current_w = upscaled_image.shape[1], upscaled_image.shape[2]
            colored_print(f"   📊 Model output: {current_w}x{current_h}", Colors.BLUE)
            if current_h != target_h or current_w != target_w:
                colored_print(
                    "   🔄 Final resize to exact target using Lanczos...", Colors.CYAN
                )
                upscaled_image = comfy.utils.lanczos(
                    upscaled_image.permute(0, 3, 1, 2), target_w, target_h
                ).permute(0, 2, 3, 1)
            upscale_model.to(mm.unet_offload_device())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            colored_print(
                f"   ✅ Model upscaling completed: {upscaled_image.shape[2]}x{upscaled_image.shape[1]}",
                Colors.GREEN,
            )
            return upscaled_image

    def _upscale_latent_simple(self, latent_samples, target_scale):
        """Simple upscale of latent using interpolation."""
        colored_print(
            f"🚀 Starting latent upscaling (simple resize {target_scale:.1f}x)...",
            Colors.CYAN,
        )

        b, c, h_latent, w_latent = latent_samples.shape
        target_h = int(h_latent * target_scale)
        target_w = int(w_latent * target_scale)

        colored_print(
            f"   📐 Latent: {w_latent}x{h_latent} → {target_w}x{target_h}", Colors.BLUE
        )
        colored_print(
            f"   📐 Pixel equivalent: {w_latent * 8}x{h_latent * 8} → {target_w * 8}x{target_h * 8}",
            Colors.BLUE,
        )

        upscaled_latent = F.interpolate(
            latent_samples,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        colored_print(
            f"   ✅ Latent upscaling completed: {upscaled_latent.shape[2]}x{upscaled_latent.shape[3]}",
            Colors.GREEN,
        )
        return upscaled_latent

    def _grid_to_dimensions(self, tile_grid):
        """Convert grid string to dimensions."""
        if tile_grid == "2x2":
            return 2, 2
        elif tile_grid == "3x3":
            return 3, 3
        elif tile_grid == "4x4":
            return 4, 4
        elif tile_grid == "5x5":
            return 5, 5
        elif tile_grid == "6x6":
            return 6, 6
        elif tile_grid == "7x7":
            return 7, 7
        elif tile_grid == "8x8":
            return 8, 8
        else:
            return 4, 4  # Default fallback

    def _tile_bounds(self, total_w, total_h, rows, columns):
        bounds = []
        for y in range(rows):
            y0 = (y * total_h) // rows
            y1 = ((y + 1) * total_h) // rows
            for x in range(columns):
                x0 = (x * total_w) // columns
                x1 = ((x + 1) * total_w) // columns
                bounds.append((x0, y0, x1, y1, x, y))
        return bounds

    def _upscale_image_per_tile(
        self,
        image,
        rows,
        columns,
        target_tile_megapixels,
        use_model,
        upscale_model_name,
    ):
        h, w = image.shape[1], image.shape[2]
        tile_w_base = max(1.0, float(w) / float(columns))
        tile_h_base = max(1.0, float(h) / float(rows))
        tile_area_base = tile_w_base * tile_h_base
        target_tile_area = max(0.1, float(target_tile_megapixels)) * 1_000_000.0
        scale = max(1.0, math.sqrt(target_tile_area / tile_area_base))

        method = "model" if use_model else "simple_resize"
        colored_print(
            f"📈 Tile-wise upscaling ({method}) at ~{target_tile_megapixels:.2f} MP/tile ({scale:.3f}x)",
            Colors.HEADER,
        )

        row_images = []
        for y in range(rows):
            row_tiles = []
            y0 = (y * h) // rows
            y1 = ((y + 1) * h) // rows
            for x in range(columns):
                x0 = (x * w) // columns
                x1 = ((x + 1) * w) // columns
                tile = image[:, y0:y1, x0:x1, :]
                up_tile = self._upscale_image(tile, method, upscale_model_name, scale)
                row_tiles.append(up_tile)

            row_heights = [t.shape[1] for t in row_tiles]
            max_row_h = max(row_heights)
            normalized_row = []
            for t in row_tiles:
                if t.shape[1] != max_row_h:
                    t = comfy.utils.lanczos(
                        t.permute(0, 3, 1, 2), t.shape[2], max_row_h
                    ).permute(0, 2, 3, 1)
                normalized_row.append(t)
            row_images.append(torch.cat(normalized_row, dim=2))

        row_widths = [r.shape[2] for r in row_images]
        max_w = max(row_widths)
        normalized_rows = []
        for r in row_images:
            if r.shape[2] != max_w:
                r = comfy.utils.lanczos(
                    r.permute(0, 3, 1, 2), max_w, r.shape[1]
                ).permute(0, 2, 3, 1)
            normalized_rows.append(r)

        stitched = torch.cat(normalized_rows, dim=1)
        return stitched, scale

    def _create_blend_mask(
        self,
        height,
        width,
        overlap_h,
        overlap_w,
        is_top,
        is_left,
        is_bottom,
        is_right,
        device,
    ):
        """Create cosine blend mask for seamless tile blending."""
        mask = torch.ones((height, width), device=device)

        if overlap_h > 0:
            x = torch.linspace(0, math.pi, overlap_h, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_top:
                mask[:overlap_h, :] *= cos[:, None]
            if not is_bottom:
                mask[-overlap_h:, :] *= cos.flip(0)[:, None]
        if overlap_w > 0:
            x = torch.linspace(0, math.pi, overlap_w, device=device)
            cos = 0.5 * (1 - torch.cos(x))
            if not is_left:
                mask[:, :overlap_w] *= cos[None, :]
            if not is_right:
                mask[:, -overlap_w:] *= cos.flip(0)[None, :]

        return mask

    def _sample_tiled(
        self,
        model,
        positive,
        negative,
        working_latent,
        cfg,
        actual_seed,
        tile_grid,
        tile_padding,
        mask_blur,
        sampler_stage1,
        sampler_stage2,
        full_sigmas,
        stage1_sigmas,
        stage2_sigmas,
        edit_model_flux2klein,
        tile_ref_latents,
        enable_noise_injection,
        injection_seed_offset,
        injection_strength,
        normalize_injected_noise,
    ):
        """Perform tiled sampling using proper overlapping tiles with blend masks."""
        colored_print("🧩 Starting tiled sampling...", Colors.HEADER)

        device = mm.get_torch_device()
        latent_samples = working_latent["samples"].clone().to(device)
        b, c, h_latent, w_latent = latent_samples.shape
        rows, columns = self._grid_to_dimensions(tile_grid)
        total_tiles = rows * columns
        colored_print(
            f"   📐 Grid configuration: {tile_grid} = {rows}x{columns} = {total_tiles} tiles",
            Colors.BLUE,
        )
        base_tile_h = h_latent // rows
        base_tile_w = w_latent // columns
        overlap_h = int(base_tile_h * (tile_padding / 100.0))
        overlap_w = int(base_tile_w * (tile_padding / 100.0))

        colored_print(
            f"   📋 Base tile size: {base_tile_w}x{base_tile_h} latent ({base_tile_w * 8}x{base_tile_h * 8} pixels)",
            Colors.BLUE,
        )
        colored_print(
            f"   🎨 Overlap: {tile_padding:.1f}% = {overlap_w}x{overlap_h} latent ({overlap_w * 8}x{overlap_h * 8} pixels)",
            Colors.BLUE,
        )
        samples = torch.zeros_like(latent_samples, device=device)
        pbar = comfy.utils.ProgressBar(total_tiles)
        for y in range(rows):
            for x in range(columns):
                tile_seed_offset = y * columns + x
                tile_seed = actual_seed + tile_seed_offset
                y_start = max(0, y * base_tile_h - overlap_h)
                y_end = min(h_latent, (y + 1) * base_tile_h + overlap_h)
                x_start = max(0, x * base_tile_w - overlap_w)
                x_end = min(w_latent, (x + 1) * base_tile_w + overlap_w)

                tile_h = y_end - y_start
                tile_w = x_end - x_start

                colored_print(
                    f"   🔄 Processing tile {y + 1},{x + 1}: {tile_w}x{tile_h} latent (seed: {tile_seed})",
                    Colors.YELLOW,
                )
                section = latent_samples[:, :, y_start:y_end, x_start:x_end].clone()
                tile_latent_dict = {"samples": section}
                tile_index = y * columns + x
                pos_cond = positive
                neg_cond = negative
                if (
                    edit_model_flux2klein
                    and tile_ref_latents is not None
                    and tile_index < len(tile_ref_latents)
                ):
                    ref_samples = tile_ref_latents[tile_index]
                    pos_cond = attach_reference_latent(positive, ref_samples)
                    neg_cond = attach_reference_latent(
                        negative,
                        torch.zeros_like(ref_samples),
                    )
                if enable_noise_injection == "enable":
                    stage1_latent = run_custom_sample(
                        model,
                        tile_latent_dict,
                        sampler_stage1,
                        stage1_sigmas,
                        pos_cond,
                        neg_cond,
                        cfg,
                        tile_seed,
                        disable_noise=False,
                    )
                    injection_seed = tile_seed + injection_seed_offset
                    injected_latent_samples = stage1_latent["samples"].clone()

                    torch.manual_seed(injection_seed)
                    new_noise = torch.randn_like(injected_latent_samples)

                    if normalize_injected_noise == "enable":
                        original_std = injected_latent_samples.std().item()
                        original_mean = injected_latent_samples.mean().item()
                        if original_std > 1e-6:
                            new_noise = new_noise * original_std + original_mean

                    injected_latent_samples += new_noise * injection_strength
                    injected_latent = stage1_latent.copy()
                    injected_latent["samples"] = injected_latent_samples
                    processed_tile_latent = run_custom_sample(
                        model,
                        injected_latent,
                        sampler_stage2,
                        stage2_sigmas,
                        pos_cond,
                        neg_cond,
                        cfg,
                        tile_seed,
                        disable_noise=True,
                    )
                else:
                    processed_tile_latent = run_custom_sample(
                        model,
                        tile_latent_dict,
                        sampler_stage1,
                        full_sigmas,
                        pos_cond,
                        neg_cond,
                        cfg,
                        tile_seed,
                        disable_noise=False,
                    )
                is_top = y_start == 0
                is_left = x_start == 0
                is_bottom = y_end == h_latent
                is_right = x_end == w_latent

                blend_mask = self._create_blend_mask(
                    tile_h,
                    tile_w,
                    overlap_h,
                    overlap_w,
                    is_top,
                    is_left,
                    is_bottom,
                    is_right,
                    device,
                )
                blend_mask = (
                    blend_mask.unsqueeze(0).unsqueeze(0).expand(b, c, tile_h, tile_w)
                )
                processed_section = processed_tile_latent["samples"].to(device)
                blend_mask = blend_mask.to(device)
                samples[:, :, y_start:y_end, x_start:x_end] = (
                    samples[:, :, y_start:y_end, x_start:x_end] * (1 - blend_mask)
                    + processed_section * blend_mask
                )

                pbar.update(1)

        colored_print(
            f"   ✅ Tiled sampling completed with {total_tiles} tiles!", Colors.GREEN
        )
        return {"samples": samples}

    def execute(
        self,
        model,
        positive,
        sampler_name,
        scheduler,
        cfg,
        steps,
        denoise,
        seed,
        seed_shift,
        edit_model_flux2klein,
        enable_upscale_model,
        upscale_model_name,
        tile_size_megapixels,
        tile_grid,
        tile_padding,
        mask_blur,
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
        vae_decode_tiled,
        color_match_strength,
        negative=None,
        image=None,
        latent=None,
        vae=None,
    ):

        colored_print(
            "\n🐎 Starting Pony Upscale Sampler with Injection, Tiling & Color Matching...",
            Colors.HEADER,
        )

        if vae is None:
            raise ValueError("Image Upscale Sampler requires a VAE input.")

        actual_seed = seed + seed_shift
        if negative is None:
            negative = []
            for t in positive:
                negative.append([torch.zeros_like(t[0]), t[1].copy()])
        colored_print("🎲 Seed Configuration:", Colors.HEADER)
        colored_print(f"   Base Seed: {seed}", Colors.BLUE)
        colored_print(f"   Seed Shift: {seed_shift:+d}", Colors.BLUE)
        colored_print(f"   Final Seed: {actual_seed}", Colors.GREEN)
        if image is None and latent is None:
            colored_print(
                "❌ ERROR: PonyUpscaleSampler requires either an 'image' or a 'latent' input.",
                Colors.RED,
            )
            raise ValueError(
                "PonyUpscaleSampler requires either an 'image' or a 'latent' input."
            )
        if image is not None:
            colored_print("🖼️  Using provided image input...", Colors.CYAN)
            source_image = image
            h, w = image.shape[1], image.shape[2]
            colored_print(f"   📐 Source image dimensions: {w}x{h}", Colors.BLUE)
        else:
            colored_print("🎯 Decoding latent to image...", Colors.CYAN)
            source_image = vae.decode(latent["samples"])
            h, w = source_image.shape[1], source_image.shape[2]
            colored_print(f"   📐 Decoded image dimensions: {w}x{h}", Colors.BLUE)
        rows, columns = self._grid_to_dimensions(tile_grid)
        tile_w_base = max(1.0, float(w) / float(columns))
        tile_h_base = max(1.0, float(h) / float(rows))
        tile_area_base = tile_w_base * tile_h_base
        target_tile_area = max(0.1, float(tile_size_megapixels)) * 1_000_000.0
        upscale_by = max(1.0, math.sqrt(target_tile_area / tile_area_base))

        tile_ref_latents = None
        if edit_model_flux2klein:
            colored_print(
                "🧠 Precomputing per-tile reference latents before tile upscale...",
                Colors.CYAN,
            )
            tile_ref_latents = []
            for x0, y0, x1, y1, _x, _y in self._tile_bounds(w, h, rows, columns):
                tile_image = source_image[:, y0:y1, x0:x1, :]
                tile_ref_latents.append(vae.encode(tile_image))

        original_image_for_color_match = None
        if color_match_strength > 0:
            colored_print(
                "🎨 Storing original image for color matching...", Colors.CYAN
            )
            original_image_for_color_match = source_image.clone()
            orig_h, orig_w = (
                original_image_for_color_match.shape[1],
                original_image_for_color_match.shape[2],
            )
            colored_print(
                f"   📐 Reference image stored: {orig_w}x{orig_h}", Colors.BLUE
            )
        can_use_upscale_model = bool(upscale_model_name)
        if enable_upscale_model and not can_use_upscale_model:
            colored_print(
                "⚠️ No upscale model selected/available. Falling back to no model upscaling.",
                Colors.YELLOW,
            )

        do_resize = upscale_by > 1.0
        do_model_upscale = enable_upscale_model and can_use_upscale_model and do_resize

        if do_resize:
            source_image, _applied_scale = self._upscale_image_per_tile(
                source_image,
                rows,
                columns,
                tile_size_megapixels,
                do_model_upscale,
                upscale_model_name,
            )
            h, w = source_image.shape[1], source_image.shape[2]
            colored_print(f"   📐 Final source dimensions: {w}x{h}", Colors.GREEN)
        else:
            colored_print(
                "🚫 Upscaling skipped (target tile size <= current)", Colors.YELLOW
            )

        colored_print("🔄 Encoding image to latent space...", Colors.CYAN)
        mm.load_model_gpu(vae.patcher)
        latent_samples = vae.encode(source_image)
        working_latent = {"samples": latent_samples}
        b, c, h_latent, w_latent = latent_samples.shape
        colored_print(
            f"   📐 Encoded latent dimensions: {w_latent}x{h_latent} (channels: {c})",
            Colors.BLUE,
        )
        colored_print("⚙️  Sampling Configuration:", Colors.HEADER)
        colored_print(f"   🎲 Seed: {actual_seed}", Colors.BLUE)
        colored_print(
            f"   📊 Sampler: {sampler_name} | Scheduler: {scheduler}", Colors.BLUE
        )
        colored_print(
            f"   🔄 Steps: {steps} | CFG: {cfg:.1f} | Denoise: {denoise:.2f}",
            Colors.BLUE,
        )
        colored_print(
            f"   🧩 Tiling: enabled | Grid: {tile_grid} (operates in latent space)",
            Colors.CYAN,
        )
        colored_print(
            f"   💉 Noise Injection: {enable_noise_injection}",
            Colors.CYAN if enable_noise_injection == "enable" else Colors.YELLOW,
        )

        base_sampler = comfy.samplers.sampler_object(sampler_name)
        sampler_stage1 = build_dd_sampler(
            base_sampler,
            details_amount_stage1,
            0.0,
            1.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.0,
            True,
            1.0,
        )
        sampler_stage2 = build_dd_sampler(
            base_sampler,
            details_amount_stage2,
            0.0,
            1.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            True,
            1.0,
        )

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
        stage1_sigmas, stage2_sigmas = split_sigmas_denoise_stateless(
            sigmas_full,
            injection_point,
        )
        stage1_sigmas = multiply_sigmas_stateless(
            stage1_sigmas,
            stage1_sigma_factor,
            stage1_sigma_start,
            stage1_sigma_end,
        )
        stage2_sigmas = multiply_sigmas_stateless(
            stage2_sigmas,
            stage2_sigma_factor,
            stage2_sigma_start,
            stage2_sigma_end,
        )
        full_sigmas = multiply_sigmas_stateless(
            sigmas_full,
            stage1_sigma_factor,
            stage1_sigma_start,
            stage1_sigma_end,
        )

        use_tiled_sampling = True
        colored_print(
            f"   🎨 Color Matching: {'Enabled' if color_match_strength > 0 else 'Disabled'} (strength: {color_match_strength:.2f})",
            Colors.CYAN if color_match_strength > 0 else Colors.YELLOW,
        )
        if use_tiled_sampling:
            colored_print(f"\n🧩 Using tiled sampling ({tile_grid})...", Colors.GREEN)
            final_latent = self._sample_tiled(
                model,
                positive,
                negative,
                working_latent,
                cfg,
                actual_seed,
                tile_grid,
                tile_padding,
                mask_blur,
                sampler_stage1,
                sampler_stage2,
                full_sigmas,
                stage1_sigmas,
                stage2_sigmas,
                edit_model_flux2klein,
                tile_ref_latents,
                enable_noise_injection,
                injection_seed_offset,
                injection_strength,
                normalize_injected_noise,
            )
        else:
            pos_cond = positive
            neg_cond = negative
            if edit_model_flux2klein:
                pos_cond = attach_reference_latent(positive, working_latent["samples"])
                neg_cond = attach_reference_latent(
                    negative,
                    torch.zeros_like(working_latent["samples"]),
                )
            if enable_noise_injection == "enable":
                stage1_latent = run_custom_sample(
                    model,
                    working_latent,
                    sampler_stage1,
                    stage1_sigmas,
                    pos_cond,
                    neg_cond,
                    cfg,
                    actual_seed,
                    disable_noise=False,
                )
                injection_seed = actual_seed + injection_seed_offset
                injected_latent_samples = stage1_latent["samples"].clone()

                torch.manual_seed(injection_seed)
                new_noise = torch.randn_like(injected_latent_samples)

                if normalize_injected_noise == "enable":
                    original_std = injected_latent_samples.std().item()
                    original_mean = injected_latent_samples.mean().item()
                    colored_print(
                        f"   📊 Original latent - Mean: {original_mean:.4f}, Std: {original_std:.4f}",
                        Colors.BLUE,
                    )
                    if original_std > 1e-6:
                        new_noise = new_noise * original_std + original_mean

                injected_latent_samples += new_noise * injection_strength
                injected_latent = stage1_latent.copy()
                injected_latent["samples"] = injected_latent_samples
                final_latent = run_custom_sample(
                    model,
                    injected_latent,
                    sampler_stage2,
                    stage2_sigmas,
                    pos_cond,
                    neg_cond,
                    cfg,
                    actual_seed,
                    disable_noise=True,
                )
            else:
                colored_print("\n🔥 Starting standard sampling...", Colors.GREEN)
                final_latent = run_custom_sample(
                    model,
                    working_latent,
                    sampler_stage1,
                    full_sigmas,
                    pos_cond,
                    neg_cond,
                    cfg,
                    actual_seed,
                    disable_noise=False,
                )
        colored_print("🎨 Decoding final latent to image...", Colors.CYAN)
        if vae_decode_tiled:
            decode_tile = int(
                round(math.sqrt(max(0.1, tile_size_megapixels) * 1_000_000.0))
            )
            decode_tile = max(64, min(4096, (decode_tile // 32) * 32))
            decode_overlap = int(round(decode_tile * max(0.0, tile_padding) / 100.0))
            decode_overlap = max(0, min(decode_tile // 4, (decode_overlap // 32) * 32))
            colored_print(
                f"   🧩 Using tiled decode: tile={decode_tile}, overlap={decode_overlap}",
                Colors.BLUE,
            )
            final_image = VAEDecodeTiled().decode(
                vae,
                final_latent,
                tile_size=decode_tile,
                overlap=decode_overlap,
            )[0]
        else:
            final_image = vae.decode(final_latent["samples"])
        final_h, final_w = final_image.shape[1], final_image.shape[2]
        colored_print(f"   📐 Final image dimensions: {final_w}x{final_h}", Colors.BLUE)
        if color_match_strength > 0 and original_image_for_color_match is not None:
            colored_print(
                "🎨 Applying color matching to final result...", Colors.HEADER
            )
            if original_image_for_color_match.shape[1:3] != final_image.shape[1:3]:
                colored_print(
                    "   🔄 Resizing reference image for color matching...", Colors.CYAN
                )
                ref_resized = comfy.utils.lanczos(
                    original_image_for_color_match.permute(0, 3, 1, 2), final_w, final_h
                ).permute(0, 2, 3, 1)
                colored_print(
                    f"   📐 Reference resized to: {ref_resized.shape[2]}x{ref_resized.shape[1]}",
                    Colors.BLUE,
                )
            else:
                ref_resized = original_image_for_color_match
            color_matched_result = ColorMatch().colormatch(
                ref_resized, final_image, method="mkl", strength=color_match_strength
            )[0]

            colored_print(
                f"   ✅ Color matching applied with strength: {color_match_strength:.2f}",
                Colors.GREEN,
            )
            final_image = color_matched_result
        elif color_match_strength > 0:
            colored_print(
                "🚫 Color matching skipped - no reference image available",
                Colors.YELLOW,
            )
        else:
            colored_print("🚫 Color matching disabled (strength = 0)", Colors.YELLOW)
        colored_print(
            "\n✅ Pony Upscale Sampler with Injection, Tiling & Color Matching completed!",
            Colors.HEADER,
        )

        if do_resize:
            original_w, original_h = (
                int(final_w / upscale_by),
                int(final_h / upscale_by),
            )
            colored_print(
                f"   🎯 Process: {original_w}x{original_h} → {final_w}x{final_h} ({upscale_by:.1f}x, {'model' if do_model_upscale else 'lanczos'})",
                Colors.GREEN,
            )
        else:
            colored_print(
                f"   🎯 Process: {final_w}x{final_h} (no upscaling)", Colors.GREEN
            )

        colored_print(
            f"   📈 Upscaling: {'Applied' if do_resize else 'Skipped'}",
            Colors.GREEN,
        )
        colored_print(
            "   🧩 Tiling: Applied (latent space)",
            Colors.GREEN,
        )
        colored_print(
            f"   💉 Noise Injection: {'Applied' if enable_noise_injection == 'enable' else 'Skipped'}",
            Colors.GREEN,
        )
        colored_print(
            f"   🎨 Color Matching: {'Applied' if color_match_strength > 0 and original_image_for_color_match is not None else 'Skipped'}",
            Colors.GREEN,
        )
        colored_print(f"   🎲 Seed used: {actual_seed}", Colors.GREEN)

        return (final_image, final_latent)


NODE_CLASS_MAPPINGS = {
    "PonyUpscaleSamplerWithInjection": PonyUpscaleSamplerWithInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PonyUpscaleSamplerWithInjection": "Pony Upscale Sampler with Injection, Tiling & Color Matching (CRT)"
}

import torch
import numpy as np
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
import comfy.sd
import folder_paths
import comfy.model_management
import latent_preview
import os
import cv2
import tempfile
import subprocess
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class Log:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def info(msg):
        print(f"{Log.CYAN}[INFO] {msg}{Log.ENDC}")

    @staticmethod
    def success(msg):
        print(f"{Log.GREEN}[âœ“] {msg}{Log.ENDC}")

    @staticmethod
    def warn(msg):
        print(f"{Log.YELLOW}[âš ] {msg}{Log.ENDC}")

    @staticmethod
    def error(msg):
        print(f"{Log.RED}[âœ—] {msg}{Log.ENDC}")

    @staticmethod
    def header(msg):
        print(f"\n{Log.HEADER}{Log.BOLD}{'='*60}\n{msg}\n{'='*60}{Log.ENDC}")

    @staticmethod
    def vram(label):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{Log.YELLOW}[VRAM {label}]{Log.ENDC} Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")


def set_shift(model, sigma_shift):
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:

        class ModelSampling(comfy.model_sampling.ModelSamplingDiscrete):
            pass

        model_sampling = ModelSampling(model.model.config)
    try:
        model_sampling.set_parameters(shift=sigma_shift)
    except AttributeError:
        model_sampling.shift = sigma_shift
    model.add_object_patch("model_sampling", model_sampling)
    return model


def conditioning_set_values(conditioning, values):
    if not isinstance(conditioning, list):
        return conditioning
    out = []
    for c in conditioning:
        new_c = [c[0].clone(), c[1].copy()]
        for k, v in values.items():
            new_c[1][k] = v
        out.append(new_c)
    return out


class CRT_WAN_BatchSampler:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "model_high_noise": (
                    "MODEL",
                    {
                        "tooltip": "Model used for the initial high-noise denoising phase. Typically handles structure and composition."
                    },
                ),
                "model_low_noise": (
                    "MODEL",
                    {
                        "tooltip": "Model used for the final low-noise refinement phase. Typically handles details and fine features."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "Positive prompt conditioning. Describes what you want to generate."},
                ),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Negative prompt conditioning. Describes what you want to avoid."},
                ),
                "width": (
                    "INT",
                    {
                        "default": 960,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Output width in pixels. Will be quantized to nearest multiple of 16.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Output height in pixels. Will be quantized to nearest multiple of 16.",
                    },
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Number of frames to generate. Use 1 for images, >1 for videos.",
                    },
                ),
                "batch_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 128,
                        "forceInput": True,
                        "tooltip": "Number of items to generate with different seeds. Higher values = more VRAM in parallel mode.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 4149210799938517,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Starting seed for generation. Each batch item uses seed+N if increment is enabled.",
                    },
                ),
                "increment_seed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, each batch item gets seed+1, seed+2, etc. If disabled, all use the same seed.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Total number of denoising steps. More steps = better quality but slower generation.",
                    },
                ),
                "boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sigma value where switching from high-noise to low-noise model occurs. Lower = earlier switch.",
                    },
                ),
                "cfg_high_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Classifier-Free Guidance scale for high-noise phase. Higher = stronger prompt adherence.",
                    },
                ),
                "cfg_low_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Classifier-Free Guidance scale for low-noise phase. Higher = stronger prompt adherence.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep",
                        "tooltip": "Sampling algorithm. Different samplers have different quality/speed tradeoffs.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "bong_tangent",
                        "tooltip": "Noise schedule that controls how denoising steps are distributed.",
                    },
                ),
                "sigma_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Shifts the noise schedule. Higher values = more denoising emphasis on high-frequency details.",
                    },
                ),
                "enable_vae_decode": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode latents to images. Disable to only output latents."},
                ),
                "enable_vae_tiled_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use tiled VAE decoding to reduce VRAM usage. Useful for high resolutions.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Tile size for tiled VAE decode (spatial). Smaller = less VRAM but slower.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Overlap between tiles for tiled VAE decode. Higher = fewer seams but slower.",
                    },
                ),
                "temporal_tile_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Tile size for tiled VAE decode (temporal/frames). Only relevant for video generation.",
                    },
                ),
                "temporal_tile_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Overlap between temporal tiles. Only relevant for video generation.",
                    },
                ),
                "create_comparison_grid": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Create a side-by-side comparison grid of all batch outputs."},
                ),
                "save_videos_images": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save outputs to disk. Disable to only use outputs in workflow."},
                ),
                "save_folder_path": (
                    "STRING",
                    {"default": output_dir, "tooltip": "Root folder where outputs will be saved."},
                ),
                "save_subfolder_name": (
                    "STRING",
                    {
                        "default": "FAST_BATCH",
                        "tooltip": "Subfolder name inside the root folder for organizing outputs.",
                    },
                ),
                "save_filename_prefix": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": "Prefix for output filenames. Seed will be appended automatically.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 120,
                        "tooltip": "Frames per second for video outputs. Only relevant when frame_count > 1.",
                    },
                ),
                "processing_mode": (
                    ["Sequential", "Parallel"],
                    {
                        "default": "Parallel",
                        "tooltip": "Parallel = process all batches together (fast, high VRAM). Sequential = one at a time (slow, low VRAM).",
                    },
                ),
                "offload_conditioning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload text conditioning to CPU between items. Reduces VRAM slightly, forces sequential mode.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": "VAE model for encoding/decoding. Required if enable_vae_decode is True or using I2V mode."
                    },
                ),
                "start_image": (
                    "IMAGE",
                    {
                        "tooltip": "Starting image for Image-to-Video generation. When provided, switches to I2V mode automatically."
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "high_noise_latent_batch",
        "final_latent_batch",
        "final_images_batch",
        "comparison_grid",
        "settings_string",
    )
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def _save_image(
        self, image_tensor, folder_path, subfolder_name, filename_prefix, seed, prompt=None, extra_pnginfo=None
    ):
        try:
            # FIX: Clean the subfolder name before joining the path
            cleaned_subfolder = subfolder_name.strip().lstrip('/\\')
            final_dir = os.path.join(folder_path, cleaned_subfolder)
            os.makedirs(final_dir, exist_ok=True)

            # FIX: Clean the filename prefix before using it in the f-string
            cleaned_prefix = filename_prefix.strip().lstrip('/\\')
            base_filename = f"{cleaned_prefix}_{seed}"
            image_filepath = os.path.join(final_dir, f"{base_filename}.png")
            counter = 1
            while os.path.exists(image_filepath):
                image_filepath = os.path.join(final_dir, f"{base_filename}_{counter}.png")
                counter += 1

            pil_image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                metadata.add_text("workflow", json.dumps(extra_pnginfo["workflow"]))

            pil_image.save(image_filepath, "PNG", pnginfo=metadata)
            Log.success(f"Image saved: {os.path.basename(image_filepath)}")
        except Exception as e:
            Log.error(f"Failed to save image (seed {seed}): {e}")

    def _save_video(
        self, images, folder_path, subfolder_name, filename_prefix, seed, fps, prompt=None, extra_pnginfo=None
    ):
        try:
            # FIX: Clean the subfolder name before joining the path
            cleaned_subfolder = subfolder_name.strip().lstrip('/\\')
            final_dir = os.path.join(folder_path, cleaned_subfolder)
            os.makedirs(final_dir, exist_ok=True)

            # FIX: Clean the filename prefix before using it in the f-string
            cleaned_prefix = filename_prefix.strip().lstrip('/\\')
            base_filename = f"{cleaned_prefix}_{seed}"
            video_filepath = os.path.join(final_dir, f"{base_filename}.mp4")
            counter = 1
            while os.path.exists(video_filepath):
                video_filepath = os.path.join(final_dir, f"{base_filename}_{counter}.mp4")
                counter += 1

            frames = (images.cpu().numpy() * 255).astype(np.uint8)
            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)

            with tempfile.TemporaryDirectory() as temp_dir:
                for i, frame in enumerate(frames):
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(temp_dir, "frame_%06d.png")]

                video_metadata = {}
                if prompt is not None:
                    video_metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    video_metadata.update(extra_pnginfo)

                if video_metadata:
                    metadata_file = os.path.join(temp_dir, "metadata.txt")
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        f.write(f";FFMETADATA1\ncomment={json.dumps(video_metadata)}")
                    ffmpeg_cmd.extend(["-i", metadata_file, "-map_metadata", "1"])

                ffmpeg_cmd.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-crf",
                        "17",
                        "-preset",
                        "fast",
                        "-pix_fmt",
                        "yuv420p",
                        "-loglevel",
                        "error",
                        video_filepath,
                    ]
                )

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    Log.error(f"FFmpeg failed (seed {seed}): {result.stderr}")

            Log.success(f"Video saved: {os.path.basename(video_filepath)}")
        except Exception as e:
            Log.error(f"Failed to save video (seed {seed}): {e}")

    def _slice_conditioning(self, conditioning, index, device=None):
        cond_tensor = conditioning[0][0]
        safe_index = min(index, cond_tensor.shape[0] - 1)
        sliced_tensor = cond_tensor[safe_index : safe_index + 1]
        if device:
            sliced_tensor = sliced_tensor.to(device)
        return [[sliced_tensor, conditioning[0][1]]]

    def _prepare_i2v_conditioning(self, start_image_single, vae, frame_count, width, height):
        s_image = comfy.utils.common_upscale(
            start_image_single[:frame_count].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        image_padded = (
            torch.ones((frame_count, height, width, s_image.shape[-1]), device=s_image.device, dtype=s_image.dtype)
            * 0.5
        )
        image_padded[: s_image.shape[0]] = s_image
        concat_latent_image = vae.encode(image_padded[:, :, :, :3])

        latent_length = ((frame_count - 1) // 4) + 1
        mask = torch.ones(
            (1, 1, latent_length, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=concat_latent_image.device,
            dtype=s_image.dtype,
        )
        mask[:, :, : ((s_image.shape[0] - 1) // 4) + 1] = 0.0

        return {"concat_latent_image": concat_latent_image, "concat_mask": mask}

    def _pad_conditioning(self, cond, batch_size):
        if cond[0][0].shape[0] >= batch_size:
            return [[cond[0][0][:batch_size], cond[0][1]]]
        last = cond[0][0][-1:].repeat(batch_size - cond[0][0].shape[0], 1, 1)
        return [[torch.cat([cond[0][0], last], dim=0), cond[0][1]]]

    def sample(
        self,
        model_high_noise,
        model_low_noise,
        positive,
        negative,
        width,
        height,
        frame_count,
        batch_count,
        seed,
        increment_seed,
        steps,
        boundary,
        cfg_high_noise,
        cfg_low_noise,
        sampler_name,
        scheduler,
        sigma_shift,
        enable_vae_decode,
        enable_vae_tiled_decode,
        tile_size,
        tile_overlap,
        temporal_tile_size,
        temporal_tile_overlap,
        create_comparison_grid,
        save_videos_images,
        save_folder_path,
        save_subfolder_name,
        save_filename_prefix,
        fps,
        processing_mode,
        offload_conditioning,
        vae=None,
        start_image=None,
        prompt=None,
        extra_pnginfo=None,
    ):

        force_model_unload = True

        Log.header("CRT WAN BATCH SAMPLER")
        Log.vram("Start")

        # I2V setup
        use_i2v = start_image is not None
        i2v_encoded_cache = []

        if use_i2v:
            if vae is None:
                Log.error("VAE required for Image-to-Video mode")
                return (None,) * 5

            Log.info("I2V mode enabled")
            num_start_images = start_image.shape[0]

            if num_start_images == batch_count:
                start_images_list = [start_image[i : i + 1] for i in range(batch_count)]
            elif num_start_images == 1:
                start_images_list = [start_image] * batch_count
            else:
                Log.error(f"Start image batch ({num_start_images}) must be 1 or match batch_count ({batch_count})")
                return (None,) * 5

            _, img_height, img_width, _ = start_images_list[0].shape
            width, height = img_width, img_height
            Log.info(f"Dimensions set from start image: {width}x{height}")

        # Quantize dimensions
        quantized_width = (width // 16) * 16
        quantized_height = (height // 16) * 16
        if quantized_width != width or quantized_height != height:
            Log.info(f"Dimensions quantized: {width}x{height} â†’ {quantized_width}x{quantized_height}")

        # Pre-encode I2V images
        if use_i2v:
            Log.info("Pre-encoding I2V start images...")
            for idx, start_img in enumerate(start_images_list):
                i2v_cond = self._prepare_i2v_conditioning(
                    start_img, vae, frame_count, quantized_width, quantized_height
                )
                i2v_encoded_cache.append({k: v.cpu() for k, v in i2v_cond.items()})
                del i2v_cond
            comfy.model_management.soft_empty_cache()
            Log.success(f"Pre-encoded {len(i2v_encoded_cache)} I2V images â†’ CPU")
            Log.vram("After I2V Encoding")

        # Offload conditioning
        if offload_conditioning:
            Log.info("Offloading text conditioning â†’ CPU")
            positive_offloaded = [[positive[0][0].cpu(), positive[0][1]]]
            negative_offloaded = [[negative[0][0].cpu(), negative[0][1]]]
        else:
            positive_offloaded = positive
            negative_offloaded = negative

        # Setup
        latent_temporal_depth = ((frame_count - 1) // 4) + 1
        base_latent = torch.zeros(
            [1, 16, latent_temporal_depth, quantized_height // 8, quantized_width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        seeds = [seed + i if increment_seed else seed for i in range(batch_count)]

        # Calculate boundary switching step
        temp_model = set_shift(model_high_noise.clone(), sigma_shift)
        sigmas = comfy.samplers.calculate_sigmas(temp_model.get_model_object("model_sampling"), scheduler, steps)
        switching_step = steps
        for j, t in enumerate(sigmas[1:], 1):
            if t < boundary:
                switching_step = j - 1
                break
        del temp_model
        Log.info(f"Boundary {boundary:.3f} â†’ switch at step {switching_step}/{steps}")

        # Determine processing mode
        latents_for_handoff_list = []
        final_latent_list = []
        run_sequential = processing_mode == 'Sequential' or use_i2v or offload_conditioning

        # PARALLEL MODE
        if not run_sequential:
            Log.header("PARALLEL BATCH MODE")

            pos_batch = self._pad_conditioning(positive, batch_count)
            neg_batch = self._pad_conditioning(negative, batch_count)
            noise = torch.cat([comfy.sample.prepare_noise(base_latent, s) for s in seeds], dim=0)

            Log.info(f"Processing {batch_count} items in parallel...")
            Log.vram("Before High-Noise")

            # HIGH-NOISE PHASE
            Log.header("PHASE 1: HIGH-NOISE BATCH")
            mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
            Log.vram("After High-Noise Model Load")

            latents_for_handoff = comfy.sample.sample(
                mh_clone,
                noise,
                steps,
                cfg_high_noise,
                sampler_name,
                scheduler,
                pos_batch,
                neg_batch,
                base_latent.repeat(batch_count, 1, 1, 1, 1),
                start_step=0,
                last_step=switching_step,
                force_full_denoise=True,
                disable_noise=(switching_step >= steps),
                callback=latent_preview.prepare_callback(mh_clone, switching_step),
            )
            Log.vram("After High-Noise Sampling")

            del mh_clone
            if force_model_unload:
                Log.info("Force unloading high-noise model...")
                comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            Log.vram("After High-Noise Cleanup")
            Log.success("High-noise batch complete")

            # LOW-NOISE PHASE
            if switching_step < steps:
                Log.header("PHASE 2: LOW-NOISE BATCH")
                Log.vram("Before Low-Noise Model Load")

                ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                Log.vram("After Low-Noise Model Load")

                final_latents = comfy.sample.sample(
                    ml_clone,
                    noise,
                    steps,
                    cfg_low_noise,
                    sampler_name,
                    scheduler,
                    pos_batch,
                    neg_batch,
                    latents_for_handoff,
                    start_step=switching_step,
                    last_step=steps,
                    callback=latent_preview.prepare_callback(ml_clone, steps - switching_step),
                )
                Log.vram("After Low-Noise Sampling (PEAK)")

                del ml_clone
                comfy.model_management.soft_empty_cache()
                Log.vram("After Low-Noise Cleanup")
                Log.success("Low-noise batch complete")
            else:
                Log.info("Switching at max step, skipping low-noise phase")
                final_latents = latents_for_handoff

            latents_for_handoff_list = [latents_for_handoff[i : i + 1] for i in range(batch_count)]
            final_latent_list = [final_latents[i : i + 1] for i in range(batch_count)]

        # SEQUENTIAL MODE
        else:
            mode_reason = "I2V" if use_i2v else ("Offload" if offload_conditioning else "Sequential")
            if processing_mode == 'Parallel':
                Log.header(f"SEQUENTIAL MODE (Fallback: {mode_reason})")
            else:
                Log.header("SEQUENTIAL MODE")

            Log.vram("Before High-Noise")

            # HIGH-NOISE PHASE
            Log.header("PHASE 1: HIGH-NOISE (ALL ITEMS)")
            mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
            Log.vram("After High-Noise Model Load")

            for i in range(batch_count):
                Log.info(f"High-noise pass {i+1}/{batch_count} (Seed: {seeds[i]})")
                device = comfy.model_management.get_torch_device()

                pos = self._slice_conditioning(positive_offloaded, i, device if offload_conditioning else None)
                neg = self._slice_conditioning(negative_offloaded, i, device if offload_conditioning else None)

                if use_i2v:
                    i2v_values = {k: v.to(device) for k, v in i2v_encoded_cache[i].items()}
                    pos = conditioning_set_values(pos, i2v_values)
                    neg = conditioning_set_values(neg, i2v_values)

                noise = comfy.sample.prepare_noise(base_latent, seeds[i])
                latent_handoff = comfy.sample.sample(
                    mh_clone,
                    noise,
                    steps,
                    cfg_high_noise,
                    sampler_name,
                    scheduler,
                    pos,
                    neg,
                    base_latent,
                    start_step=0,
                    last_step=switching_step,
                    force_full_denoise=True,
                    disable_noise=(switching_step >= steps),
                    callback=latent_preview.prepare_callback(mh_clone, switching_step),
                )
                latents_for_handoff_list.append(latent_handoff)

                if i == 0:
                    Log.vram("After First High-Noise Item")

            Log.vram("After All High-Noise Items")
            del mh_clone
            if force_model_unload:
                Log.info("Force unloading high-noise model...")
                comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            Log.vram("After High-Noise Cleanup")
            Log.success("All high-noise passes complete")

            # LOW-NOISE PHASE
            if switching_step < steps:
                Log.header("PHASE 2: LOW-NOISE (ALL ITEMS)")
                Log.vram("Before Low-Noise Model Load")

                ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                Log.vram("After Low-Noise Model Load")

                for i in range(batch_count):
                    Log.info(f"Low-noise pass {i+1}/{batch_count} (Seed: {seeds[i]})")
                    device = comfy.model_management.get_torch_device()

                    pos = self._slice_conditioning(positive_offloaded, i, device if offload_conditioning else None)
                    neg = self._slice_conditioning(negative_offloaded, i, device if offload_conditioning else None)

                    if use_i2v:
                        i2v_values = {k: v.to(device) for k, v in i2v_encoded_cache[i].items()}
                        pos = conditioning_set_values(pos, i2v_values)
                        neg = conditioning_set_values(neg, i2v_values)

                    noise = comfy.sample.prepare_noise(base_latent, seeds[i])
                    final_latent = comfy.sample.sample(
                        ml_clone,
                        noise,
                        steps,
                        cfg_low_noise,
                        sampler_name,
                        scheduler,
                        pos,
                        neg,
                        latents_for_handoff_list[i],
                        start_step=switching_step,
                        last_step=steps,
                        callback=latent_preview.prepare_callback(ml_clone, steps - switching_step),
                    )
                    final_latent_list.append(final_latent)

                    if i == 0:
                        Log.vram("After First Low-Noise Item (PEAK)")

                Log.vram("After All Low-Noise Items")
                del ml_clone
                comfy.model_management.soft_empty_cache()
                Log.vram("After Low-Noise Cleanup")
                Log.success("All low-noise passes complete")
            else:
                Log.info("Switching at max step, skipping low-noise phase")
                final_latent_list = latents_for_handoff_list

        # DECODE & SAVE
        decoded_images_list = []
        if (enable_vae_decode or enable_vae_tiled_decode) and vae:
            Log.header("PHASE 3: VAE DECODING & SAVING")
            Log.vram("Before VAE Decode")

            for i in range(batch_count):
                Log.info(f"Decoding item {i+1}/{batch_count} (Seed: {seeds[i]})")
                latent = final_latent_list[i].to(vae.device)

                if enable_vae_tiled_decode:
                    try:
                        decoded = vae.decode_tiled(
                            latent,
                            tile_x=tile_size // 8,
                            tile_y=tile_size // 8,
                            overlap=tile_overlap // 8,
                            tile_t=temporal_tile_size,
                            overlap_t=temporal_tile_overlap,
                        )
                    except Exception as e:
                        Log.warn(f"Tiled decode failed: {e}. Using standard decode")
                        decoded = vae.decode(latent)
                else:
                    decoded = vae.decode(latent)

                frames = decoded.squeeze(0).cpu()

                if save_videos_images:
                    if frame_count == 1:
                        self._save_image(
                            frames.squeeze(0),
                            save_folder_path,
                            save_subfolder_name,
                            save_filename_prefix,
                            seeds[i],
                            prompt,
                            extra_pnginfo,
                        )
                    else:
                        self._save_video(
                            frames,
                            save_folder_path,
                            save_subfolder_name,
                            save_filename_prefix,
                            seeds[i],
                            fps,
                            prompt,
                            extra_pnginfo,
                        )

                decoded_images_list.append(frames)

            Log.vram("After VAE Decode")
            Log.success("All items decoded")

        # OUTPUT PREPARATION
        high_noise_batch_out = torch.cat(latents_for_handoff_list, dim=0)
        final_latent_batch_out = torch.cat(final_latent_list, dim=0)
        images_out = None
        grid_out = None

        if decoded_images_list:
            images_out = torch.cat(decoded_images_list, dim=0)

            if create_comparison_grid and batch_count > 1:
                Log.info("Creating comparison grid...")
                grid_tensor = torch.stack(decoded_images_list, dim=0)
                n, f, h, w, c = grid_tensor.shape
                grid_out = grid_tensor.permute(1, 2, 0, 3, 4).reshape(f, h, n * w, c)
                Log.success("Comparison grid created")

        settings_str = (
            f"Dims: {quantized_width}x{quantized_height} | Frames: {frame_count} | "
            f"Batches: {batch_count} | Seeds: {seeds[0]}-{seeds[-1] if increment_seed else seeds[0]}"
        )

        Log.vram("Final")
        Log.success("Batch sampling complete")

        return (
            {"samples": high_noise_batch_out},
            {"samples": final_latent_batch_out},
            images_out,
            grid_out,
            settings_str,
        )


NODE_CLASS_MAPPINGS = {"CRT_WAN_BatchSampler": CRT_WAN_BatchSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"CRT_WAN_BatchSampler": "CRT WAN Batch Sampler"}

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
import sys
import cv2
import tempfile
import subprocess
import json
from einops import rearrange
import types
from PIL import Image
from PIL.PngImagePlugin import PngInfo

class Log:
    HEADER = '\033[95m'; CYAN = '\033[96m'; GREEN = '\033[92m'; FAIL = '\033[91m'
    ENDC = '\033[0m'; BOLD = '\033[1m'
    @staticmethod
    def info(message): print(f"{Log.CYAN}[INFO] {message}{Log.ENDC}")
    @staticmethod
    def success(message): print(f"{Log.GREEN}[SUCCESS] {message}{Log.ENDC}")
    @staticmethod
    def fail(message): print(f"{Log.FAIL}[ERROR] {message}{Log.ENDC}")
    @staticmethod
    def header(message): print(f"\n{Log.HEADER}{Log.BOLD}--- {message} ---{Log.ENDC}")

# --- FETA implementation ---
def feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool().unsqueeze(0).expand(attn_temp.shape[0], -1, -1)
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag
    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores

def get_feta_scores(query, key, num_frames, enhance_weight):
    img_q, img_k = query, key
    _, ST, num_heads, head_dim = img_q.shape
    spatial_dim = ST // num_frames
    query_image = rearrange(img_q, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim)
    key_image = rearrange(img_k, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim)
    return feta_score(query_image, key_image, head_dim, num_frames, enhance_weight)

def modified_wan_self_attention_forward(self, x, freqs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v
    q, k, v = qkv_fn(x)
    
    if freqs is not None and hasattr(comfy.ldm.flux.math, 'apply_rope'):
        q, k = comfy.ldm.flux.math.apply_rope(q, k, freqs)
    
    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight)
    
    x = comfy.ldm.modules.attention.optimized_attention(q.view(b, s, n * d), k.view(b, s, n * d), v, heads=self.num_heads)
    x = self.o(x)
    x *= feta_scores
    return x

class WanAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight
        
    def __get__(self, obj, objtype=None):
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return modified_wan_self_attention_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

def set_shift(model, sigma_shift):
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:
        class ModelSampling(comfy.model_sampling.ModelSamplingDiscrete): pass
        model_sampling = ModelSampling(model.model.config)
    try: model_sampling.set_parameters(shift=sigma_shift)
    except AttributeError: model_sampling.shift = sigma_shift
    model.add_object_patch("model_sampling", model_sampling)
    return model

class CRT_WAN_BatchSampler:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "model_high_noise": ("MODEL",), "model_low_noise": ("MODEL",),
                "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 384, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "frame_count": ("INT", {"default": 41, "min": 1, "max": 4096}),
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 128}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "increment_seed": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "enhance_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "enable_vae_decode": ("BOOLEAN", {"default": True}),
                "enable_vae_tiled_decode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_tile_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_tile_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
                "create_comparison_grid": ("BOOLEAN", {"default": True}),
                "save_videos_images": ("BOOLEAN", {"default": True}),
                "save_folder_path": ("STRING", {"default": output_dir}),
                "save_subfolder_name": ("STRING", {"default": "FAST_BATCH"}),
                "save_filename_prefix": ("STRING", {"default": "output"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "low_vram_mode": ("BOOLEAN", {"default": True}),
            }, "optional": { "vae": ("VAE",), },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("high_noise_latent_batch", "final_latent_batch", "final_images_batch", "comparison_grid", "settings_string")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def _apply_enhancement(self, model, weight, latent_temporal_depth):
        Log.info(f"Applying FETA (Enhance-A-Video) patch with weight: {weight} and latent temporal depth: {latent_temporal_depth}")
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanAttentionPatch(latent_temporal_depth, weight).__get__(block.self_attn, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.self_attn.forward", patched_attn)
        return model_clone

    def save_single_image(self, image_tensor, folder_path, subfolder_name, filename_prefix, seed, prompt=None, extra_pnginfo=None):
        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            prefix_clean = filename_prefix.strip().lstrip('/\\')
            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            
            base_filename = f"{prefix_clean}_{seed}"
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
            Log.success(f"Image for seed {seed} saved to: {image_filepath}")

        except Exception as e:
            Log.fail(f"Error saving image for seed {seed}: {e}")

    def save_single_video(self, images, folder_path, subfolder_name, filename_prefix, seed, fps, prompt=None, extra_pnginfo=None):
        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            prefix_clean = filename_prefix.strip().lstrip('/\\')
            final_dir = os.path.join(folder_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)
            base_filename = f"{prefix_clean}_{seed}"
            video_filepath = os.path.join(final_dir, f"{base_filename}.mp4")
            counter = 1
            while os.path.exists(video_filepath):
                video_filepath = os.path.join(final_dir, f"{base_filename}_{counter}.mp4")
                counter += 1
            frames = (images.cpu().numpy() * 255).astype(np.uint8)
            if frames.ndim == 3: frames = np.expand_dims(frames, axis=0)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(temp_dir, "frame_%06d.png")]
                
                video_metadata = {}
                if prompt is not None:
                    video_metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    video_metadata.update(extra_pnginfo)

                if video_metadata:
                    metadata_str = json.dumps(video_metadata)
                    metadata_file = os.path.join(temp_dir, "metadata.txt")
                    escaped_metadata = metadata_str.replace('\\', '\\\\').replace('=', '\\=').replace(';', '\\;').replace('#', '\\#')
                    formatted_metadata = escaped_metadata.replace('\n', '\\\n')
                    
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        f.write(";FFMETADATA1\n")
                        f.write(f"comment={formatted_metadata}")
                    ffmpeg_cmd.extend(["-i", metadata_file])
                
                output_options = ["-c:v", "libx264", "-crf", "3", "-preset", "fast", "-pix_fmt", "yuv420p"]
                
                if video_metadata:
                    output_options.extend(["-map", "0:v", "-map_metadata", "1"])
                
                output_options.append(video_filepath)
                ffmpeg_cmd.extend(output_options)

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    Log.fail(f"FFmpeg failed for seed {seed}: {result.stderr}")
                    return
            Log.success(f"Video for seed {seed} saved to: {video_filepath}")
        except Exception as e:
            Log.fail(f"Error saving video for seed {seed}: {e}")

    def _slice_conditioning(self, conditioning, index):
        cond_prompts = conditioning[0][0]
        num_prompts = cond_prompts.shape[0]
        safe_index = min(index, num_prompts - 1)
        cond_tensor = cond_prompts[safe_index : safe_index + 1]
        extras_dict = conditioning[0][1]
        extras_slice = {}
        if extras_dict is not None and isinstance(extras_dict, dict):
            for key, val in extras_dict.items():
                if isinstance(val, torch.Tensor):
                    num_extra_items = val.shape[0]
                    safe_extra_index = min(index, num_extra_items - 1)
                    extras_slice[key] = val[safe_extra_index : safe_extra_index + 1]
                else:
                    extras_slice[key] = val
        return [[cond_tensor, extras_slice]]

    def sample(self, model_high_noise, model_low_noise, positive, negative, width, height, frame_count, batch_count, seed, increment_seed, steps, boundary, cfg_high_noise, cfg_low_noise, sampler_name, scheduler, sigma_shift, enhance_weight, enable_vae_decode, enable_vae_tiled_decode, tile_size, tile_overlap, temporal_tile_size, temporal_tile_overlap, create_comparison_grid, save_videos_images, save_folder_path, save_subfolder_name, save_filename_prefix, fps, low_vram_mode, vae=None, prompt=None, extra_pnginfo=None):
        
        actual_batch_size = batch_count
        Log.info(f"Batch count set to: {actual_batch_size}.")
        quantized_width = (width // 16) * 16
        quantized_height = (height // 16) * 16
        if quantized_width != width or quantized_height != height:
            Log.info(f"Adjusting dimensions to be multiples of 16: {width}x{height} -> {quantized_width}x{quantized_height}")
        
        # FIX: If frame_count is 1, force enhance_weight to 0 to prevent errors and useless processing.
        if frame_count == 1 and enhance_weight > 0.0:
            Log.info(f"Frame count is 1. FETA enhance weight automatically set to 0 (was {enhance_weight}).")
            enhance_weight = 0.0
        
        latent_temporal_depth = ((frame_count - 1) // 4) + 1
        
        if enhance_weight > 0.0:
            model_high_noise = self._apply_enhancement(model_high_noise, enhance_weight, latent_temporal_depth)
            model_low_noise = self._apply_enhancement(model_low_noise, enhance_weight, latent_temporal_depth)

        base_latent = torch.zeros([1, 16, latent_temporal_depth, quantized_height // 8, quantized_width // 8], device=comfy.model_management.intermediate_device())
        seeds = [seed + i if increment_seed else seed for i in range(actual_batch_size)]
        
        temp_model = set_shift(model_high_noise.clone(), sigma_shift)
        sigmas = comfy.samplers.calculate_sigmas(temp_model.get_model_object("model_sampling"), scheduler, steps)
        switching_step = steps
        for j, t in enumerate(sigmas[1:], 1):
            if t < boundary:
                switching_step = j - 1
                break
        del temp_model
        Log.info(f"Boundary {boundary} corresponds to a switch AT step {switching_step}/{steps}.")

        latents_for_handoff_list, final_latent_list, decoded_images_list = [], [], []

        if not low_vram_mode:
            Log.header("RUNNING IN HIGH VRAM (FAST BATCH) MODE")
            def pad_conditioning(conditioning, target_size):
                cond_tensor = conditioning[0][0]
                num_conds = cond_tensor.shape[0]
                if num_conds >= target_size: return [[cond_tensor[:target_size], conditioning[0][1]]]
                last_cond = cond_tensor[-1:]
                repeats_needed = target_size - num_conds
                padded_cond = torch.cat([cond_tensor, last_cond.repeat(repeats_needed, 1, 1)], dim=0)
                padded_extras = {}
                extras = conditioning[0][1]
                if extras is not None:
                    for key, val in extras.items():
                        if isinstance(val, torch.Tensor):
                            last_extra = val[-1:]
                            padded_extras[key] = torch.cat([val, last_extra.repeat(repeats_needed, 1)], dim=0)
                        else: padded_extras[key] = val
                return [[padded_cond, padded_extras]]
            positive, negative = pad_conditioning(positive, actual_batch_size), pad_conditioning(negative, actual_batch_size)
            latents = base_latent.repeat(actual_batch_size, 1, 1, 1, 1)
            noise_batch = torch.cat([comfy.sample.prepare_noise(base_latent, s) for s in seeds], dim=0)
            Log.header("PHASE 1: HIGH-NOISE BATCH PROCESSING")
            mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
            latents_for_handoff_batch = comfy.sample.sample(mh_clone, noise_batch, steps, cfg_high_noise, sampler_name, scheduler, positive, negative, latents, start_step=0, last_step=switching_step, force_full_denoise=True, disable_noise=(switching_step >= steps), callback=latent_preview.prepare_callback(mh_clone, switching_step))
            del mh_clone; comfy.model_management.soft_empty_cache()
            Log.header("PHASE 2: LOW-NOISE BATCH PROCESSING")
            final_latent_batch = latents_for_handoff_batch
            if switching_step < steps:
                ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                final_latent_batch = comfy.sample.sample(ml_clone, noise_batch, steps, cfg_low_noise, sampler_name, scheduler, positive, negative, latents_for_handoff_batch, start_step=switching_step, last_step=steps, callback=latent_preview.prepare_callback(ml_clone, steps - switching_step))
                del ml_clone; comfy.model_management.soft_empty_cache()
            latents_for_handoff_list = [latents_for_handoff_batch[i].unsqueeze(0) for i in range(actual_batch_size)]
            final_latent_list = [final_latent_batch[i].unsqueeze(0) for i in range(actual_batch_size)]
        else:
            Log.header("RUNNING IN LOW VRAM (SEQUENTIAL) MODE")
            Log.header("PHASE 1: HIGH-NOISE (SEQUENTIAL)")
            mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
            pbar_high = comfy.utils.ProgressBar(actual_batch_size)
            for i in range(actual_batch_size):
                Log.info(f"High-noise pass for video {i+1}/{actual_batch_size} (Seed: {seeds[i]})")
                noise = comfy.sample.prepare_noise(base_latent, seeds[i])
                positive_single, negative_single = self._slice_conditioning(positive, i), self._slice_conditioning(negative, i)
                latent_for_handoff = comfy.sample.sample(mh_clone, noise, steps, cfg_high_noise, sampler_name, scheduler, positive_single, negative_single, base_latent, start_step=0, last_step=switching_step, force_full_denoise=True, disable_noise=(switching_step >= steps), callback=latent_preview.prepare_callback(mh_clone, switching_step))
                latents_for_handoff_list.append(latent_for_handoff)
                pbar_high.update(1)
            del mh_clone; comfy.model_management.soft_empty_cache()
            Log.success("All high-noise passes complete.")
            Log.header("PHASE 2: LOW-NOISE (SEQUENTIAL)")
            if switching_step < steps:
                ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                pbar_low = comfy.utils.ProgressBar(actual_batch_size)
                for i in range(actual_batch_size):
                    Log.info(f"Low-noise pass for video {i+1}/{actual_batch_size} (Seed: {seeds[i]})")
                    noise = comfy.sample.prepare_noise(base_latent, seeds[i])
                    positive_single, negative_single = self._slice_conditioning(positive, i), self._slice_conditioning(negative, i)
                    latent_for_handoff = latents_for_handoff_list[i]
                    final_latent = comfy.sample.sample(ml_clone, noise, steps, cfg_low_noise, sampler_name, scheduler, positive_single, negative_single, latent_for_handoff, start_step=switching_step, last_step=steps, callback=latent_preview.prepare_callback(ml_clone, steps - switching_step))
                    final_latent_list.append(final_latent)
                    pbar_low.update(1)
                del ml_clone; comfy.model_management.soft_empty_cache()
                Log.success("All low-noise passes complete.")
            else:
                Log.info("Switching step is at max; skipping low-noise phase.")
                final_latent_list = latents_for_handoff_list

        should_decode = (enable_vae_decode or enable_vae_tiled_decode) and vae
        if should_decode:
            Log.header("PHASE 3: VAE DECODING & SAVING")
            pbar_decode = comfy.utils.ProgressBar(actual_batch_size)
            for i in range(actual_batch_size):
                latent_video = final_latent_list[i]
                
                if enable_vae_tiled_decode:
                    Log.info(f"Decoding video {i+1}/{actual_batch_size} (Seed: {seeds[i]}) using TILED VAE decode.")
                    try:
                        decoded_video = vae.decode_tiled(
                            latent_video.to(vae.device),
                            tile_x=tile_size // 8,
                            tile_y=tile_size // 8,
                            overlap=tile_overlap // 8,
                            tile_t=temporal_tile_size,
                            overlap_t=temporal_tile_overlap
                        )
                    except Exception as e:
                        Log.fail(f"Tiled VAE Decode failed with error: '{e}'. Falling back to standard decode.")
                        Log.info(f"Decoding video {i+1}/{actual_batch_size} (Seed: {seeds[i]}) using FULL VAE decode (fallback).")
                        decoded_video = vae.decode(latent_video.to(vae.device))
                else:
                    Log.info(f"Decoding video {i+1}/{actual_batch_size} (Seed: {seeds[i]}) using FULL VAE decode.")
                    decoded_video = vae.decode(latent_video.to(vae.device))

                video_frames_cpu = decoded_video.squeeze(0).to(torch.device('cpu'))

                if save_videos_images:
                    if frame_count == 1:
                        single_frame_cpu = video_frames_cpu.squeeze(0)
                        self.save_single_image(single_frame_cpu, save_folder_path, save_subfolder_name, save_filename_prefix, seeds[i], prompt, extra_pnginfo)
                    else:
                        self.save_single_video(video_frames_cpu, save_folder_path, save_subfolder_name, save_filename_prefix, seeds[i], fps, prompt, extra_pnginfo)
                
                decoded_images_list.append(video_frames_cpu)
                pbar_decode.update(1)
            Log.success("All videos decoded.")

        high_noise_batch_out = torch.cat(latents_for_handoff_list, dim=0)
        final_latent_batch_out = torch.cat(final_latent_list, dim=0)
        final_images_batch, comparison_grid = None, None
        
        if decoded_images_list:
            final_images_batch = torch.cat(decoded_images_list, dim=0)
            if create_comparison_grid and actual_batch_size > 0:
                Log.info("Creating comparison grid...")
                all_videos_tensor = torch.stack(decoded_images_list, dim=0)
                n, f, h, w, c = all_videos_tensor.shape
                permuted_tensor = all_videos_tensor.permute(1, 2, 0, 3, 4)
                comparison_grid = permuted_tensor.reshape(f, h, n * w, c)
                Log.success("Comparison grid created successfully.")

        settings_string = f"Dimensions: {quantized_width}x{quantized_height} | Frames: {frame_count} | Batch Size: {actual_batch_size} | Seed: {seed} | Steps: {steps}"
        return ({"samples": high_noise_batch_out}, {"samples": final_latent_batch_out}, final_images_batch, comparison_grid, settings_string)
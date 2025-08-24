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
from PIL import Image, ImageDraw, ImageFont

# --- UTILITY FOR CLEAR, COLORED LOGGING ---
class Log:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def info(message): print(f"{Log.CYAN}{message}{Log.ENDC}")
    @staticmethod
    def success(message): print(f"{Log.GREEN}{message}{Log.ENDC}")
    @staticmethod
    def fail(message): print(f"{Log.FAIL}{message}{Log.ENDC}")
    @staticmethod
    def header(message): print(f"\n{Log.HEADER}{Log.BOLD}{message}{Log.ENDC}")

def find_lora_path_by_name(lora_name):
    """
    Fixed LoRA path finding to handle subdirectories properly
    """
    all_loras = folder_paths.get_filename_list("loras")
    
    # First try exact match (for subdirectory paths like "WAN2.2\SS_14_HighNoise\SS_14_HighNoise-000005.safetensors")
    if lora_name in all_loras:
        return folder_paths.get_full_path("loras", lora_name)
    
    # Fallback: try basename matching
    lora_map = {os.path.basename(f): f for f in all_loras}
    if lora_name in lora_map:
        relative_path = lora_map[lora_name]
        return folder_paths.get_full_path("loras", relative_path)
    
    # Debug logging
    Log.fail(f"LoRA not found: {lora_name}")
    Log.info(f"Available LoRAs with 'WAN2.2': {[l for l in all_loras if 'WAN2.2' in l]}")
    return None

def parse_lora_configs(config_string: str):
    configs = []
    lines = [line.strip() for line in config_string.strip().split('\n') if line.strip()]
    for i, line in enumerate(lines):
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4: continue
        high_lora, low_lora, high_str, low_str = parts
        try:
            configs.append({ 
                "high_name": high_lora if high_lora.lower() not in ['none', ''] else None, 
                "low_name": low_lora if low_lora.lower() not in ['none', ''] else None, 
                "high_strength": float(high_str), 
                "low_strength": float(low_str) 
            })
        except ValueError: continue
    return configs

def parse_label_configs(label_string: str):
    """Parse label configuration string into list of labels"""
    if not label_string.strip():
        return []
    
    labels = []
    lines = [line.strip() for line in label_string.strip().split('\n')]
    for line in lines:
        if line:  # Only add non-empty lines
            labels.append(line)
    return labels

def get_fallback_label(low_lora_name):
    """Generate fallback label from LoRA filename"""
    if not low_lora_name:
        return "No LoRA"
    
    # Remove .safetensors extension and path
    label = os.path.basename(low_lora_name)
    if label.endswith('.safetensors'):
        label = label[:-12]  # Remove .safetensors
    elif label.endswith('.ckpt'):
        label = label[:-5]   # Remove .ckpt
    
    return label

def tensor_to_pil(tensor_image):
    """Convert tensor to PIL Image"""
    # tensor_image should be (H, W, C) in range [0, 1]
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)
    
    # Convert to numpy and scale to 0-255
    np_image = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)  # Add batch dimension

def add_label_to_image(image_tensor, label_text, font_size=24):
    """Add text label to the bottom of an image"""
    pil_image = tensor_to_pil(image_tensor)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Create a temporary drawer to measure text size
    draw = ImageDraw.Draw(pil_image)
    
    # Calculate a FIXED height for the label area based on a standard character.
    _, top, _, bottom = draw.textbbox((0, 0), "Ag", font=font)
    fixed_text_height = bottom - top
    label_height = fixed_text_height + 20  # Add consistent padding

    # Now, measure the actual width of the current label for centering
    left, _, right, _ = draw.textbbox((0, 0), label_text, font=font)
    text_width = right - left

    # Create new image with space for label
    old_width, old_height = pil_image.size
    new_height = old_height + label_height
    
    labeled_image = Image.new('RGB', (old_width, new_height), (40, 40, 40))  # Dark gray background
    labeled_image.paste(pil_image, (0, 0))
    
    # Draw the label text
    draw = ImageDraw.Draw(labeled_image)
    text_x = (old_width - text_width) // 2  # Center text horizontally
    text_y = old_height + (label_height - fixed_text_height) // 2  # Center vertically in label area
    
    draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
    
    return pil_to_tensor(labeled_image)

def set_shift(model, sigma_shift):
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:
        class ModelSampling(comfy.model_sampling.ModelSamplingDiscrete): pass
        model_sampling = ModelSampling(model.model.model_config)
    try: model_sampling.set_parameters(shift=sigma_shift)
    except AttributeError: model_sampling.shift = sigma_shift
    model.add_object_patch("model_sampling", model_sampling)
    return model

class WAN2_2LoraCompareSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high_noise": ("MODEL",),
                "width": ("INT", {"default": 432, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 768, "min": 16, "max": 4096, "step": 16}),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 4096}),
                "model_low_noise": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "seed": ("INT", {"forceInput": True}),
                "lora_batch_config": ("STRING", { "multiline": True, "default": "none, none, 0.0, 0.0\n" }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,), "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "enable_vae_decode": ("BOOLEAN", {"default": False}),
                "create_comparison_grid": ("BOOLEAN", {"default": False}),
                "add_labels": ("BOOLEAN", {"default": False}),
                "custom_labels": ("STRING", { "multiline": True, "default": "" }),
                "label_font_size": ("INT", {"default": 24, "min": 8, "max": 72}),
            },
            "optional": { "vae": ("VAE",), }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("high_noise_latent_batch", "final_latent_batch", "final_images_batch", "comparison_grid", "settings_string")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(self, model_high_noise, width, height, frame_count, model_low_noise, positive, negative, seed, lora_batch_config, steps, boundary, cfg_high_noise, cfg_low_noise, sampler_name, scheduler, sigma_shift, enable_vae_decode, create_comparison_grid, add_labels, custom_labels, label_font_size, vae=None):
        denoise = 1.0  # Hardcoded denoise value
        lora_configs = parse_lora_configs(lora_batch_config)
        if not lora_configs: raise ValueError("[CRT] No valid LoRA configurations found.")
        
        # Parse custom labels if provided
        custom_label_list = parse_label_configs(custom_labels) if custom_labels.strip() else []
        
        Log.info(f"Generating empty latent of size {width}x{height} with {frame_count} frames.")
        base_latent = torch.zeros([1, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        # Calculate switching step once
        temp_model = set_shift(model_high_noise.clone(), sigma_shift)
        sampling = temp_model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
        timesteps = [sampling.timestep(sigma) / 1000 for sigma in sigmas.tolist()]
        switching_step = steps
        for j, t in enumerate(timesteps[1:]):
            if t < boundary:
                switching_step = j
                break
        del temp_model
        
        Log.info(f"Boundary {boundary} on scheduler '{scheduler}' corresponds to a switch AT step {switching_step}/{steps}.")

        # =============================================================================
        # PHASE 1: PROCESS ALL HIGH-NOISE PASSES (CACHE LATENTS)
        # =============================================================================
        Log.header("=== PHASE 1: HIGH-NOISE PROCESSING (OPTIMIZED BATCHING) ===")
        
        high_noise_latents = []
        current_high_lora = None
        mh_clone = None
        
        try:
            for i, config in enumerate(lora_configs):
                Log.info(f"--- HIGH-NOISE CONFIG {i+1}/{len(lora_configs)} ---")
                
                if config["high_name"] != current_high_lora:
                    if mh_clone is not None:
                        del mh_clone
                        comfy.model_management.soft_empty_cache()
                    
                    mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
                    
                    if config["high_name"]:
                        Log.info(f"Loading HIGH-NOISE LoRA: {config['high_name']}")
                        lora_path = find_lora_path_by_name(config["high_name"])
                        if lora_path:
                            try:
                                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                                mh_clone, _ = comfy.sd.load_lora_for_models(mh_clone, None, lora_data, config["high_strength"], config["high_strength"])
                                Log.success(f"Applied HIGH-NOISE LoRA with strength {config['high_strength']}")
                            except Exception as e:
                                Log.fail(f"Failed to load HIGH-NOISE LoRA: {config['high_name']} - {str(e)}")
                        else: 
                            Log.fail(f"Could not find HIGH-NOISE LoRA: {config['high_name']}")
                    else:
                        Log.info("No HIGH-NOISE LoRA for this config")
                    
                    current_high_lora = config["high_name"]

                if mh_clone is None:
                    mh_clone = set_shift(model_high_noise.clone(), sigma_shift)

                current_seed = seed
                noise = comfy.sample.prepare_noise(base_latent, current_seed)
                will_do_low_noise_pass = switching_step < steps
                
                Log.info(f"Running high-noise inference from step 0 TO {switching_step}...")
                callback_high = latent_preview.prepare_callback(mh_clone, steps)
                    
                latent_for_handoff = comfy.sample.sample(
                    mh_clone, noise, steps, cfg_high_noise, sampler_name, scheduler, positive, negative, base_latent,
                    denoise=denoise, start_step=0, last_step=switching_step, force_full_denoise=True,
                    disable_noise=will_do_low_noise_pass, callback=callback_high, seed=current_seed
                )
                
                high_noise_latents.append({
                    'latent': latent_for_handoff.clone(),
                    'noise': noise.clone(),
                    'seed': current_seed
                })
                Log.success(f"Cached high-noise latent for config {i+1}")

        finally:
            if mh_clone is not None:
                del mh_clone
                comfy.model_management.soft_empty_cache()

        # =============================================================================
        # PHASE 2: PROCESS ALL LOW-NOISE PASSES (USING CACHED LATENTS)
        # =============================================================================
        Log.header("=== PHASE 2: LOW-NOISE PROCESSING (USING CACHED LATENTS) ===")
        
        final_latents_for_output = []
        decoded_images_for_output = []
        current_low_lora = None
        ml_clone = None
        
        try:
            for i, config in enumerate(lora_configs):
                Log.info(f"--- LOW-NOISE CONFIG {i+1}/{len(lora_configs)} ---")
                
                if config["low_name"] != current_low_lora:
                    if ml_clone is not None:
                        del ml_clone
                        comfy.model_management.soft_empty_cache()
                    
                    ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                    
                    if config["low_name"]:
                        Log.info(f"Loading LOW-NOISE LoRA: {config['low_name']}")
                        lora_path = find_lora_path_by_name(config["low_name"])
                        if lora_path:
                            try:
                                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                                ml_clone, _ = comfy.sd.load_lora_for_models(ml_clone, None, lora_data, config["low_strength"], config["low_strength"])
                                Log.success(f"Applied LOW-NOISE LoRA with strength {config['low_strength']}")
                            except Exception as e:
                                Log.fail(f"Failed to load LOW-NOISE LoRA: {config['low_name']} - {str(e)}")
                        else: 
                            Log.fail(f"Could not find LOW-NOISE LoRA: {config['low_name']}")
                    else:
                        Log.info("No LOW-NOISE LoRA for this config")
                    
                    current_low_lora = config["low_name"]

                if ml_clone is None:
                    ml_clone = set_shift(model_low_noise.clone(), sigma_shift)

                cached_data = high_noise_latents[i]
                latent_for_handoff = cached_data['latent']
                noise = cached_data['noise']
                current_seed = cached_data['seed']
                
                Log.info(f"Running low-noise polishing FROM step {switching_step} TO {steps}...")
                if switching_step < steps:
                    callback_low = latent_preview.prepare_callback(ml_clone, steps)
                        
                    final_latent = comfy.sample.sample(
                        ml_clone, noise, steps, cfg_low_noise, sampler_name, scheduler, positive, negative, latent_for_handoff,
                        denoise=denoise, start_step=switching_step, last_step=steps, 
                        disable_noise=False, force_full_denoise=True, 
                        callback=callback_low, seed=current_seed
                    )
                else: 
                    final_latent = latent_for_handoff

                final_latents_for_output.append(final_latent)

                if enable_vae_decode and vae is not None:
                    Log.info("VAE Decode is enabled, decoding current latent to image...")
                    images = vae.decode(final_latent.to(vae.device))
                    if len(images.shape) == 5:
                        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    decoded_images_for_output.append(images.to(torch.device('cpu')))
                elif enable_vae_decode and vae is None:
                    Log.fail("VAE Decode was enabled, but no VAE was connected. Skipping decode.")

                Log.success(f"--- CONFIG {i+1} PROCESSED SUCCESSFULLY ---")

        finally:
            if ml_clone is not None:
                del ml_clone
                comfy.model_management.soft_empty_cache()

        # =============================================================================
        # PREPARE OUTPUTS
        # =============================================================================
        
        high_noise_latents_for_output = [cached['latent'] for cached in high_noise_latents]
        
        high_noise_batch = {"samples": torch.cat(high_noise_latents_for_output, dim=0)}
        final_latent_batch = {"samples": torch.cat(final_latents_for_output, dim=0)}
        
        final_images_batch = None
        comparison_grid = None
        
        if decoded_images_for_output:
            Log.info("Concatenating image batches (preserving sequence structure)...")
            final_images_batch = torch.cat(decoded_images_for_output, dim=0)
            
            Log.info(f"Final batch structure:")
            for i in range(len(decoded_images_for_output)):
                start_idx = i * frame_count
                end_idx = (i + 1) * frame_count - 1
                Log.info(f"  Config {i}: images {start_idx} to {end_idx}")
                
            if create_comparison_grid:
                Log.info("Creating side-by-side comparison grid...")
                comparison_frames = []
                
                config_labels = []
                for i, config in enumerate(lora_configs):
                    if i < len(custom_label_list) and custom_label_list[i].strip():
                        label = custom_label_list[i].strip()
                    else:
                        label = get_fallback_label(config["low_name"])
                    config_labels.append(label)
                
                Log.info(f"Configuration labels: {config_labels}")
                
                for frame_idx in range(frame_count):
                    frame_row = []
                    for config_idx in range(len(decoded_images_for_output)):
                        img_idx = config_idx * frame_count + frame_idx
                        frame_img = final_images_batch[img_idx]
                        
                        if add_labels:
                            label_text = config_labels[config_idx]
                            frame_img = add_label_to_image(frame_img, label_text, label_font_size)
                            frame_img = frame_img.squeeze(0)
                        
                        frame_img = frame_img.unsqueeze(0)
                        frame_row.append(frame_img)
                    
                    comparison_frame = torch.cat(frame_row, dim=2)
                    comparison_frames.append(comparison_frame)
                
                comparison_grid = torch.cat(comparison_frames, dim=0)
                
                if len(comparison_grid.shape) == 4:
                    Log.success(f"Comparison grid created: {comparison_grid.shape} ({len(lora_configs)} configs side-by-side, {frame_count} frames)")
                    if add_labels:
                        Log.success(f"Labels added to comparison grid with font size {label_font_size}")
                else:
                    Log.fail(f"Unexpected comparison grid shape: {comparison_grid.shape}")
                    comparison_grid = None

        Log.success(f"\nAll configurations processed. Final image batch shape: {final_images_batch.shape if final_images_batch is not None else 'No images decoded'}")
        Log.info(f"To select config N, use batch_index={frame_count}*N and length={frame_count} in ImageFromBatch")
        
        settings_string = (
            f"Dimensions: {width}x{height} | Frames: {frame_count}\n"
            f"Seed: {seed} | Steps: {steps} | Boundary: {boundary}\n"
            f"Sampler: {sampler_name} | Scheduler: {scheduler}\n"
            f"Sigma Shift: {sigma_shift}"
        )

        return ({"samples": high_noise_batch["samples"]}, {"samples": final_latent_batch["samples"]}, final_images_batch, comparison_grid, settings_string)
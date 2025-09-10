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

    all_loras = folder_paths.get_filename_list("loras")
    
    if lora_name in all_loras:
        return folder_paths.get_full_path("loras", lora_name)
    
    lora_map = {os.path.basename(f): f for f in all_loras}
    if lora_name in lora_map:
        relative_path = lora_map[lora_name]
        return folder_paths.get_full_path("loras", relative_path)
    
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
    if not label_string.strip(): return []
    return [line.strip() for line in label_string.strip().split('\n') if line.strip()]

def get_fallback_label(lora_name):
    if not lora_name: return "No LoRA"
    label = os.path.basename(lora_name)
    if label.endswith('.safetensors'): label = label[:-12]
    elif label.endswith('.ckpt'): label = label[:-5]
    return label

def tensor_to_pil(tensor_image):
    if tensor_image.dim() == 4: tensor_image = tensor_image.squeeze(0)
    np_image = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)

def add_label_to_image(image_tensor, label_text, font_size=24):
    pil_image = tensor_to_pil(image_tensor)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except: font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(pil_image)
    _, top, _, bottom = draw.textbbox((0, 0), "Ag", font=font)
    fixed_text_height = bottom - top
    label_height = fixed_text_height + 20
    left, _, right, _ = draw.textbbox((0, 0), label_text, font=font)
    text_width = right - left

    old_width, old_height = pil_image.size
    new_height = old_height + label_height
    labeled_image = Image.new('RGB', (old_width, new_height), (40, 40, 40))
    labeled_image.paste(pil_image, (0, 0))
    
    draw = ImageDraw.Draw(labeled_image)
    text_x = (old_width - text_width) // 2
    text_y = old_height + (label_height - fixed_text_height) // 2
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
            }, "optional": { "vae": ("VAE",), }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("high_noise_latent_batch", "final_latent_batch", "final_images_batch", "comparison_grid", "settings_string")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(self, model_high_noise, width, height, frame_count, model_low_noise, positive, negative, seed, lora_batch_config, steps, boundary, cfg_high_noise, cfg_low_noise, sampler_name, scheduler, sigma_shift, enable_vae_decode, create_comparison_grid, add_labels, custom_labels, label_font_size, vae=None):
        denoise = 1.0
        lora_configs = parse_lora_configs(lora_batch_config)
        if not lora_configs: raise ValueError("[CRT] No valid LoRA configurations found.")
        
        custom_label_list = parse_label_configs(custom_labels)
        base_latent = torch.zeros([1, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        
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

        Log.header("=== PHASE 1: HIGH-NOISE PROCESSING (WITH CACHING) ===")
        high_noise_latents = []
        high_noise_cache = {}
        current_high_lora = None
        current_high_strength = None
        mh_clone = None
        
        try:
            for i, config in enumerate(lora_configs):
                Log.info(f"--- HIGH-NOISE CONFIG {i+1}/{len(lora_configs)} ---")
                
                cache_key = (config["high_name"], config["high_strength"])

                if cache_key in high_noise_cache:
                    Log.success(f"Found cached high-noise latent for '{str(cache_key[0])}' @ strength {cache_key[1]}. Reusing.")
                    high_noise_latents.append(high_noise_cache[cache_key])
                    continue

                Log.info(f"No cache entry for '{str(cache_key[0])}' @ strength {cache_key[1]}. Computing new latent.")
                
                if config["high_name"] != current_high_lora or config["high_strength"] != current_high_strength:
                    if mh_clone is not None:
                        del mh_clone
                        comfy.model_management.soft_empty_cache()
                    
                    mh_clone = set_shift(model_high_noise.clone(), sigma_shift)
                    
                    if config["high_name"]:
                        lora_path = find_lora_path_by_name(config["high_name"])
                        if lora_path:
                            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            mh_clone, _ = comfy.sd.load_lora_for_models(mh_clone, None, lora_data, config["high_strength"], config["high_strength"])
                            Log.success(f"Applied HIGH-NOISE LoRA '{config['high_name']}' with strength {config['high_strength']}")
                        else: 
                            Log.fail(f"Could not find HIGH-NOISE LoRA: {config['high_name']}")
                    else:
                        Log.info("No HIGH-NOISE LoRA for this config.")
                    
                    current_high_lora = config["high_name"]
                    current_high_strength = config["high_strength"]

                if mh_clone is None: mh_clone = set_shift(model_high_noise.clone(), sigma_shift)

                noise = comfy.sample.prepare_noise(base_latent, seed)
                latent_for_handoff = comfy.sample.sample(mh_clone, noise, steps, cfg_high_noise, sampler_name, scheduler, positive, negative, base_latent, denoise=denoise, start_step=0, last_step=switching_step, force_full_denoise=True, disable_noise=(switching_step < steps), callback=latent_preview.prepare_callback(mh_clone, steps), seed=seed)
                
                computed_data = {'latent': latent_for_handoff.clone(), 'noise': noise.clone(), 'seed': seed}
                high_noise_cache[cache_key] = computed_data
                high_noise_latents.append(computed_data)
                Log.success(f"Computed and cached high-noise latent for config {i+1}")
        finally:
            if mh_clone is not None:
                del mh_clone
                comfy.model_management.soft_empty_cache()
        
        final_latents_for_output = []
        decoded_images_for_output = []

        # <<< MODIFICATION: Check if low-noise pass should be skipped
        if cfg_low_noise == 0:
            Log.header("=== LOW-NOISE CFG is 0: SKIPPING PHASE 2 ===")
            Log.info("Decoding high-noise latents directly.")
            final_latents_for_output = [cached['latent'] for cached in high_noise_latents]

            if enable_vae_decode and vae:
                for i, latent in enumerate(final_latents_for_output):
                    Log.info(f"--- Decoding high-noise latent for config {i+1}/{len(lora_configs)} ---")
                    images = vae.decode(latent.to(vae.device))
                    if len(images.shape) == 5: images = images.reshape(-1, *images.shape[-3:])
                    decoded_images_for_output.append(images.to(torch.device('cpu')))
                    Log.success(f"--- HIGH-NOISE CONFIG {i+1} DECODED SUCCESSFULLY ---")
            elif enable_vae_decode:
                 Log.fail("VAE Decode enabled, but no VAE connected.")
        else:
            Log.header("=== PHASE 2: LOW-NOISE PROCESSING ===")
            current_low_lora = None
            current_low_strength = None
            ml_clone = None
            
            try:
                for i, config in enumerate(lora_configs):
                    Log.info(f"--- LOW-NOISE CONFIG {i+1}/{len(lora_configs)} ---")
                    
                    if config["low_name"] != current_low_lora or config["low_strength"] != current_low_strength:
                        if ml_clone is not None:
                            del ml_clone
                            comfy.model_management.soft_empty_cache()
                        
                        ml_clone = set_shift(model_low_noise.clone(), sigma_shift)
                        
                        if config["low_name"]:
                            lora_path = find_lora_path_by_name(config["low_name"])
                            if lora_path:
                                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                                ml_clone, _ = comfy.sd.load_lora_for_models(ml_clone, None, lora_data, config["low_strength"], config["low_strength"])
                                Log.success(f"Applied LOW-NOISE LoRA '{config['low_name']}' with strength {config['low_strength']}")
                            else: 
                                Log.fail(f"Could not find LOW-NOISE LoRA: {config['low_name']}")
                        else:
                            Log.info("No LOW-NOISE LoRA for this config.")
                        
                        current_low_lora = config["low_name"]
                        current_low_strength = config["low_strength"]

                    if ml_clone is None: ml_clone = set_shift(model_low_noise.clone(), sigma_shift)

                    cached_data = high_noise_latents[i]
                    latent_for_handoff, noise, current_seed = cached_data['latent'], cached_data['noise'], cached_data['seed']
                    
                    if switching_step < steps:
                        final_latent = comfy.sample.sample(ml_clone, noise, steps, cfg_low_noise, sampler_name, scheduler, positive, negative, latent_for_handoff, denoise=denoise, start_step=switching_step, last_step=steps, disable_noise=False, force_full_denoise=True, callback=latent_preview.prepare_callback(ml_clone, steps), seed=current_seed)
                    else: 
                        final_latent = latent_for_handoff

                    final_latents_for_output.append(final_latent)

                    if enable_vae_decode and vae:
                        images = vae.decode(final_latent.to(vae.device))
                        if len(images.shape) == 5: images = images.reshape(-1, *images.shape[-3:])
                        decoded_images_for_output.append(images.to(torch.device('cpu')))
                    elif enable_vae_decode: Log.fail("VAE Decode enabled, but no VAE connected.")
                    Log.success(f"--- CONFIG {i+1} PROCESSED SUCCESSFULLY ---")
            finally:
                if ml_clone is not None:
                    del ml_clone
                    comfy.model_management.soft_empty_cache()

        high_noise_batch = {"samples": torch.cat([cached['latent'] for cached in high_noise_latents], dim=0)}
        final_latent_batch = {"samples": torch.cat(final_latents_for_output, dim=0)}
        
        final_images_batch, comparison_grid = None, None
        
        if decoded_images_for_output:
            final_images_batch = torch.cat(decoded_images_for_output, dim=0)
            if create_comparison_grid:
                # <<< MODIFICATION: Adjust labels based on whether low-noise pass was skipped
                if cfg_low_noise == 0:
                    Log.info("Using HIGH-NOISE LoRA names for grid labels as low-noise pass was skipped.")
                    config_labels = [custom_label_list[i] if i < len(custom_label_list) and custom_label_list[i] else get_fallback_label(config["high_name"]) for i, config in enumerate(lora_configs)]
                else:
                    config_labels = [custom_label_list[i] if i < len(custom_label_list) and custom_label_list[i] else get_fallback_label(config["low_name"]) for i, config in enumerate(lora_configs)]
                
                comparison_frames = []
                for frame_idx in range(frame_count):
                    frame_row = []
                    for config_idx in range(len(decoded_images_for_output)):
                        img_idx = config_idx * frame_count + frame_idx
                        frame_img = final_images_batch[img_idx]
                        if add_labels:
                            frame_img = add_label_to_image(frame_img, config_labels[config_idx], label_font_size).squeeze(0)
                        frame_row.append(frame_img.unsqueeze(0))
                    comparison_frames.append(torch.cat(frame_row, dim=2))
                comparison_grid = torch.cat(comparison_frames, dim=0)
        
        settings_string = f"Dimensions: {width}x{height} | Frames: {frame_count} | Seed: {seed} | Steps: {steps} | Boundary: {boundary} | Sampler: {sampler_name} | Scheduler: {scheduler} | Sigma Shift: {sigma_shift}"
        return ({"samples": high_noise_batch["samples"]}, {"samples": final_latent_batch["samples"]}, final_images_batch, comparison_grid, settings_string)
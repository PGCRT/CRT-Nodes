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
import hashlib

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
    @staticmethod
    def debug(message): print(f"{Log.CYAN}[DEBUG] {message}{Log.ENDC}")

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

def find_lora_path_by_name(lora_name):
    all_loras = folder_paths.get_filename_list("loras")
    if lora_name in all_loras:
        return folder_paths.get_full_path("loras", lora_name)
    lora_map = {os.path.basename(f): f for f in all_loras}
    if lora_name in lora_map:
        relative_path = lora_map[lora_name]
        return folder_paths.get_full_path("loras", relative_path)
    Log.fail(f"LoRA not found: {lora_name}")
    return None

def parse_lora_configs(config_string: str):
    groups = []
    for line in config_string.strip().split('\n'):
        if not line.strip(): continue
        parts = line.split('§')
        if len(parts) != 2: continue
        
        rows_str, enabled_str = parts
        lora_rows = []
        for row_str in rows_str.split('|'):
            row_parts = row_str.split(',')
            if len(row_parts) != 5: continue
            try:
                lora_rows.append({
                    "high_name": row_parts[0] if row_parts[0] != 'none' else None,
                    "high_strength": float(row_parts[1]),
                    "low_name": row_parts[2] if row_parts[2] != 'none' else None,
                    "low_strength": float(row_parts[3]),
                    "enabled": row_parts[4].lower() == 'true'
                })
            except (ValueError, IndexError): continue
        groups.append({"rows": lora_rows, "enabled": enabled_str.lower() == 'true'})
    return groups

def parse_label_configs(label_string: str):
    if not label_string.strip(): return []
    return [line.strip() for line in label_string.strip().split('\n')]

def get_fallback_label(lora_rows, is_bypassed):
    lines = []
    def format_name(name):
        if not name: return "None"
        label = os.path.basename(name)
        if label.endswith('.safensors'): label = label[:-12]
        return label

    high_loras_in_stack = [lora for lora in lora_rows if lora.get("high_name") and lora.get("enabled")]
    if high_loras_in_stack:
        for lora in high_loras_in_stack:
            lines.append(f"H: {format_name(lora['high_name'])} @ {lora['high_strength']:.2f}")
    
    if is_bypassed:
        lines.append("L: Bypassed")
    else:
        low_loras_in_stack = [lora for lora in lora_rows if lora.get("low_name") and lora.get("enabled")]
        if low_loras_in_stack:
            for lora in low_loras_in_stack:
                lines.append(f"L: {format_name(lora['low_name'])} @ {lora['low_strength']:.2f}")
    
    return "\n".join(lines) if lines else "No LoRAs"

def apply_lora_stack(model, lora_list):
    model_clone = model.clone()
    for lora in lora_list:
        if not lora.get('name') or not lora.get('enabled', True): continue
        lora_path = find_lora_path_by_name(lora['name'])
        if lora_path:
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_clone, _ = comfy.sd.load_lora_for_models(model_clone, None, lora_data, lora['strength'], lora['strength'])
            # --- LOGGING RESTORED ---
            Log.success(f"Applied LoRA '{lora['name']}' @ {lora['strength']}")
    return model_clone

def tensor_to_pil(tensor_image):
    if tensor_image.dim() == 4: tensor_image = tensor_image.squeeze(0)
    return Image.fromarray((tensor_image.cpu().numpy() * 255).astype(np.uint8))

def pil_to_tensor(pil_image):
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

def add_label_to_image(image_tensor, label_text, font_size=24, fixed_label_area_height=None):
    pil_image = tensor_to_pil(image_tensor)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(pil_image)
    current_font_size = font_size
    
    while True:
        font = ImageFont.truetype("arial.ttf", current_font_size) if current_font_size > 8 else ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font, spacing=4)
        text_width = right - left
        text_height = bottom - top
        if text_width <= pil_image.width or current_font_size <= 8:
            break
        current_font_size -= 2
    
    label_area_height = fixed_label_area_height if fixed_label_area_height is not None else text_height + 10
    new_height = pil_image.height + label_area_height
    labeled_image = Image.new('RGB', (pil_image.width, new_height), (40, 40, 40))
    labeled_image.paste(pil_image, (0, 0))
    draw = ImageDraw.Draw(labeled_image)
    text_x = (pil_image.width - text_width) // 2
    text_y = pil_image.height + (label_area_height - text_height) // 2
    draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font, spacing=4)
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

def get_conditioning_hash(conditioning):
    if not isinstance(conditioning, list): return ""
    tensor_bytes = bytearray()
    for item in conditioning:
        if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
            tensor_bytes.extend(item[0].cpu().numpy().tobytes())
    if not tensor_bytes: return ""
    return hashlib.sha256(tensor_bytes).hexdigest()

class WAN2_2LoraCompareSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high_noise": ("MODEL",),
                "model_low_noise": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"forceInput": True}),
                "width": ("INT", {"default": 432, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 768, "min": 16, "max": 4096, "step": 16}),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 4096}),
                "lora_batch_config": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "enable_vae_decode": ("BOOLEAN", {"default": True}),
                "create_comparison_grid": ("BOOLEAN", {"default": True}),
                "add_labels": ("BOOLEAN", {"default": True}),
                "custom_labels": ("STRING", {"multiline": True, "default": ""}),
                "label_font_size": ("INT", {"default": 24, "min": 8, "max": 72}),
            },
            "optional": { 
                "vae": ("VAE",),
                "start_image": ("IMAGE",),
            }
        }
    
    # MODIFICATION START: Added MODEL outputs to return types and names
    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING", "MODEL", "MODEL")
    RETURN_NAMES = ("high_noise_latent_batch", "final_latent_batch", "final_images_batch", "comparison_grid", "settings_string", "model_high_noise_out", "model_low_noise_out")
    # MODIFICATION END

    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(self, model_high_noise, model_low_noise, positive, negative, seed, width, height, frame_count, lora_batch_config, steps, boundary, cfg_high_noise, cfg_low_noise, sampler_name, scheduler, sigma_shift, enable_vae_decode, create_comparison_grid, add_labels, custom_labels, label_font_size, vae=None, start_image=None):
        
        # MODIFICATION START: Initialize variables to hold the last used models
        last_mh_clone = model_high_noise
        last_ml_clone = model_low_noise
        # MODIFICATION END
        
        if start_image is not None:
            if vae is None:
                Log.fail("A VAE is required for Image-to-Video mode but was not provided.")
                return ({"samples": torch.zeros([0, 16, 1, 1, 1])}, {"samples": torch.zeros([0, 16, 1, 1, 1])}, torch.zeros([0, 1, 1, 3]), None, "VAE required for I2V mode.", model_high_noise, model_low_noise)

            Log.info("Start image provided. Switching to Image-to-Video mode.")
            _, img_height, img_width, _ = start_image.shape
            width, height = img_width, img_height
            Log.info(f"Overriding dimensions to {width}x{height} based on start image.")

            s_image = comfy.utils.common_upscale(start_image[:frame_count].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image_padded = torch.ones((frame_count, height, width, s_image.shape[-1]), device=s_image.device, dtype=s_image.dtype) * 0.5
            image_padded[:s_image.shape[0]] = s_image
            
            concat_latent_image = vae.encode(image_padded[:, :, :, :3])
            
            latent_length = ((frame_count - 1) // 4) + 1
            mask = torch.ones((1, 1, latent_length, concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=s_image.device, dtype=s_image.dtype)
            mask[:, :, :((s_image.shape[0] - 1) // 4) + 1] = 0.0

            cond_values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            positive = conditioning_set_values(positive, cond_values)
            negative = conditioning_set_values(negative, cond_values)
        
        all_lora_groups = parse_lora_configs(lora_batch_config)
        lora_groups = [g for g in all_lora_groups if g.get("enabled", True)]

        if not lora_groups:
            Log.fail("All LoRA groups are disabled. Nothing to sample.")
            return ({"samples": torch.zeros([0, 4, 1, 1])}, {"samples": torch.zeros([0, 4, 1, 1])}, torch.zeros([0, 1, 1, 3]), None, "All LoRA groups disabled.", model_high_noise, model_low_noise)

        positive_hash = get_conditioning_hash(positive)
        negative_hash = get_conditioning_hash(negative)
        custom_label_list = [label for i, label in enumerate(parse_label_configs(custom_labels)) if i < len(all_lora_groups) and all_lora_groups[i].get("enabled", True)]
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
        # --- LOGGING RESTORED ---
        Log.info(f"Boundary {boundary} on scheduler '{scheduler}' corresponds to a switch AT step {switching_step}/{steps}.")
        
        # --- LOGGING RESTORED ---
        Log.header("=== PHASE 1: HIGH-NOISE PROCESSING ===")
        high_noise_latents = []
        high_noise_cache = {}
        cache_dir = os.path.join(folder_paths.get_temp_directory(), "wan_high_noise_cache")
        os.makedirs(cache_dir, exist_ok=True)

        for i, group in enumerate(lora_groups):
            # --- LOGGING RESTORED ---
            Log.info(f"--- High-Noise Group {i+1}/{len(lora_groups)} ---")
            high_lora_stack = [{'name': r['high_name'], 'strength': r['high_strength'], 'enabled': r['enabled']} for r in group['rows'] if r['high_name'] and r['enabled']]
            cache_key = tuple((l['name'], l['strength']) for l in high_lora_stack)
            
            if cache_key in high_noise_cache:
                # --- LOGGING RESTORED ---
                Log.success("Found in-memory cached high-noise latent for this stack. Reusing.")
                high_noise_latents.append(high_noise_cache[cache_key])
                continue

            full_config = {'high_lora_stack': cache_key, 'seed': seed, 'steps': steps, 'switching_step': switching_step, 'cfg_high_noise': cfg_high_noise, 'sampler_name': sampler_name, 'scheduler': scheduler, 'sigma_shift': sigma_shift, 'width': width, 'height': height, 'frame_count': frame_count, 'positive_hash': positive_hash, 'negative_hash': negative_hash}
            config_hash = hashlib.sha256(str(full_config).encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{config_hash}.pt")

            if os.path.exists(cache_file):
                try:
                    cached_data = torch.load(cache_file, map_location=comfy.model_management.intermediate_device())
                    # --- LOGGING RESTORED ---
                    Log.success(f"Loaded persistent cache from {cache_file}")
                    high_noise_latents.append((cached_data, config_hash))
                    high_noise_cache[cache_key] = (cached_data, config_hash)
                    continue
                except Exception as e:
                    # --- LOGGING RESTORED ---
                    Log.fail(f"Failed to load persistent cache: {e}. Recomputing.")
            
            # --- LOGGING RESTORED ---
            Log.info("No cache entry found. Computing new high-noise latent.")
            comfy.model_management.load_model_gpu(model_high_noise)
            mh_clone = apply_lora_stack(model_high_noise, high_lora_stack)
            
            # MODIFICATION START: Keep track of the last modified high-noise model
            last_mh_clone = mh_clone
            # MODIFICATION END

            mh_clone = set_shift(mh_clone, sigma_shift)
            
            noise = comfy.sample.prepare_noise(base_latent, seed)
            latent_for_handoff = comfy.sample.sample(mh_clone, noise, steps, cfg_high_noise, sampler_name, scheduler, positive, negative, base_latent, denoise=1.0, start_step=0, last_step=switching_step, force_full_denoise=True, seed=seed)
            
            computed_data = {'latent': latent_for_handoff.clone(), 'noise': noise.clone()}
            high_noise_cache[cache_key] = (computed_data, config_hash)
            high_noise_latents.append((computed_data, config_hash))
            try:
                torch.save(computed_data, cache_file)
                # --- LOGGING RESTORED ---
                Log.success(f"Saved persistent cache to {cache_file}")
            except Exception as e:
                # --- LOGGING RESTORED ---
                Log.fail(f"Failed to save persistent cache: {e}")
            del mh_clone
            comfy.model_management.soft_empty_cache()

        # --- LOGGING RESTORED ---
        Log.header("=== PHASE 2: LOW-NOISE PROCESSING ===")
        final_latents_for_output = []
        low_noise_cache = {}
        low_noise_cache_dir = os.path.join(folder_paths.get_temp_directory(), "wan_low_noise_cache")
        os.makedirs(low_noise_cache_dir, exist_ok=True)
        decoded_images_for_output = []

        is_bypassed = cfg_low_noise == 0
        if is_bypassed: Log.info("Low-noise CFG is 0. Bypassing low-noise pass for all groups.")
        
        # MODIFICATION: Initialize last_ml_clone to prevent undefined variable error when bypassed
        last_ml_clone = model_low_noise
        
        if not is_bypassed and switching_step < steps:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()

        for i, group in enumerate(lora_groups):
            cached_data, high_noise_hash = high_noise_latents[i]
            low_lora_stack = [{'name': r['low_name'], 'strength': r['low_strength'], 'enabled': r['enabled']} for r in group['rows'] if r['low_name'] and r['enabled']]
            low_lora_tuple = tuple((l['name'], l['strength']) for l in low_lora_stack)
            in_mem_low_cache_key = low_lora_tuple + (cfg_low_noise, high_noise_hash)
            
            final_latent = None
            if not is_bypassed and switching_step < steps:
                if in_mem_low_cache_key in low_noise_cache:
                    # --- LOGGING RESTORED ---
                    Log.success(f"Found in-memory cached low-noise latent for group {i+1}/{len(lora_groups)}. Reusing.")
                    final_latent = low_noise_cache[in_mem_low_cache_key]['latent']
                else:
                    low_full_config = {'low_lora_stack': low_lora_tuple, 'cfg_low_noise': cfg_low_noise, 'sampler_name': sampler_name, 'scheduler': scheduler, 'sigma_shift': sigma_shift, 'positive_hash': positive_hash, 'negative_hash': negative_hash, 'high_noise_hash': high_noise_hash}
                    low_config_hash = hashlib.sha256(str(low_full_config).encode()).hexdigest()
                    low_cache_file = os.path.join(low_noise_cache_dir, f"{low_config_hash}.pt")
                    
                    if os.path.exists(low_cache_file):
                        try:
                            cached_low_data = torch.load(low_cache_file, map_location=comfy.model_management.intermediate_device())
                            # --- LOGGING RESTORED ---
                            Log.success(f"Loaded persistent cache for low-noise pass from {low_cache_file}")
                            final_latent = cached_low_data['latent']
                            low_noise_cache[in_mem_low_cache_key] = cached_low_data
                        except Exception as e:
                            # --- LOGGING RESTORED ---
                            Log.fail(f"Failed to load persistent low-noise cache: {e}. Recomputing.")
                    
                    if final_latent is None:
                        # --- LOGGING RESTORED ---
                        Log.info(f"--- Low-Noise Group {i+1}/{len(lora_groups)} ---")
                        comfy.model_management.load_model_gpu(model_low_noise)
                        ml_clone = apply_lora_stack(model_low_noise, low_lora_stack)

                        # MODIFICATION START: Keep track of the last modified low-noise model
                        last_ml_clone = ml_clone
                        # MODIFICATION END
                        
                        ml_clone = set_shift(ml_clone, sigma_shift)
                        final_latent = comfy.sample.sample(ml_clone, cached_data['noise'], steps, cfg_low_noise, sampler_name, scheduler, positive, negative, cached_data['latent'], denoise=1.0, start_step=switching_step, last_step=steps, force_full_denoise=True, seed=seed)
                        computed_low_data = {'latent': final_latent.clone()}
                        low_noise_cache[in_mem_low_cache_key] = computed_low_data
                        try:
                            torch.save(computed_low_data, low_cache_file)
                            # --- LOGGING RESTORED ---
                            Log.success(f"Saved persistent cache for low-noise pass to {low_cache_file}")
                        except Exception as e:
                            # --- LOGGING RESTORED ---
                            Log.fail(f"Failed to save persistent low-noise cache: {e}")
                        del ml_clone
                        comfy.model_management.soft_empty_cache()
            else:
                final_latent = cached_data['latent']
            
            final_latents_for_output.append(final_latent)
            if enable_vae_decode and vae:
                try:
                    images = vae.decode(final_latent.to(vae.device))
                    if images is not None:
                        if len(images.shape) == 5: images = images.reshape(-1, *images.shape[-3:])
                        decoded_images_for_output.append(images.to(torch.device('cpu')))
                    else:
                        Log.fail(f"VAE decode returned None for group {i+1}/{len(lora_groups)}")
                        decoded_images_for_output.append(torch.zeros([frame_count, height, width, 3], device='cpu'))
                except Exception as e:
                    Log.fail(f"VAE decode failed for group {i+1}/{len(lora_groups)}: {e}")
                    decoded_images_for_output.append(torch.zeros([frame_count, height, width, 3], device='cpu'))

        high_noise_batch = {"samples": torch.cat([cached[0]['latent'] for cached in high_noise_latents], dim=0)}
        final_latent_batch = {"samples": torch.cat(final_latents_for_output, dim=0)}
        final_images_batch = torch.cat(decoded_images_for_output, dim=0) if decoded_images_for_output else torch.zeros([0, height, width, 3])
        comparison_grid = None

        if final_images_batch.numel() > 0 and create_comparison_grid:
            config_labels = [custom_label_list[i] if i < len(custom_label_list) and custom_label_list[i] else get_fallback_label(group["rows"], is_bypassed) for i, group in enumerate(lora_groups)]
            max_label_area_height = 0
            if add_labels:
                max_lines = max(len(label.split('\n')) for label in config_labels) if config_labels else 0
                try: font = ImageFont.truetype("arial.ttf", label_font_size)
                except IOError: font = ImageFont.load_default()
                dummy_draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
                _, _, _, bottom = dummy_draw.textbbox((0,0), "\n".join(["L"] * max_lines) if max_lines > 0 else "L", font=font, spacing=4)
                max_label_area_height = bottom + 10

            comparison_frames = []
            for frame_idx in range(frame_count):
                frame_row = []
                for config_idx in range(len(lora_groups)):
                    img_idx = config_idx * frame_count + frame_idx
                    if img_idx < final_images_batch.shape[0]:
                        frame_img = final_images_batch[img_idx]
                        if add_labels:
                            frame_img = add_label_to_image(frame_img, config_labels[config_idx], label_font_size, fixed_label_area_height=max_label_area_height).squeeze(0)
                        frame_row.append(frame_img.unsqueeze(0))
                if frame_row:
                    comparison_frames.append(torch.cat(frame_row, dim=2))
            if comparison_frames:
                comparison_grid = torch.cat(comparison_frames, dim=0)
        
        settings_string = f"Dimensions: {width}x{height} | Frames: {frame_count} | Seed: {seed} | Steps: {steps} | Boundary: {boundary}"
        
        return (high_noise_batch, final_latent_batch, final_images_batch, comparison_grid, settings_string, last_mh_clone, last_ml_clone)
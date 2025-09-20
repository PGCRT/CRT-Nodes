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
            if len(row_parts) != 4: continue
            lora_rows.append({
                "high_name": row_parts[0] if row_parts[0] != 'none' else None,
                "high_strength": float(row_parts[1]),
                "low_name": row_parts[2] if row_parts[2] != 'none' else None,
                "low_strength": float(row_parts[3]),
            })

        if lora_rows:
            groups.append({
                "rows": lora_rows,
                "enabled": enabled_str.lower() == 'true'
            })
    return groups

def parse_label_configs(label_string: str):
    if not label_string.strip(): return []
    return [line.strip() for line in label_string.strip().split('\n')]

def get_fallback_label(lora_rows, is_bypassed):
    lines = []
    def format_name(name):
        if not name: return "None"
        label = os.path.basename(name)
        if label.endswith('.safetensors'): label = label[:-12]
        return label

    high_loras_in_stack = [lora for lora in lora_rows if lora.get("high_name")]
    if high_loras_in_stack:
        for lora in high_loras_in_stack:
            lines.append(f"H: {format_name(lora['high_name'])} @ {lora['high_strength']:.2f}")
    
    if is_bypassed:
        lines.append("L: Bypassed")
    else:
        low_loras_in_stack = [lora for lora in lora_rows if lora.get("low_name")]
        if low_loras_in_stack:
            for lora in low_loras_in_stack:
                lines.append(f"L: {format_name(lora['low_name'])} @ {lora['low_strength']:.2f}")
    
    return "\n".join(lines) if lines else "No LoRAs"

def apply_lora_stack(model, lora_list):
    model_clone = model.clone()
    for lora in lora_list:
        if not lora.get('name'): continue
        lora_path = find_lora_path_by_name(lora['name'])
        if lora_path:
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_clone, _ = comfy.sd.load_lora_for_models(model_clone, None, lora_data, lora['strength'], lora['strength'])
            Log.success(f"Applied LoRA '{lora['name']}' @ {lora['strength']}")
    return model_clone

def tensor_to_pil(tensor_image):
    if tensor_image.dim() == 4: tensor_image = tensor_image.squeeze(0)
    np_image = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)

def add_label_to_image(image_tensor, label_text, font_size=24, fixed_label_area_height=None):
    pil_image = tensor_to_pil(image_tensor)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: font = ImageFont.load_default()
    draw = ImageDraw.Draw(pil_image)
    left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font, spacing=4)
    text_width = right - left
    text_height = bottom - top
    label_area_height = fixed_label_area_height if fixed_label_area_height is not None else text_height + 10
    old_width, old_height = pil_image.size
    new_height = old_height + label_area_height
    labeled_image = Image.new('RGB', (old_width, new_height), (40, 40, 40))
    labeled_image.paste(pil_image, (0, 0))
    draw = ImageDraw.Draw(labeled_image)
    text_x = (old_width - text_width) // 2
    text_y = old_height + (label_area_height - text_height) // 2
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
                "lora_batch_config": ("STRING", { "multiline": True, "default": "" }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,), "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "enable_vae_decode": ("BOOLEAN", {"default": True}),
                "create_comparison_grid": ("BOOLEAN", {"default": True}),
                "add_labels": ("BOOLEAN", {"default": True}),
                "custom_labels": ("STRING", { "multiline": True, "default": "" }),
                "label_font_size": ("INT", {"default": 24, "min": 8, "max": 72}),
            }, "optional": { "vae": ("VAE",), }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("high_noise_latent_batch", "final_latent_batch", "final_images_batch", "comparison_grid", "settings_string")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(self, model_high_noise, width, height, frame_count, model_low_noise, positive, negative, seed, lora_batch_config, steps, boundary, cfg_high_noise, cfg_low_noise, sampler_name, scheduler, sigma_shift, enable_vae_decode, create_comparison_grid, add_labels, custom_labels, label_font_size, vae=None):
        all_lora_groups = parse_lora_configs(lora_batch_config)
        lora_groups = [g for g in all_lora_groups if g.get("enabled", True)]

        if not lora_groups:
            Log.fail("All LoRA groups are disabled. Nothing to sample.")
            empty_latent = {"samples": torch.zeros([0, 4, height // 8, width // 8])}
            return (empty_latent, empty_latent, None, None, "All LoRA groups were disabled.")

        all_custom_labels = parse_label_configs(custom_labels)
        custom_label_list = [label for i, label in enumerate(all_custom_labels) if i < len(all_lora_groups) and all_lora_groups[i].get("enabled", True)]

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

        Log.header("=== PHASE 1: HIGH-NOISE PROCESSING ===")
        high_noise_latents = []
        high_noise_cache = {}

        for i, group in enumerate(lora_groups):
            Log.info(f"--- High-Noise Group {i+1}/{len(lora_groups)} ---")
            high_lora_stack = [{'name': r['high_name'], 'strength': r['high_strength']} for r in group['rows'] if r['high_name']]
            cache_key = tuple((l['name'], l['strength']) for l in high_lora_stack)
            
            if cache_key in high_noise_cache:
                Log.success("Found cached high-noise latent for this stack. Reusing.")
                high_noise_latents.append(high_noise_cache[cache_key])
                continue

            Log.info("No cache entry found. Computing new high-noise latent.")
            mh_clone = apply_lora_stack(model_high_noise, high_lora_stack)
            mh_clone = set_shift(mh_clone, sigma_shift)
            
            noise = comfy.sample.prepare_noise(base_latent, seed)
            latent_for_handoff = comfy.sample.sample(mh_clone, noise, steps, cfg_high_noise, sampler_name, scheduler, positive, negative, base_latent, denoise=1.0, start_step=0, last_step=switching_step, force_full_denoise=True, seed=seed)
            
            computed_data = {'latent': latent_for_handoff.clone(), 'noise': noise.clone()}
            high_noise_cache[cache_key] = computed_data
            high_noise_latents.append(computed_data)
            del mh_clone
            comfy.model_management.soft_empty_cache()

        Log.header("=== PHASE 2: LOW-NOISE PROCESSING ===")
        final_latents_for_output = []
        decoded_images_for_output = []
        is_bypassed = cfg_low_noise == 0
        if is_bypassed:
            Log.info("Low-noise CFG is 0. Bypassing low-noise pass for all groups.")

        for i, group in enumerate(lora_groups):
            cached_data = high_noise_latents[i]
            
            if is_bypassed or switching_step >= steps:
                final_latent = cached_data['latent']
            else:
                Log.info(f"--- Low-Noise Group {i+1}/{len(lora_groups)} ---")
                low_lora_stack = [{'name': r['low_name'], 'strength': r['low_strength']} for r in group['rows'] if r['low_name']]
                ml_clone = apply_lora_stack(model_low_noise, low_lora_stack)
                ml_clone = set_shift(ml_clone, sigma_shift)
                
                final_latent = comfy.sample.sample(ml_clone, cached_data['noise'], steps, cfg_low_noise, sampler_name, scheduler, positive, negative, cached_data['latent'], denoise=1.0, start_step=switching_step, last_step=steps, force_full_denoise=True, seed=seed)
                del ml_clone
                comfy.model_management.soft_empty_cache()

            final_latents_for_output.append(final_latent)

            if enable_vae_decode and vae:
                images = vae.decode(final_latent.to(vae.device))
                if len(images.shape) == 5: images = images.reshape(-1, *images.shape[-3:])
                decoded_images_for_output.append(images.to(torch.device('cpu')))

        high_noise_batch = {"samples": torch.cat([cached['latent'] for cached in high_noise_latents], dim=0)}
        final_latent_batch = {"samples": torch.cat(final_latents_for_output, dim=0)}
        final_images_batch, comparison_grid = None, None

        if decoded_images_for_output:
            final_images_batch = torch.cat(decoded_images_for_output, dim=0)
            if create_comparison_grid:
                config_labels = [custom_label_list[i] if i < len(custom_label_list) and custom_label_list[i] else get_fallback_label(group["rows"], is_bypassed) for i, group in enumerate(lora_groups)]
                
                max_label_area_height = 0
                if add_labels:
                    max_lines = 0
                    for label in config_labels:
                        max_lines = max(max_lines, len(label.split('\n')))
                    
                    try: font = ImageFont.truetype("arial.ttf", label_font_size)
                    except IOError: font = ImageFont.load_default()
                    
                    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
                    dummy_label = "\n".join(["L"] * max_lines) if max_lines > 0 else "L"
                    _, _, _, bottom = dummy_draw.textbbox((0,0), dummy_label, font=font, spacing=4)
                    max_label_area_height = bottom + 10

                comparison_frames = []
                for frame_idx in range(frame_count):
                    frame_row = []
                    for config_idx in range(len(decoded_images_for_output)):
                        img_idx = config_idx * frame_count + frame_idx
                        frame_img = final_images_batch[img_idx]
                        if add_labels:
                            frame_img = add_label_to_image(frame_img, config_labels[config_idx], label_font_size, fixed_label_area_height=max_label_area_height).squeeze(0)
                        frame_row.append(frame_img.unsqueeze(0))
                    comparison_frames.append(torch.cat(frame_row, dim=2))
                comparison_grid = torch.cat(comparison_frames, dim=0)
        
        settings_string = f"Dimensions: {width}x{height} | Frames: {frame_count} | Seed: {seed} | Steps: {steps} | Boundary: {boundary}"
        return (high_noise_batch, final_latent_batch, final_images_batch, comparison_grid, settings_string)
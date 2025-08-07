import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union
import comfy.model_management
import comfy.utils
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor
import logging

print("Loading crt-nodes module")

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored_print(message, color=Colors.ENDC):
    print(f"{color}{message}{Colors.ENDC}")

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class CRTPostProcessNode:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    precision_options = ["auto", "32", "16", "bfloat16"]
    radial_blur_types = ["spin", "zoom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "enable_upscale": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "upscale_model_path": (folder_paths.get_filename_list("upscale_models"), {"default": "None"}),
                "downscale_by": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05}),
                "rescale_method": (cls.rescale_methods, {"default": "bicubic"}),
                "precision": (cls.precision_options, {"default": "auto"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                
                "enable_levels": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                "enable_color_wheels": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "lift_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "lift_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "lift_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "gamma_r": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gamma_g": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gamma_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                
                "enable_temp_tint": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "decimals": 0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "decimals": 0}),
                
                "enable_sharpen": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "sharpen_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "sharpen_radius": ("FLOAT", {"default": 1.85, "min": 0.1, "max": 5.0, "step": 0.1, "decimals": 1}),
                "sharpen_threshold": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                "enable_small_glow": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "small_glow_intensity": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "small_glow_radius": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1, "decimals": 3}),
                "small_glow_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "enable_large_glow": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "large_glow_intensity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "large_glow_radius": ("FLOAT", {"default": 50.0, "min": 30.0, "max": 100.0, "step": 0.5, "decimals": 3}),
                "large_glow_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                "enable_glare": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "glare_type": (["star_4", "star_6", "star_8", "anamorphic_h"], {"default": "star_4"}),
                "glare_intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 100.0, "step": 0.01, "decimals": 3}), 
                "glare_length": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 100.0, "step": 0.01, "decimals": 3}),    
                "glare_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "decimals": 0}),
                "glare_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "glare_quality": ("INT", {"default": 16, "min": 4, "max": 32, "step": 4, "decimals": 0}), 
                "glare_ray_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "decimals": 2}), 

                "enable_chromatic_aberration": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "ca_strength": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "decimals": 3}),
                "ca_edge_falloff": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "enable_ca_hue_shift": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "ca_hue_shift_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "decimals": 0}),
                
                "enable_vignette": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "vignette_radius": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "vignette_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01, "decimals": 3}),
                
                "enable_radial_blur": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "radial_blur_type": (cls.radial_blur_types, {"default": "spin"}),
                "radial_blur_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005, "decimals": 3}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "radial_blur_falloff": ("FLOAT", {"default": 0.25, "min": 0.001, "max": 1.0, "step": 0.01, "decimals": 3}),
                "radial_blur_samples": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8, "decimals": 0}),
                
                "enable_film_grain": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "grain_intensity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.15, "step": 0.01, "decimals": 3}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "decimals": 2}),
                "grain_color_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                "enable_lens_distortion": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "barrel_distortion": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001, "decimals": 3}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "CRT/FX"

    DEFAULTS = {
        "downscale_by": 0.5,
        "rescale_method": "bicubic", 
        "precision": "auto", 
        "batch_size": 1,
        "exposure": 0.0, "gamma": 1.0, "brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "vibrance": 0.0,
        "lift_r": 0.0, "lift_g": 0.0, "lift_b": 0.0, "gamma_r": 1.0, "gamma_g": 1.0, "gamma_b": 1.0,
        "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0, "temperature": 0.0, "tint": 0.0,
        "sharpen_strength": 2.5, "sharpen_radius": 1.85, "sharpen_threshold": 0.015,
        "small_glow_intensity": 0.015, "small_glow_radius": 0.1, "small_glow_threshold": 0.25,
        "large_glow_intensity": 0.25, "large_glow_radius": 50.0, "large_glow_threshold": 0.3,
        "glare_intensity": 0.65, "glare_length": 1.5, "glare_angle": 0.0, "glare_threshold": 0.95, "glare_quality": 16, "glare_ray_width": 1.0, 
        "ca_strength": 0.005, "ca_edge_falloff": 2.0, 
        "enable_ca_hue_shift": False, "ca_hue_shift_degrees": 0.0,
        "vignette_strength": 0.5, "vignette_radius": 0.7, "vignette_softness": 2.0, 
        "radial_blur_type": "spin", "radial_blur_strength": 0.02, 
        "radial_blur_center_x": 0.5, "radial_blur_center_y": 0.25,
        "radial_blur_falloff": 0.25, "radial_blur_samples": 16, 
        "grain_intensity": 0.02, "grain_size": 1.0, "grain_color_amount": 0.0, 
        "barrel_distortion": 0.0
    }

    OPERATION_GROUPS = {
        "UPSCALE": ["enable_upscale", "upscale_model_path", "downscale_by", "rescale_method", "precision", "batch_size"],
        "LEVELS": ["enable_levels", "exposure", "gamma", "brightness", "contrast", "saturation", "vibrance"],
        "COLOR WHEELS": ["enable_color_wheels", "lift_r", "lift_g", "lift_b", "gamma_r", "gamma_g", "gamma_b", "gain_r", "gain_g", "gain_b"],
        "TEMPERATURE & TINT": ["enable_temp_tint", "temperature", "tint"],
        "SHARPENING": ["enable_sharpen", "sharpen_strength", "sharpen_radius", "sharpen_threshold"],
        "GLOWS": ["enable_small_glow", "small_glow_intensity", "small_glow_radius", "small_glow_threshold", "enable_large_glow", "large_glow_intensity", "large_glow_radius", "large_glow_threshold"],
        "GLARE/FLARES": ["enable_glare", "glare_type", "glare_intensity", "glare_length", "glare_angle", "glare_threshold", "glare_quality", "glare_ray_width"], 
        "CHROMATIC ABERRATION": ["enable_chromatic_aberration", "ca_strength", "ca_edge_falloff", "enable_ca_hue_shift", "ca_hue_shift_degrees"],
        "VIGNETTE": ["enable_vignette", "vignette_strength", "vignette_radius", "vignette_softness"],
        "RADIAL BLUR": ["enable_radial_blur", "radial_blur_type", "radial_blur_strength", "radial_blur_center_x", "radial_blur_center_y", "radial_blur_falloff", "radial_blur_samples"],
        "LENS DISTORTION": ["enable_lens_distortion", "barrel_distortion"],
        "FILM GRAIN": ["enable_film_grain", "grain_intensity", "grain_size", "grain_color_amount"]
    }

    def __init__(self):
        self.slider_positions = {} 
        self.capture = False
        self.unlock = False
        self.loaded_upscale_model = None
        self.current_model_path = None
        colored_print("🎨 CRT Post-Process Suite initialized!", Colors.HEADER)

    def _count_enabled_effects(self, kwargs):
        """Count how many effects are enabled"""
        effect_keys = [
            'enable_upscale', 'enable_levels', 'enable_color_wheels', 'enable_temp_tint',
            'enable_sharpen', 'enable_small_glow', 'enable_large_glow', 'enable_glare',
            'enable_chromatic_aberration', 'enable_vignette', 'enable_radial_blur',
            'enable_film_grain', 'enable_lens_distortion'
        ]
        return sum(1 for key in effect_keys if kwargs.get(key, False))

    def _get_enabled_effects_list(self, kwargs):
        """Get list of enabled effect names"""
        effect_mapping = {
            'enable_upscale': 'Upscaling',
            'enable_levels': 'Levels',
            'enable_color_wheels': 'Color Wheels',
            'enable_temp_tint': 'Temperature & Tint',
            'enable_sharpen': 'Sharpening',
            'enable_small_glow': 'Small Glow',
            'enable_large_glow': 'Large Glow',
            'enable_glare': 'Glare/Flares',
            'enable_chromatic_aberration': 'Chromatic Aberration',
            'enable_vignette': 'Vignette',
            'enable_radial_blur': 'Radial Blur',
            'enable_film_grain': 'Film Grain',
            'enable_lens_distortion': 'Lens Distortion'
        }
        return [effect_mapping[key] for key in effect_mapping if kwargs.get(key, False)]

    def _hue_shift_rgb_color_vector(self, rgb_color_tensor, hue_shift_degrees):
        if hue_shift_degrees == 0.0:
            return rgb_color_tensor.clone()

        r, g, b = rgb_color_tensor[0].item(), rgb_color_tensor[1].item(), rgb_color_tensor[2].item()

        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val
        
        v = max_val
        s = 0.0
        if max_val > 1e-6:
            s = delta / max_val
        
        h = 0.0
        if delta > 1e-6:
            if r == max_val: h = (g - b) / delta
            elif g == max_val: h = 2.0 + (b - r) / delta
            else: h = 4.0 + (r - g) / delta
        h *= 60.0
        if h < 0: h += 360.0
        
        h = (h + hue_shift_degrees) % 360.0
        
        c = v * s
        h_prime = h / 60.0
        
        if abs(h_prime - 6.0) < 1e-6: h_prime = 0.0 

        x = c * (1.0 - abs(h_prime % 2.0 - 1.0))
        m = v - c
        r_out, g_out, b_out = 0.0, 0.0, 0.0

        if 0 <= h_prime < 1: r_out, g_out, b_out = c, x, 0
        elif 1 <= h_prime < 2: r_out, g_out, b_out = x, c, 0
        elif 2 <= h_prime < 3: r_out, g_out, b_out = 0, c, x
        elif 3 <= h_prime < 4: r_out, g_out, b_out = 0, x, c
        elif 4 <= h_prime < 5: r_out, g_out, b_out = x, 0, c
        elif 5 <= h_prime < 6: r_out, g_out, b_out = c, 0, x

        final_r = r_out + m
        final_g = g_out + m
        final_b = b_out + m
        
        return torch.tensor([final_r, final_g, final_b], device=rgb_color_tensor.device, dtype=rgb_color_tensor.dtype)

    def load_upscale_model(self, model_path):
        if model_path == "None" or not model_path:
            if self.loaded_upscale_model is not None:
                colored_print("🗑️ Clearing upscale model (None selected)", Colors.YELLOW)
            self.loaded_upscale_model = None
            self.current_model_path = None
            return None

        if self.current_model_path == model_path and self.loaded_upscale_model is not None:
            colored_print(f"🎯 [Cache Hit] Reusing upscale model: {model_path}", Colors.GREEN)
            return self.loaded_upscale_model

        colored_print(f"📦 Loading upscale model: {model_path}", Colors.CYAN)

        try:
            model_full_path = folder_paths.get_full_path_or_raise("upscale_models", model_path)
            colored_print(f"   📁 Full path: {model_full_path}", Colors.BLUE)
            
            sd = comfy.utils.load_torch_file(model_full_path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                colored_print("   🔧 Applying state dict prefix replacement", Colors.BLUE)
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
            
            out = ModelLoader().load_from_state_dict(sd).eval()

            if not isinstance(out, ImageModelDescriptor):
                colored_print("❌ ERROR: Model must be a single-image model (Spandrel ImageModelDescriptor)", Colors.RED)
                raise ValueError("Upscale model must be a single-image model (Spandrel ImageModelDescriptor).")

            self.loaded_upscale_model = out
            self.current_model_path = model_path
            colored_print(f"✅ Upscale model loaded successfully!", Colors.GREEN)
            colored_print(f"   📊 Model scale: {out.scale}x", Colors.GREEN)
            return self.loaded_upscale_model

        except Exception as e:
            colored_print(f"❌ Failed to load upscale model: {e}", Colors.RED)
            self.loaded_upscale_model = None
            self.current_model_path = None
            return None
            
    def reset_operation(self, operation_name):
        colored_print(f"🔄 Reset operation '{operation_name}' called", Colors.YELLOW)
        colored_print("   ⚠️ This function relies on UI widget interaction", Colors.YELLOW)

    def process(self, image, **kwargs):
        colored_print("\n🎨 Starting CRT Post-Process Pipeline...", Colors.HEADER)
        
        device = comfy.model_management.get_torch_device()
        image_input = image.to(device)
        batch_size, height, width, channels = image_input.shape
        colored_print(f"📐 Input Analysis:", Colors.BLUE)
        colored_print(f"   Batch Size: {batch_size}", Colors.BLUE)
        colored_print(f"   Dimensions: {width}x{height}", Colors.BLUE)
        colored_print(f"   Channels: {channels}", Colors.BLUE)
        colored_print(f"   Device: {device}", Colors.BLUE)
        colored_print(f"   Data Type: {image_input.dtype}", Colors.BLUE)
        enabled_count = self._count_enabled_effects(kwargs)
        enabled_effects = self._get_enabled_effects_list(kwargs)
        
        colored_print(f"🎭 Effect Configuration:", Colors.HEADER)
        colored_print(f"   Enabled Effects: {enabled_count}", Colors.BLUE)
        if enabled_effects:
            for i, effect in enumerate(enabled_effects, 1):
                colored_print(f"   {i}. {effect}", Colors.CYAN)
        else:
            colored_print("   🚫 No effects enabled - passthrough mode", Colors.YELLOW)
            return (image_input,)
        if kwargs.get('enable_upscale', False):
            model_path = kwargs.get('upscale_model_path', "None")
            colored_print(f"📦 Upscale Model Loading:", Colors.HEADER)
            self.load_upscale_model(model_path)
        colored_print(f"🔄 Processing {batch_size} frame(s)...", Colors.HEADER)
        processed_frames = []
        
        for i in range(image_input.shape[0]):
            if batch_size > 1:
                colored_print(f"   🖼️ Processing frame {i+1}/{batch_size}...", Colors.CYAN)
            
            frame = image_input[i:i+1].clone() 
            processed_frame = self._process_single_frame(frame, device, i+1, batch_size, **kwargs)
            processed_frames.append(processed_frame)
        
        result = torch.cat(processed_frames, dim=0)
        final_h, final_w = result.shape[1:3]
        colored_print(f"\n✅ CRT Post-Processing completed successfully!", Colors.HEADER)
        colored_print(f"   🎯 Process Summary:", Colors.GREEN)
        colored_print(f"     Input: {width}x{height} → Output: {final_w}x{final_h}", Colors.GREEN)
        colored_print(f"     Effects Applied: {enabled_count}", Colors.GREEN)
        colored_print(f"     Frames Processed: {batch_size}", Colors.GREEN)
        
        return (result,)
    
    def _process_single_frame(self, image, device, frame_num, total_frames, **kwargs):
        img = image
        effect_count = 0
        if kwargs.get('enable_upscale', False):
            effect_count += 1
            colored_print(f"   📈 [{effect_count}] Applying Upscaling...", Colors.CYAN)
            img = self._apply_upscale(img, device, **kwargs)
        
        if kwargs.get('enable_lens_distortion', False):
            effect_count += 1
            colored_print(f"   🔍 [{effect_count}] Applying Lens Distortion...", Colors.CYAN)
            img = self._apply_lens_distortion(img, kwargs)
        
        if kwargs.get('enable_chromatic_aberration', False):
            effect_count += 1
            colored_print(f"   🌈 [{effect_count}] Applying Chromatic Aberration...", Colors.CYAN)
            img = self._apply_chromatic_aberration(img, kwargs)
        
        if kwargs.get('enable_temp_tint', False):
            effect_count += 1
            colored_print(f"   🌡️ [{effect_count}] Applying Temperature & Tint...", Colors.CYAN)
            img = self._apply_temperature_tint(img, kwargs)
        
        if kwargs.get('enable_levels', False):
            effect_count += 1
            colored_print(f"   📊 [{effect_count}] Applying Levels...", Colors.CYAN)
            img = self._apply_levels(img, kwargs)
        
        if kwargs.get('enable_color_wheels', False):
            effect_count += 1
            colored_print(f"   🎨 [{effect_count}] Applying Color Wheels...", Colors.CYAN)
            img = self._apply_color_wheels(img, kwargs)
        
        if kwargs.get('enable_sharpen', False):
            effect_count += 1
            colored_print(f"   ⚡ [{effect_count}] Applying Sharpening...", Colors.CYAN)
            img = self._apply_sharpening(img, kwargs)
        
        if kwargs.get('enable_small_glow', False):
            effect_count += 1
            colored_print(f"   ✨ [{effect_count}] Applying Small Glow...", Colors.CYAN)
            img = self._apply_glow(img, kwargs, 'small')
        
        if kwargs.get('enable_large_glow', False):
            effect_count += 1
            colored_print(f"   🌟 [{effect_count}] Applying Large Glow...", Colors.CYAN)
            img = self._apply_glow(img, kwargs, 'large')
        
        if kwargs.get('enable_glare', False):
            effect_count += 1
            colored_print(f"   ⭐ [{effect_count}] Applying Glare/Flares...", Colors.CYAN)
            img = self._apply_glare(img, kwargs)
        
        if kwargs.get('enable_radial_blur', False):
            effect_count += 1
            blur_type = kwargs.get('radial_blur_type', self.DEFAULTS['radial_blur_type'])
            colored_print(f"   🌀 [{effect_count}] Applying Radial Blur ({blur_type})...", Colors.CYAN)
            if blur_type == 'zoom':
                img = self._apply_zoom_blur(img, kwargs)
            elif blur_type == 'spin':
                img = self._apply_radial_spin_blur(img, kwargs)
        
        if kwargs.get('enable_vignette', False):
            effect_count += 1
            colored_print(f"   🎭 [{effect_count}] Applying Vignette...", Colors.CYAN)
            img = self._apply_vignette(img, kwargs)
        
        if kwargs.get('enable_film_grain', False):
            effect_count += 1
            colored_print(f"   📹 [{effect_count}] Applying Film Grain...", Colors.CYAN)
            img = self._apply_film_grain(img, kwargs)
        
        if total_frames > 1:
            colored_print(f"   ✅ Frame {frame_num} completed ({effect_count} effects applied)", Colors.GREEN)
        
        return img

    def _apply_upscale(self, img, current_device, **kwargs):
        downscale_by = kwargs.get('downscale_by', self.DEFAULTS['downscale_by'])
        rescale_method = kwargs.get('rescale_method', self.DEFAULTS['rescale_method'])
        precision_str = kwargs.get('precision', self.DEFAULTS['precision'])

        colored_print(f"      📈 Upscale Configuration:", Colors.BLUE)
        colored_print(f"         Model: {kwargs.get('upscale_model_path', 'None')}", Colors.BLUE)
        colored_print(f"         Downscale Factor: {downscale_by:.2f}", Colors.BLUE)
        colored_print(f"         Rescale Method: {rescale_method}", Colors.BLUE)
        colored_print(f"         Precision: {precision_str}", Colors.BLUE)

        if self.loaded_upscale_model is None:
            colored_print("      ❌ No upscale model loaded, skipping upscale step", Colors.RED)
            return img

        original_device = img.device
        original_dtype = img.dtype
        original_h, original_w = img.shape[1], img.shape[2]
        colored_print(f"      📐 Input: {original_w}x{original_h}, {original_dtype}@{original_device}", Colors.BLUE)
        processing_dtype = torch.float32 
        if precision_str == "auto":
            if current_device.type == 'cuda': processing_dtype = torch.float16
        elif precision_str == "16":
            if current_device.type == 'cuda': processing_dtype = torch.float16
            else: processing_dtype = torch.bfloat16
        elif precision_str == "bfloat16":
            processing_dtype = torch.bfloat16
        
        if processing_dtype == torch.float16 and current_device.type != 'cuda':
            colored_print("      ⚠️ float16 not supported on non-CUDA, using bfloat16", Colors.YELLOW)
            processing_dtype = torch.bfloat16
        
        colored_print(f"      🔧 Processing dtype: {processing_dtype}", Colors.BLUE)

        upscale_model_execution_device = comfy.model_management.get_torch_device()
        
        model_descriptor = self.loaded_upscale_model.to(upscale_model_execution_device)
        colored_print(f"      🚀 Model scale: {model_descriptor.scale}x", Colors.GREEN)

        in_img_bchw = img.permute(0, 3, 1, 2).contiguous().to(dtype=processing_dtype, device=upscale_model_execution_device)
        tile = 512 
        overlap = 32 
        if model_descriptor.scale > 1:
             overlap = max(32, tile // 8) 

        oom = True
        retries = 0
        max_retries = 4
        s = None 

        while oom and retries < max_retries:
            try:
                colored_print(f"      🔄 Attempt {retries + 1}: tile={tile}, overlap={overlap}", Colors.BLUE)
                pbar_steps = in_img_bchw.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img_bchw.shape[3], in_img_bchw.shape[2], 
                    tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(pbar_steps)

                def model_function_for_tiled_scale(tile_tensor_bchw):
                    tile_tensor_bchw = tile_tensor_bchw.to(processing_dtype)
                    
                    autocast_enabled = (processing_dtype == torch.float16 or processing_dtype == torch.bfloat16)
                    if autocast_enabled and upscale_model_execution_device.type == 'cuda':
                        with torch.autocast(device_type=upscale_model_execution_device.type, dtype=processing_dtype, enabled=True):
                            return model_descriptor(tile_tensor_bchw) 
                    else:
                        return model_descriptor(tile_tensor_bchw)

                s = comfy.utils.tiled_scale(
                    in_img_bchw,
                    model_function_for_tiled_scale,
                    tile_x=tile, tile_y=tile,
                    overlap=overlap,
                    upscale_amount=model_descriptor.scale,
                    pbar=pbar
                )
                oom = False
                colored_print("      ✅ Upscaling successful!", Colors.GREEN)
            except comfy.model_management.OOM_EXCEPTION as e:
                colored_print(f"      ⚠️ OOM detected with tile size {tile}, reducing...", Colors.YELLOW)
                tile //= 2
                if tile < 64: 
                    colored_print(f"      ❌ Tile size too small ({tile}), cannot upscale", Colors.RED)
                    return img 
                comfy.model_management.soft_empty_cache()
            retries += 1

        if oom or s is None: 
            colored_print("      ❌ Failed to upscale after multiple retries", Colors.RED)
            return img 
        
        s_bhwc = s.permute(0, 2, 3, 1).contiguous()
        s_bhwc = torch.clamp(s_bhwc, min=0.0, max=1.0)
        upscaled_h, upscaled_w = s_bhwc.shape[1], s_bhwc.shape[2]
        colored_print(f"      📈 Upscaled to: {upscaled_w}x{upscaled_h}", Colors.GREEN)
        if downscale_by < 1.0:
            target_h = round(s_bhwc.shape[1] * downscale_by)
            target_w = round(s_bhwc.shape[2] * downscale_by)
            if target_h < 1 or target_w < 1:
                colored_print(f"      ⚠️ Downscale target too small ({target_h}x{target_w}), skipping", Colors.YELLOW)
            else:
                colored_print(f"      📉 Downscaling to: {target_w}x{target_h} using {rescale_method}", Colors.BLUE)
                s_bhwc_permuted = s_bhwc.permute(0, 3, 1, 2).contiguous()
                s_interpolated = F.interpolate(
                    s_bhwc_permuted,
                    size=(target_h, target_w),
                    mode=rescale_method if rescale_method != "lanczos" else "bicubic", 
                    antialias=True if rescale_method in ["bilinear", "bicubic", "area"] else None
                )
                s_bhwc = s_interpolated.permute(0, 2, 3, 1).contiguous()

        if s_bhwc.dtype != original_dtype or s_bhwc.device != original_device:
            s_bhwc = s_bhwc.to(dtype=original_dtype, device=original_device)
        
        final_h, final_w = s_bhwc.shape[1], s_bhwc.shape[2]
        colored_print(f"      🎯 Final result: {final_w}x{final_h}", Colors.GREEN)
        return s_bhwc

    def _apply_temperature_tint(self, img, kwargs):
        temp = kwargs.get('temperature', 0.0) / 100.0
        tint = kwargs.get('tint', 0.0) / 100.0
        
        colored_print(f"      🌡️ Temperature: {kwargs.get('temperature', 0.0)}, Tint: {kwargs.get('tint', 0.0)}", Colors.BLUE)
        
        if temp == 0.0 and tint == 0.0:
            colored_print("      🚫 No adjustment needed (both values are 0)", Colors.YELLOW)
            return img
        
        img_copy = img.clone() 

        if temp != 0.0:
            if temp > 0:  
                img_copy[..., 0] = img_copy[..., 0] * (1.0 + temp * 0.3) 
                img_copy[..., 2] = img_copy[..., 2] * (1.0 - temp * 0.2)
                colored_print(f"      🔥 Warmer: R+{temp*0.3:.3f}, B-{temp*0.2:.3f}", Colors.BLUE)
            else:  
                img_copy[..., 0] = img_copy[..., 0] * (1.0 + temp * 0.2) 
                img_copy[..., 2] = img_copy[..., 2] * (1.0 - temp * 0.3)
                colored_print(f"      ❄️ Cooler: R{temp*0.2:.3f}, B{temp*0.3:.3f}", Colors.BLUE)
        
        if tint != 0.0:
            if tint > 0:  
                img_copy[..., 1] = img_copy[..., 1] * (1.0 + tint * 0.3)
                colored_print(f"      💚 Green tint: G+{tint*0.3:.3f}", Colors.BLUE)
            else:  
                img_copy[..., 0] = img_copy[..., 0] * (1.0 - tint * 0.15) 
                img_copy[..., 2] = img_copy[..., 2] * (1.0 - tint * 0.15)
                colored_print(f"      💜 Magenta tint: R&B{tint*0.15:.3f}", Colors.BLUE)
        
        return torch.clamp(img_copy, 0, 1)

    def _apply_levels(self, img, kwargs):
        exposure = kwargs.get('exposure', 0.0)
        gamma = kwargs.get('gamma', 1.0)
        brightness = kwargs.get('brightness', 0.0)
        contrast = kwargs.get('contrast', 1.0)
        saturation = kwargs.get('saturation', 1.0)
        vibrance = kwargs.get('vibrance', 0.0)
        
        colored_print(f"      📊 Levels Config:", Colors.BLUE)
        colored_print(f"         Exposure: {exposure:.2f}, Gamma: {gamma:.2f}", Colors.BLUE)
        colored_print(f"         Brightness: {brightness:.2f}, Contrast: {contrast:.2f}", Colors.BLUE)
        colored_print(f"         Saturation: {saturation:.2f}, Vibrance: {vibrance:.2f}", Colors.BLUE)
        
        img_copy = img.clone()

        if exposure != 0.0:
            img_copy = img_copy * (2.0 ** exposure)
            colored_print(f"      ☀️ Applied exposure: {2.0**exposure:.3f}x multiplier", Colors.BLUE)
        
        if gamma != 1.0 and gamma > 0: 
            img_copy = torch.pow(torch.clamp(img_copy, 1e-6, 1.0), 1.0 / gamma) 
            colored_print(f"      🔆 Applied gamma: 1/{gamma:.3f} = {1.0/gamma:.3f}", Colors.BLUE)
        
        if brightness != 0.0:
            img_copy = img_copy + brightness
            colored_print(f"      💡 Applied brightness: {brightness:+.3f}", Colors.BLUE)
        
        if contrast != 1.0:
            img_copy = (img_copy - 0.5) * contrast + 0.5
            colored_print(f"      ⚫ Applied contrast: {contrast:.3f}x", Colors.BLUE)
        
        img_copy = torch.clamp(img_copy, 0, 1) 

        if saturation != 1.0:
            gray = torch.mean(img_copy, dim=-1, keepdim=True)
            img_copy = gray + (img_copy - gray) * saturation
            colored_print(f"      🌈 Applied saturation: {saturation:.3f}x", Colors.BLUE)
        
        if vibrance != 0.0:
            gray = torch.mean(img_copy, dim=-1, keepdim=True) 
            current_sat_approx = torch.abs(img_copy - gray).max(dim=-1, keepdim=True)[0] 
            vibrance_mask = 1.0 - torch.clamp(current_sat_approx * 2.0, 0, 1) 
            img_copy = gray + (img_copy - gray) * (1.0 + vibrance * vibrance_mask)
            colored_print(f"      ✨ Applied vibrance: {vibrance:+.3f} (selective)", Colors.BLUE)
        
        return torch.clamp(img_copy, 0, 1)

    def _apply_color_wheels(self, img, kwargs):
        lift = torch.tensor([kwargs.get('lift_r', 0.0), 
                           kwargs.get('lift_g', 0.0), 
                           kwargs.get('lift_b', 0.0)], device=img.device, dtype=img.dtype)
        
        gamma_adj = torch.tensor([kwargs.get('gamma_r', 1.0), 
                                kwargs.get('gamma_g', 1.0), 
                                kwargs.get('gamma_b', 1.0)], device=img.device, dtype=img.dtype)
        
        gain = torch.tensor([kwargs.get('gain_r', 1.0), 
                           kwargs.get('gain_g', 1.0), 
                           kwargs.get('gain_b', 1.0)], device=img.device, dtype=img.dtype)
        
        colored_print(f"      🎨 Color Wheels Config:", Colors.BLUE)
        colored_print(f"         Lift RGB: ({lift[0]:.3f}, {lift[1]:.3f}, {lift[2]:.3f})", Colors.BLUE)
        colored_print(f"         Gamma RGB: ({gamma_adj[0]:.3f}, {gamma_adj[1]:.3f}, {gamma_adj[2]:.3f})", Colors.BLUE)
        colored_print(f"         Gain RGB: ({gain[0]:.3f}, {gain[1]:.3f}, {gain[2]:.3f})", Colors.BLUE)
        
        img_copy = img.clone()
        if lift.abs().sum() > 1e-6:
            img_copy = img_copy + lift * (1.0 - img_copy) 
            img_copy = torch.clamp(img_copy, 0, 1)
            colored_print(f"      🌑 Lift applied to shadows", Colors.BLUE)
        if (gamma_adj - 1.0).abs().sum() > 1e-6:
            safe_gamma_adj = torch.where(gamma_adj <= 1e-6, torch.tensor(1e-6, device=img.device, dtype=img.dtype), gamma_adj)
            img_copy = torch.pow(torch.clamp(img_copy, 1e-6, 1.0), 1.0 / safe_gamma_adj)
            img_copy = torch.clamp(img_copy, 0, 1)
            colored_print(f"      🌗 Gamma applied to midtones", Colors.BLUE)
        if (gain - 1.0).abs().sum() > 1e-6:
            img_copy = img_copy * gain
            colored_print(f"      🌕 Gain applied to highlights", Colors.BLUE)
        
        return torch.clamp(img_copy, 0, 1)

    def _apply_sharpening(self, img, kwargs):
        strength = kwargs.get('sharpen_strength', 0.5)
        radius = kwargs.get('sharpen_radius', 1.0) 
        threshold = kwargs.get('sharpen_threshold', 0.0)
        
        colored_print(f"      ⚡ Sharpen Config: strength={strength:.3f}, radius={radius:.2f}, threshold={threshold:.3f}", Colors.BLUE)
        
        if strength == 0.0:
            colored_print("      🚫 No sharpening (strength = 0)", Colors.YELLOW)
            return img
        
        img_conv = img.permute(0, 3, 1, 2) 
        
        kernel_size = max(3, int(radius * 6) | 1) 
        sigma_val = radius 
        
        colored_print(f"      🔧 Kernel size: {kernel_size}, sigma: {sigma_val:.3f}", Colors.BLUE)
        
        coords = torch.arange(kernel_size, dtype=img.dtype, device=img.device)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma_val ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        blurred = img_conv
        k_h = kernel_1d.view(1, 1, 1, -1).repeat(img_conv.shape[1], 1, 1, 1) 
        blurred = F.conv2d(blurred, k_h, padding=(0, kernel_size//2), groups=img_conv.shape[1])
        
        k_v = kernel_1d.view(1, 1, -1, 1).repeat(img_conv.shape[1], 1, 1, 1) 
        blurred = F.conv2d(blurred, k_v, padding=(kernel_size//2, 0), groups=img_conv.shape[1])
        
        blurred = blurred.permute(0, 2, 3, 1) 
        unsharp_mask = img - blurred
        
        if threshold > 0:
            mask = torch.abs(unsharp_mask) > threshold
            affected_pixels = mask.float().mean().item()
            unsharp_mask = unsharp_mask * mask.float()
            colored_print(f"      🎯 Threshold applied: {affected_pixels*100:.1f}% of pixels affected", Colors.BLUE)
        
        return torch.clamp(img + unsharp_mask * strength, 0, 1)

    def _apply_glow(self, img, kwargs, glow_type):
        if glow_type == 'small':
            intensity = kwargs.get('small_glow_intensity', 0.3)
            radius = kwargs.get('small_glow_radius', 2.0) 
            threshold = kwargs.get('small_glow_threshold', 0.7)
        else: 
            intensity = kwargs.get('large_glow_intensity', 0.2)
            radius = kwargs.get('large_glow_radius', 8.0) 
            threshold = kwargs.get('large_glow_threshold', 0.8)
        
        colored_print(f"      ✨ {glow_type.title()} Glow: intensity={intensity:.3f}, radius={radius:.2f}, threshold={threshold:.3f}", Colors.BLUE)
        
        if intensity == 0.0 or radius == 0.0:
            colored_print(f"      🚫 No {glow_type} glow (intensity or radius = 0)", Colors.YELLOW)
            return img
        
        img_conv = img.permute(0, 3, 1, 2) 
        
        kernel_size = max(3, int(radius * 6) | 1) 
        sigma_val = radius
        
        coords = torch.arange(kernel_size, dtype=img.dtype, device=img.device)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma_val ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        blurred = img_conv
        k_h = kernel_1d.view(1, 1, 1, -1).repeat(img_conv.shape[1], 1, 1, 1)
        blurred = F.conv2d(blurred, k_h, padding=(0, kernel_size//2), groups=img_conv.shape[1])
        
        k_v = kernel_1d.view(1, 1, -1, 1).repeat(img_conv.shape[1], 1, 1, 1)
        blurred = F.conv2d(blurred, k_v, padding=(kernel_size//2, 0), groups=img_conv.shape[1])
        
        blurred_bhwc = blurred.permute(0, 2, 3, 1) 
        
        brightness = torch.mean(img, dim=-1, keepdim=True) 
        glow_mask = torch.clamp((brightness - threshold) / (1.0 - threshold + 1e-6), 0, 1)
        mask_coverage = (glow_mask > 0.1).float().mean().item()
        
        colored_print(f"      🌟 Glow mask coverage: {mask_coverage*100:.1f}% of image", Colors.BLUE)
        
        glow_component = blurred_bhwc * glow_mask 
        return torch.clamp(img + glow_component * intensity, 0, 1)

    def _apply_glare(self, img, kwargs):
        glare_type = kwargs.get('glare_type', 'star_4')
        intensity = kwargs.get('glare_intensity', 0.65)
        length = kwargs.get('glare_length', 1.5)
        angle = kwargs.get('glare_angle', 0.0)
        threshold = kwargs.get('glare_threshold', 0.95)
        quality = kwargs.get('glare_quality', 16) 
        ray_width = kwargs.get('glare_ray_width', self.DEFAULTS['glare_ray_width'])
        
        colored_print(f"      ⭐ Glare Config:", Colors.BLUE)
        colored_print(f"         Type: {glare_type}, Intensity: {intensity:.3f}", Colors.BLUE)
        colored_print(f"         Length: {length:.2f}, Angle: {angle:.1f}°", Colors.BLUE)
        colored_print(f"         Threshold: {threshold:.3f}, Quality: {quality}", Colors.BLUE)
        
        if intensity == 0.0 or length == 0.0:
            colored_print("      🚫 No glare (intensity or length = 0)", Colors.YELLOW)
            return img
        
        brightness = torch.mean(img, dim=-1, keepdim=True)
        glare_mask = torch.clamp((brightness - threshold) / (1.0 - threshold + 1e-6), 0, 1)
        glare_source = img * glare_mask
        
        mask_coverage = (glare_mask > 0.1).float().mean().item()
        colored_print(f"      ✨ Glare source coverage: {mask_coverage*100:.1f}% of image", Colors.BLUE)
        
        glare_source_bchw = glare_source.permute(0, 3, 1, 2)
        h, w = glare_source_bchw.shape[2], glare_source_bchw.shape[3]
        
        glare_effect_bchw = torch.zeros_like(glare_source_bchw)

        if 'star' in glare_type:
            rays = int(glare_type.split('_')[1])
            colored_print(f"      🌟 Creating {rays}-ray star glare", Colors.BLUE)
            glare_effect_bchw = self._create_star_glare(glare_source_bchw, rays, length, angle, h, w, quality, ray_width)
        elif glare_type == 'anamorphic_h':
            colored_print(f"      📏 Creating horizontal anamorphic glare", Colors.BLUE)
            if angle != 0.0:
                colored_print(f"      ⚠️ Angle {angle}° not applied for anamorphic glare", Colors.YELLOW)
            glare_effect_bchw = self._create_anamorphic_glare(glare_source_bchw, length, angle, h, w, quality)
        
        if torch.isnan(glare_effect_bchw).any() or torch.isinf(glare_effect_bchw).any():
            colored_print("      ❌ NaN or Inf detected in glare effect, skipping", Colors.RED)
            return img
        
        glare_effect_bhwc = glare_effect_bchw.permute(0, 2, 3, 1)
        result = img + glare_effect_bhwc * intensity
        
        return torch.clamp(result, 0, 1)

    def _create_star_glare(self, img_bchw, rays, length, angle, h, w, quality, ray_width):
        device = img_bchw.device
        dtype = img_bchw.dtype

        kernel_size_factor = 6.0 
        calculated_kernel_size = max(int(length * kernel_size_factor), 3) | 1 
        
        max_dim_kernel = min(h, w) // 2 
        max_dim_kernel = max(max_dim_kernel, 3) | 1

        kernel_size = min(calculated_kernel_size, max_dim_kernel)
        
        coords_1d = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords_1d, coords_1d, indexing='ij') 

        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        xr = xx * cos_a - yy * sin_a
        yr = xx * sin_a + yy * cos_a

        kernel = torch.zeros_like(xx)
        
        sigma_for_ray_thickness = ray_width 
        sigma_for_ray_length = length 

        for i in range(rays):
            ray_angle_rad = (i * 2 * math.pi / rays) 
            cos_r, sin_r = math.cos(ray_angle_rad), math.sin(ray_angle_rad)
            
            dist_to_ray_line_sq = (xr * sin_r - yr * cos_r)**2
            profile_across_ray = torch.exp(-dist_to_ray_line_sq / (2 * sigma_for_ray_thickness**2 + 1e-6))
            
            radial_dist_sq = xr**2 + yr**2
            profile_along_ray = torch.exp(-radial_dist_sq / (2 * sigma_for_ray_length**2 + 1e-6))
            
            kernel += profile_across_ray * profile_along_ray
        
        if kernel.sum() < 1e-6: 
             colored_print("      ⚠️ Star glare kernel sum near zero, skipping", Colors.YELLOW)
             return torch.zeros_like(img_bchw)

        kernel = kernel / (kernel.sum() + 1e-6) 
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        convolved_img = F.conv2d(img_bchw, kernel.expand(img_bchw.shape[1], 1, -1, -1), 
                                 padding=kernel_size//2, groups=img_bchw.shape[1])
        
        return convolved_img

    def _create_anamorphic_glare(self, img_bchw, length, angle, h, w, quality):
        device = img_bchw.device
        dtype = img_bchw.dtype

        sigma_h = length * 2.0  
        sigma_v = length * 0.1 + 0.3 
        
        kernel_size_h = min(max(int(sigma_h * 6), 3) | 1, w -1 | 1 ) 
        kernel_size_v = min(max(int(sigma_v * 6), 3) | 1, h -1 | 1 ) 
        kernel_size_h = max(3, kernel_size_h) | 1
        kernel_size_v = max(3, kernel_size_v) | 1

        xh = torch.linspace(-kernel_size_h//2, kernel_size_h//2, kernel_size_h, device=device, dtype=dtype)
        kernel_h_1d = torch.exp(-(xh ** 2) / (2 * sigma_h ** 2 + 1e-6))
        kernel_h_1d = kernel_h_1d / (kernel_h_1d.sum() + 1e-6)
        
        yv = torch.linspace(-kernel_size_v//2, kernel_size_v//2, kernel_size_v, device=device, dtype=dtype)
        kernel_v_1d = torch.exp(-(yv ** 2) / (2 * sigma_v ** 2 + 1e-6))
        kernel_v_1d = kernel_v_1d / (kernel_v_1d.sum() + 1e-6)
        
        num_passes = max(1, quality // 8) 
        
        convolved_img = img_bchw
        for pass_num in range(num_passes):
            k_h = kernel_h_1d.view(1, 1, 1, kernel_size_h).expand(img_bchw.shape[1], 1, 1, -1)
            convolved_img = F.conv2d(convolved_img, k_h, padding=(0, kernel_size_h//2), groups=img_bchw.shape[1])
            if num_passes > 1: convolved_img = torch.clamp(convolved_img, 0, 1)
            
            k_v = kernel_v_1d.view(1, 1, kernel_size_v, 1).expand(img_bchw.shape[1], 1, -1, 1)
            convolved_img = F.conv2d(convolved_img, k_v, padding=(kernel_size_v//2, 0), groups=img_bchw.shape[1])
            if num_passes > 1: convolved_img = torch.clamp(convolved_img, 0, 1)
            
        return convolved_img

    def _apply_chromatic_aberration(self, img, kwargs):
        strength = kwargs.get('ca_strength', 0.002)
        edge_falloff = kwargs.get('ca_edge_falloff', 1.0)
        enable_hue_shift = kwargs.get('enable_ca_hue_shift', False)
        hue_shift_degrees = kwargs.get('ca_hue_shift_degrees', 0.0)
        
        colored_print(f"      🌈 CA Config: strength={strength:.4f}, falloff={edge_falloff:.2f}", Colors.BLUE)
        if enable_hue_shift:
            colored_print(f"         Hue shift: {hue_shift_degrees:.1f}°", Colors.BLUE)
        
        if strength == 0.0:
            colored_print("      🚫 No chromatic aberration (strength = 0)", Colors.YELLOW)
            return img
        
        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype
        
        if c != 3 and enable_hue_shift:
            colored_print(f"      ⚠️ Hue shift disabled for {c}-channel image", Colors.YELLOW)
            enable_hue_shift = False

        y_coords_1d = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords_1d = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords_1d, x_coords_1d, indexing='ij')
        
        dist_from_center = torch.sqrt(grid_x**2 + grid_y**2)
        dist_from_center_normalized = dist_from_center / (math.sqrt(2.0) + 1e-6) 
        ca_displacement_scaler = strength * (dist_from_center_normalized ** edge_falloff)
        
        max_displacement = ca_displacement_scaler.max().item()
        colored_print(f"      📊 Max displacement: {max_displacement:.4f}", Colors.BLUE)
        
        r_disp_x = ca_displacement_scaler * grid_x 
        r_disp_y = ca_displacement_scaler * grid_y
        grid_r_x = grid_x + r_disp_x
        grid_r_y = grid_y + r_disp_y
        
        b_disp_x = ca_displacement_scaler * grid_x
        b_disp_y = ca_displacement_scaler * grid_y
        grid_b_x = grid_x - b_disp_x 
        grid_b_y = grid_y - b_disp_y
        
        grid_r = torch.stack([grid_r_x, grid_r_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        grid_b = torch.stack([grid_b_x, grid_b_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        
        img_bchw = img.permute(0, 3, 1, 2) 
        
        r_channel_shifted = F.grid_sample(img_bchw[:, 0:1, ...], grid_r, mode='bilinear', padding_mode='border', align_corners=True)
        g_channel_original = img_bchw[:, 1:2, ...] 
        b_channel_shifted = F.grid_sample(img_bchw[:, 2:3, ...], grid_b, mode='bilinear', padding_mode='border', align_corners=True)

        if enable_hue_shift and c == 3 and hue_shift_degrees != 0.0:
            colored_print(f"      🎨 Applying hue shift: {hue_shift_degrees:.1f}°", Colors.BLUE)
            base_red_color_vec = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            base_blue_color_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

            shifted_r_fringe_color = self._hue_shift_rgb_color_vector(base_red_color_vec, hue_shift_degrees)
            shifted_b_fringe_color = self._hue_shift_rgb_color_vector(base_blue_color_vec, -hue_shift_degrees)

            final_r_channel = r_channel_shifted * shifted_r_fringe_color[0] + \
                              b_channel_shifted * shifted_b_fringe_color[0]
            
            final_g_channel = r_channel_shifted * shifted_r_fringe_color[1] + \
                              g_channel_original * 1.0 + \
                              b_channel_shifted * shifted_b_fringe_color[1]
            
            final_b_channel = r_channel_shifted * shifted_r_fringe_color[2] + \
                              b_channel_shifted * shifted_b_fringe_color[2]
            
            result_bchw = torch.cat([final_r_channel, final_g_channel, final_b_channel], dim=1)
        else:
            result_bchw = torch.cat([r_channel_shifted, g_channel_original, b_channel_shifted], dim=1)
        
        return result_bchw.permute(0, 2, 3, 1) 

    def _apply_vignette(self, img, kwargs):
        strength = kwargs.get('vignette_strength', 0.3)
        radius = kwargs.get('vignette_radius', 1.0) 
        softness = kwargs.get('vignette_softness', 0.5) 
        
        colored_print(f"      🎭 Vignette: strength={strength:.3f}, radius={radius:.2f}, softness={softness:.2f}", Colors.BLUE)
        
        if strength == 0.0:
            colored_print("      🚫 No vignette (strength = 0)", Colors.YELLOW)
            return img
        
        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype
        
        min_dim = max(1, min(h,w))
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(h, 1) * (h / min_dim) 
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, w) * (w / min_dim) 
        
        dist_sq = x_coords**2 + y_coords**2 
        
        dist = torch.sqrt(dist_sq)
        safe_softness = max(softness, 1e-6) 
        vignette_val = torch.clamp(1.0 - (dist - radius) / safe_softness, 0.0, 1.0)
        
        vignette_mask = 1.0 - strength * (1.0 - vignette_val)
        mask_coverage = (vignette_mask < 0.9).float().mean().item()
        colored_print(f"      📊 Vignette affects {mask_coverage*100:.1f}% of image", Colors.BLUE)

        return img * vignette_mask.unsqueeze(0).unsqueeze(-1) 

    def _apply_zoom_blur(self, img, kwargs):
        strength = kwargs.get('radial_blur_strength', 0.05)
        center_x_norm = kwargs.get('radial_blur_center_x', 0.5)
        center_y_norm = kwargs.get('radial_blur_center_y', 0.5)
        falloff_param = kwargs.get('radial_blur_falloff', self.DEFAULTS['radial_blur_falloff'])
        num_samples = kwargs.get('radial_blur_samples', 16)

        colored_print(f"      🔍 Zoom Blur: strength={strength:.4f}, center=({center_x_norm:.2f},{center_y_norm:.2f}), samples={num_samples}", Colors.BLUE)

        if strength == 0.0 or num_samples <= 0:
            colored_print("      🚫 No zoom blur (strength = 0 or samples <= 0)", Colors.YELLOW)
            return img

        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing='ij'
        )

        dx_norm = x_grid - center_x_norm 
        dy_norm = y_grid - center_y_norm 
        
        result_accumulator = torch.zeros_like(img)
        img_bchw = img.permute(0, 3, 1, 2) 

        for i in range(num_samples):
            if num_samples > 1:
                t = (i / (num_samples - 1.0)) * strength - (strength / 2.0) 
            else:
                t = 0.0 

            sample_x = (x_grid + dx_norm * t) * 2.0 - 1.0
            sample_y = (y_grid + dy_norm * t) * 2.0 - 1.0
            
            grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            
            sampled_frame = F.grid_sample(img_bchw, grid, mode='bilinear', padding_mode='border', align_corners=True)
            result_accumulator += sampled_frame.permute(0, 2, 3, 1) 

        blurred_result = result_accumulator / num_samples
        
        dist_n_sq = dx_norm**2 + dy_norm**2
        
        mask_sharp_center = torch.exp(-dist_n_sq / (2.0 * (falloff_param + 1e-6)**2))
        
        blur_amount_mask = 1.0 - mask_sharp_center
        blur_amount_mask = blur_amount_mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, c)
        
        final_result = img * (1.0 - blur_amount_mask) + blurred_result * blur_amount_mask
        
        return torch.clamp(final_result, 0, 1)

    def _apply_radial_spin_blur(self, img, kwargs):
        strength_rad = kwargs.get('radial_blur_strength', 0.05) 
        center_x_norm = kwargs.get('radial_blur_center_x', 0.5)
        center_y_norm = kwargs.get('radial_blur_center_y', 0.5)
        falloff_param = kwargs.get('radial_blur_falloff', self.DEFAULTS['radial_blur_falloff'])
        num_samples = kwargs.get('radial_blur_samples', 16)

        colored_print(f"      🌀 Spin Blur: strength={strength_rad:.4f}, center=({center_x_norm:.2f},{center_y_norm:.2f}), samples={num_samples}", Colors.BLUE)

        if strength_rad == 0.0 or num_samples <= 0:
            colored_print("      🚫 No spin blur (strength = 0 or samples <= 0)", Colors.YELLOW)
            return img

        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing='ij'
        )

        rel_x = x_grid - center_x_norm 
        rel_y = y_grid - center_y_norm 
        
        result_accumulator = torch.zeros_like(img)
        img_bchw = img.permute(0, 3, 1, 2)

        for i in range(num_samples):
            if num_samples > 1:
                current_angle_rad = (i / (num_samples - 1.0)) * strength_rad - (strength_rad / 2.0)
            else:
                current_angle_rad = 0.0

            cos_a = math.cos(current_angle_rad)
            sin_a = math.sin(current_angle_rad)
            
            rot_rel_x = rel_x * cos_a - rel_y * sin_a
            rot_rel_y = rel_x * sin_a + rel_y * cos_a
            
            sample_abs_x = rot_rel_x + center_x_norm
            sample_abs_y = rot_rel_y + center_y_norm
            
            sample_grid_x = sample_abs_x * 2.0 - 1.0
            sample_grid_y = sample_abs_y * 2.0 - 1.0
            
            grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            
            sampled_frame = F.grid_sample(img_bchw, grid, mode='bilinear', padding_mode='border', align_corners=True)
            result_accumulator += sampled_frame.permute(0, 2, 3, 1)

        blurred_result = result_accumulator / num_samples

        dist_n_sq = rel_x**2 + rel_y**2 
        mask_sharp_center = torch.exp(-dist_n_sq / (2.0 * (falloff_param + 1e-6)**2))
        
        blur_amount_mask = 1.0 - mask_sharp_center
        blur_amount_mask = blur_amount_mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, c)
        
        final_result = img * (1.0 - blur_amount_mask) + blurred_result * blur_amount_mask
        return torch.clamp(final_result, 0, 1)

    def _apply_lens_distortion(self, img, kwargs):
        distortion_coeff = kwargs.get('barrel_distortion', 0.0) 
        
        colored_print(f"      🔍 Lens Distortion: coefficient={distortion_coeff:.4f}", Colors.BLUE)
        
        if distortion_coeff == 0.0:
            colored_print("      🚫 No lens distortion (coefficient = 0)", Colors.YELLOW)
            return img
        
        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype
        
        y_coords_1d = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords_1d = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords_1d, x_coords_1d, indexing='ij')

        if w >= h:
            x_for_r_sq = grid_x * (w / h)
            y_for_r_sq = grid_y
        else:
            x_for_r_sq = grid_x
            y_for_r_sq = grid_y * (h / w)
        
        r_sq = x_for_r_sq**2 + y_for_r_sq**2
        
        scale_factor = 1.0 - distortion_coeff * r_sq
        scale_factor = torch.where(scale_factor < 1e-4, torch.tensor(1e-4, device=device, dtype=dtype), scale_factor)

        src_x = grid_x / scale_factor
        src_y = grid_y / scale_factor
            
        grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

        img_bchw = img.permute(0, 3, 1, 2)
        distorted_bchw = F.grid_sample(img_bchw, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        if distortion_coeff > 0:
            colored_print("      📦 Applied barrel distortion", Colors.BLUE)
        else:
            colored_print("      📐 Applied pincushion distortion", Colors.BLUE)
        
        return distorted_bchw.permute(0, 2, 3, 1)

    def _apply_film_grain(self, img, kwargs):
        intensity = kwargs.get('grain_intensity', 0.1)
        grain_s = kwargs.get('grain_size', 1.0) 
        color_amount = kwargs.get('grain_color_amount', 0.0)
        
        colored_print(f"      📹 Film Grain: intensity={intensity:.4f}, size={grain_s:.2f}, color={color_amount:.2f}", Colors.BLUE)
        
        if intensity == 0.0:
            colored_print("      🚫 No film grain (intensity = 0)", Colors.YELLOW)
            return img
        
        b, h, w, c = img.shape
        device = img.device
        dtype = img.dtype
        
        grain_h = max(1, math.ceil(h / grain_s))
        grain_w = max(1, math.ceil(w / grain_s))
        
        colored_print(f"      🎲 Noise grid: {grain_w}x{grain_h} (original: {w}x{h})", Colors.BLUE)
        
        if c == 3 and color_amount > 0.0:
            num_noise_channels = 3
            colored_print("      🌈 Using color noise", Colors.BLUE)
        else:
            num_noise_channels = 1
            colored_print("      ⚫ Using monochrome noise", Colors.BLUE)
            
        noise = torch.randn(b, grain_h, grain_w, num_noise_channels, device=device, dtype=dtype)
        
        if num_noise_channels == 1 and c == 3: 
            noise = noise.expand(b, grain_h, grain_w, c)
        elif num_noise_channels == 3 and c == 3 and color_amount < 1.0: 
            mono_version_of_color_noise = torch.mean(noise, dim=-1, keepdim=True).expand_as(noise)
            noise = mono_version_of_color_noise * (1.0 - color_amount) + noise * color_amount
            colored_print(f"      🎨 Mixed {color_amount*100:.0f}% color with {(1-color_amount)*100:.0f}% mono", Colors.BLUE)
        
        if grain_h != h or grain_w != w:
            noise_bchw = noise.permute(0, 3, 1, 2)
            scaled_noise_bchw = F.interpolate(noise_bchw, size=(h, w), mode='bilinear', align_corners=False)
            noise = scaled_noise_bchw.permute(0, 2, 3, 1)
        
        img_lightness = torch.mean(img, dim=-1, keepdim=True)
        grain_mask = 4.0 * img_lightness * (1.0 - img_lightness) 
        
        avg_grain_strength = (grain_mask * intensity).mean().item()
        colored_print(f"      📊 Average grain strength: {avg_grain_strength:.4f}", Colors.BLUE)
        
        grained_img = img + noise * intensity * grain_mask
        
        return torch.clamp(grained_img, 0, 1)

WEB_DIRECTORY = "./js" 
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

NODE_CLASS_MAPPINGS = {
    "CRTPostProcess": CRTPostProcessNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTPostProcess": "CRT Post-Process Suite (CRT)"
}
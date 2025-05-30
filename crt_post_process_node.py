import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union
import comfy.model_management as model_management
import comfy.utils
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor
import logging

print("Loading crt-nodes module")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non-commercial upscale models.")
except:
    logging.warning("Failed to import spandrel_extra_arches; non-commercial upscale models may not be supported.")

class CRTPostProcessNode:
    """
    CRT Post-Processing Node for ComfyUI
    Combines multiple effects in optimal order for cinematic results with standalone sliders
    """
    
    # Define rescale methods for upscaling
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
                # === UPSCALE ===
                "enable_upscale": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "upscale_model_path": (folder_paths.get_filename_list("upscale_models"), {"default": "None"}),
                "downscale_by": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05}),
                "rescale_method": (cls.rescale_methods, {"default": "bicubic"}),
                "precision": (cls.precision_options, {"default": "auto"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                
                # === LEVELS (Primary) ===
                "enable_levels": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                # === COLOR WHEELS (Lift/Gamma/Gain) ===
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
                
                # === TEMPERATURE & TINT ===
                "enable_temp_tint": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "decimals": 0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "decimals": 0}),
                
                # === SHARPENING ===
                "enable_sharpen": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "sharpen_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.01, "decimals": 3}),
                "sharpen_radius": ("FLOAT", {"default": 1.85, "min": 0.1, "max": 5.0, "step": 0.1, "decimals": 1}),
                "sharpen_threshold": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                # === GLOWS ===
                "enable_small_glow": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "small_glow_intensity": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "small_glow_radius": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1, "decimals": 3}),
                "small_glow_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "enable_large_glow": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "large_glow_intensity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "large_glow_radius": ("FLOAT", {"default": 50.0, "min": 30.0, "max": 100.0, "step": 0.5, "decimals": 3}),
                "large_glow_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                # === GLARE/FLARES ===
                "enable_glare": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "glare_type": (["star_4", "star_6", "star_8", "anamorphic_h", "anamorphic_v", "cross", "ring", "bloom"], {"default": "star_4"}),
                "glare_intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 4.0, "step": 0.01, "decimals": 3}),
                "glare_length": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.01, "decimals": 3}),
                "glare_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "decimals": 0}),
                "glare_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "glare_quality": ("INT", {"default": 16, "min": 4, "max": 32, "step": 4, "decimals": 0}),
                
                # === CHROMATIC ABERRATION ===
                "enable_chromatic_aberration": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "ca_strength": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "decimals": 3}),
                "ca_edge_falloff": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                
                # === VIGNETTE ===
                "enable_vignette": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "decimals": 3}),
                "vignette_radius": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 3.0, "step": 0.01, "decimals": 3}),
                "vignette_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01, "decimals": 3}),
                
                # === RADIAL/ZOOM BLUR ===
                "enable_radial_blur": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "radial_blur_type": (cls.radial_blur_types, {"default": "spin"}),
                "radial_blur_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005, "decimals": 3}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                "radial_blur_falloff": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.1, "decimals": 3}),
                "radial_blur_samples": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8, "decimals": 0}),
                
                # === FILM GRAIN ===
                "enable_film_grain": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "grain_intensity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.15, "step": 0.01, "decimals": 3}),
                "grain_size": ("FLOAT", {"default": 0.03, "min": 0.03, "max": 2.0, "step": 0.01, "decimals": 3}),
                "grain_color_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3}),
                
                # === LENS DISTORTION ===
                "enable_lens_distortion": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "barrel_distortion": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001, "decimals": 3}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    # Define default values and operation groups for reset
    DEFAULTS = {
        "downscale_by": 0.5,
        "batch_size": 1,
        "exposure": 0.0, "gamma": 1.0, "brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "vibrance": 0.0,
        "lift_r": 0.0, "lift_g": 0.0, "lift_b": 0.0, "gamma_r": 1.0, "gamma_g": 1.0, "gamma_b": 1.0,
        "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0, "temperature": 0.0, "tint": 0.0,
        "sharpen_strength": 2.5, "sharpen_radius": 1.85, "sharpen_threshold": 0.015,
        "small_glow_intensity": 0.015, "small_glow_radius": 0.1, "small_glow_threshold": 0.25,
        "large_glow_intensity": 0.25, "large_glow_radius": 50.0, "large_glow_threshold": 0.3,
        "glare_intensity": 0.65, "glare_length": 1.5, "glare_angle": 0.0, "glare_threshold": 0.95,
        "ca_strength": 0.005, "ca_edge_falloff": 2.0, "vignette_strength": 0.5,
        "vignette_radius": 0.7, "vignette_softness": 2.0, "radial_blur_type": "spin",
        "radial_blur_strength": 0.02, "radial_blur_center_x": 0.5, "radial_blur_center_y": 0.25,
        "radial_blur_falloff": 0.05, "radial_blur_samples": 16,
        "grain_intensity": 0.02, "grain_size": 0.03, "grain_color_amount": 0.0,
        "barrel_distortion": 0.0
    }

    OPERATION_GROUPS = {
        "UPSCALE": ["enable_upscale", "upscale_model_path", "downscale_by", "rescale_method", "precision", "batch_size"],
        "LEVELS": ["enable_levels", "exposure", "gamma", "brightness", "contrast", "saturation", "vibrance"],
        "COLOR WHEELS": ["enable_color_wheels", "lift_r", "lift_g", "lift_b", "gamma_r", "gamma_g", "gamma_b", "gain_r", "gain_g", "gain_b"],
        "TEMPERATURE & TINT": ["enable_temp_tint", "temperature", "tint"],
        "SHARPENING": ["enable_sharpen", "sharpen_strength", "sharpen_radius", "sharpen_threshold"],
        "GLOWS": ["enable_small_glow", "small_glow_intensity", "small_glow_radius", "small_glow_threshold", "enable_large_glow", "large_glow_intensity", "large_glow_radius", "large_glow_threshold"],
        "GLARE/FLARES": ["enable_glare", "glare_type", "glare_intensity", "glare_length", "glare_angle", "glare_threshold", "glare_quality"],
        "CHROMATIC ABERRATION": ["enable_chromatic_aberration", "ca_strength", "ca_edge_falloff"],
        "VIGNETTE": ["enable_vignette", "vignette_strength", "vignette_radius", "vignette_softness"],
        "RADIAL BLUR": ["enable_radial_blur", "radial_blur_type", "radial_blur_strength", "radial_blur_center_x", "radial_blur_center_y", "radial_blur_falloff", "radial_blur_samples"],
        "LENS DISTORTION": ["enable_lens_distortion", "barrel_distortion"],
        "FILM GRAIN": ["enable_film_grain", "grain_intensity", "grain_size", "grain_color_amount"]
    }

    def __init__(self):
        self.slider_positions = {}  # For slider UI, managed by JavaScript
        self.capture = False
        self.unlock = False
        self.loaded_upscale_model = None
        self.current_model_path = None

    def load_upscale_model(self, model_path):
        """Load the upscale model from the specified path"""
        if model_path == "None" or not model_path:
            self.loaded_upscale_model = None
            self.current_model_path = None
            logger.debug("No upscale model selected, skipping load")
            return None

        if self.current_model_path == model_path and self.loaded_upscale_model is not None:
            logger.debug(f"Upscale model {model_path} already loaded, using cached model")
            return self.loaded_upscale_model

        try:
            model_full_path = folder_paths.get_full_path_or_raise("upscale_models", model_path)
            logger.info(f"Loading upscale model from {model_full_path}")

            # Load the state dict
            sd = comfy.utils.load_torch_file(model_full_path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

            # Load the model using ModelLoader
            out = ModelLoader().load_from_state_dict(sd).eval()

            if not isinstance(out, ImageModelDescriptor):
                raise Exception("Upscale model must be a single-image model.")

            self.loaded_upscale_model = out
            self.current_model_path = model_path
            logger.info(f"Successfully loaded upscale model: {model_path}")
            return self.loaded_upscale_model

        except Exception as e:
            logger.error(f"Failed to load upscale model from {model_path}: {e}")
            self.loaded_upscale_model = None
            self.current_model_path = None
            return None

    def reset_operation(self, operation_name):
        """Reset all parameters for a given operation group to their defaults"""
        for param in self.OPERATION_GROUPS.get(operation_name, []):
            if param in self.DEFAULTS:
                widget = next((w for w in self.widgets if w.name == param), None)
                if widget:
                    widget.value = self.DEFAULTS[param]
                    self.setDirtyCanvas(True, True)
        logger.debug(f"Reset operation: {operation_name} to defaults")

    def process(self, image, **kwargs):
        device = model_management.get_torch_device()
        image = image.to(device)
        
        logger.debug(f"Processing with kwargs: {kwargs}")
        
        # Load the upscale model if needed
        if kwargs.get('enable_upscale', False):
            model_path = kwargs.get('upscale_model_path', "None")
            self.load_upscale_model(model_path)
        
        # Process each frame in batch
        processed_frames = []
        for i in range(image.shape[0]):
            frame = image[i:i+1]
            processed_frame = self._process_single_frame(frame, device, **kwargs)
            processed_frames.append(processed_frame)
        
        result = torch.cat(processed_frames, dim=0)
        return (result,)
    
    def _process_single_frame(self, image, device, **kwargs):
        """
        Process single frame with all effects in optimal order
        Order follows professional post-production pipeline
        """
        img = image.clone()
        
        logger.debug("Starting frame processing")
        
        # 1. UPSCALE - Early in the pipeline to ensure quality for subsequent effects
        if kwargs.get('enable_upscale', False):
            logger.debug("Applying upscale")
            img = self._apply_upscale(img, device, **kwargs)
        
        # 2. LENS DISTORTION CORRECTION - Early correction
        if kwargs.get('enable_lens_distortion', False):
            logger.debug("Applying lens distortion")
            img = self._apply_lens_distortion(img, kwargs)
        
        # 3. CHROMATIC ABERRATION - Before color grading
        if kwargs.get('enable_chromatic_aberration', False):
            logger.debug("Applying chromatic aberration")
            img = self._apply_chromatic_aberration(img, kwargs)
        
        # 4. TEMPERATURE & TINT - Primary color correction
        if kwargs.get('enable_temp_tint', False):
            logger.debug("Applying temperature and tint")
            img = self._apply_temperature_tint(img, kwargs)
        
        # 5. LEVELS - Primary adjustments
        if kwargs.get('enable_levels', False):
            logger.debug("Applying levels")
            img = self._apply_levels(img, kwargs)
        
        # 6. COLOR WHEELS - Advanced color grading
        if kwargs.get('enable_color_wheels', False):
            logger.debug("Applying color wheels")
            img = self._apply_color_wheels(img, kwargs)
        
        # 7. SHARPENING - Before blur effects
        if kwargs.get('enable_sharpen', False):
            logger.debug("Applying sharpening")
            img = self._apply_sharpening(img, kwargs)
        
        # 8. GLOWS - Light effects
        if kwargs.get('enable_small_glow', False):
            logger.debug("Applying small glow")
            img = self._apply_glow(img, kwargs, 'small')
        
        if kwargs.get('enable_large_glow', False):
            logger.debug("Applying large glow")
            img = self._apply_glow(img, kwargs, 'large')
        
        # 9. GLARE/FLARES - Dramatic light effects
        if kwargs.get('enable_glare', False):
            logger.debug("Applying glare")
            img = self._apply_glare(img, kwargs)
        
        # 10. RADIAL BLUR - Selective focus effects
        if kwargs.get('enable_radial_blur', False):
            logger.debug("Applying radial blur")
            blur_type = kwargs.get('radial_blur_type', 'zoom')
            if blur_type == 'zoom':
                img = self._apply_zoom_blur(img, kwargs)
            elif blur_type == 'spin':
                img = self._apply_radial_spin_blur(img, kwargs)
        
        # 11. VIGNETTE - Frame darkening
        if kwargs.get('enable_vignette', False):
            logger.debug("Applying vignette")
            img = self._apply_vignette(img, kwargs)
        
        # 12. FILM GRAIN - Final texture
        if kwargs.get('enable_film_grain', False):
            logger.debug("Applying film grain")
            img = self._apply_film_grain(img, kwargs)
        
        logger.debug("Frame processing completed")
        return img

    def _apply_upscale(self, img, device, **kwargs):
        """Apply upscaling using the loaded model"""
        downscale_by = kwargs.get('downscale_by', 1.0)
        rescale_method = kwargs.get('rescale_method', 'bicubic')
        precision = kwargs.get('precision', 'auto')
        batch_size = kwargs.get('batch_size', 1)

        if self.loaded_upscale_model is None:
            logger.warning("No upscale model loaded, skipping upscale step")
            return img

        original_device = img.device
        original_dtype = img.dtype

        # Determine the appropriate dtype based on precision and device
        if precision == "auto":
            dtype = torch.float16 if device.type == "cuda" else torch.float32
        elif precision == "16":
            dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        elif precision == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Ensure the chosen dtype is supported on the current device
        if dtype == torch.float16 and device.type != "cuda":
            logger.warning("float16 is not supported on CPU. Falling back to bfloat16.")
            dtype = torch.bfloat16

        upscale_model = self.loaded_upscale_model.to(device)

        # Memory management and tiling
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * img.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += img.nelement() * img.element_size()
        model_management.free_memory(memory_required, device)

        in_img = img.movedim(-1, 1).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(1, -1), min=0, max=1.0)

        if downscale_by < 1.0:
            target_height = round(s.shape[1] * downscale_by)
            target_width = round(s.shape[2] * downscale_by)
            s = s.permute(0, 3, 1, 2).contiguous()
            s = torch.nn.functional.interpolate(
                s,
                size=(target_height, target_width),
                mode=rescale_method if rescale_method != "lanczos" else "bicubic",
                align_corners=False if rescale_method in ["bilinear", "bicubic"] else None
            )
            s = s.permute(0, 2, 3, 1).contiguous()

        if dtype != original_dtype or downscale_by < 1.0:
            s = s.clamp(0, 1).to(original_dtype).to(original_device)

        logger.info(f"Upscaling complete for frame.")
        return s

    def _apply_temperature_tint(self, img, kwargs):
        """Apply temperature and tint adjustments"""
        temp = kwargs.get('temperature', 0.0) / 100.0
        tint = kwargs.get('tint', 0.0) / 100.0
        
        if temp == 0.0 and tint == 0.0:
            return img
        
        # Temperature adjustment (blue/orange)
        if temp != 0.0:
            if temp > 0:  # Warmer (more orange/red)
                img[..., 0] *= (1.0 + temp * 0.3)  # Red
                img[..., 2] *= (1.0 - temp * 0.2)  # Blue
            else:  # Cooler (more blue)
                img[..., 0] *= (1.0 + temp * 0.2)  # Red
                img[..., 2] *= (1.0 - temp * 0.3)  # Blue
        
        # Tint adjustment (green/magenta)
        if tint != 0.0:
            if tint > 0:  # More green
                img[..., 1] *= (1.0 + tint * 0.3)  # Green
            else:  # More magenta
                img[..., 0] *= (1.0 - tint * 0.15)  # Red
                img[..., 2] *= (1.0 - tint * 0.15)  # Blue
        
        return torch.clamp(img, 0, 1)

    def _apply_levels(self, img, kwargs):
        """Apply exposure, gamma, brightness, contrast, saturation, vibrance"""
        exposure = kwargs.get('exposure', 0.0)
        gamma = kwargs.get('gamma', 1.0)
        brightness = kwargs.get('brightness', 0.0)
        contrast = kwargs.get('contrast', 1.0)
        saturation = kwargs.get('saturation', 1.0)
        vibrance = kwargs.get('vibrance', 0.0)
        
        # Exposure (multiplicative)
        if exposure != 0.0:
            img *= (2.0 ** exposure)
        
        # Gamma correction
        if gamma != 1.0:
            img = torch.pow(torch.clamp(img, 0.001, 1.0), 1.0 / gamma)
        
        # Brightness (additive)
        if brightness != 0.0:
            img += brightness
        
        # Contrast (around 0.5 middle gray)
        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5
        
        # Saturation
        if saturation != 1.0:
            gray = torch.mean(img, dim=-1, keepdim=True)
            img = gray + (img - gray) * saturation
        
        # Vibrance (selective saturation boost for less saturated areas)
        if vibrance != 0.0:
            gray = torch.mean(img, dim=-1, keepdim=True)
            current_sat = torch.abs(img - gray).max(dim=-1, keepdim=True)[0]
            vibrance_mask = 1.0 - current_sat
            img = gray + (img - gray) * (1.0 + vibrance * vibrance_mask)
        
        return torch.clamp(img, 0, 1)

    def _apply_color_wheels(self, img, kwargs):
        """Apply lift/gamma/gain color correction"""
        lift = torch.tensor([kwargs.get('lift_r', 0.0), 
                           kwargs.get('lift_g', 0.0), 
                           kwargs.get('lift_b', 0.0)], device=img.device)
        
        gamma = torch.tensor([kwargs.get('gamma_r', 1.0), 
                            kwargs.get('gamma_g', 1.0), 
                            kwargs.get('gamma_b', 1.0)], device=img.device)
        
        gain = torch.tensor([kwargs.get('gain_r', 1.0), 
                           kwargs.get('gain_g', 1.0), 
                           kwargs.get('gain_b', 1.0)], device=img.device)
        
        # Apply lift (shadows)
        img = img + lift * (1.0 - img)
        
        # Apply gamma (midtones)
        img = torch.pow(torch.clamp(img, 0.001, 1.0), 1.0 / gamma)
        
        # Apply gain (highlights)
        img = img * gain
        
        return torch.clamp(img, 0, 1)

    def _apply_sharpening(self, img, kwargs):
        """Apply unsharp mask sharpening"""
        strength = kwargs.get('sharpen_strength', 0.5)
        radius = kwargs.get('sharpen_radius', 1.0)
        threshold = kwargs.get('sharpen_threshold', 0.0)
        
        if strength == 0.0:
            return img
        
        # Convert to BCHW for convolution
        img_conv = img.permute(0, 3, 1, 2)
        
        # Create Gaussian blur kernel
        kernel_size = max(3, int(radius * 4) | 1)  # Ensure odd and minimum size
        sigma = radius / 3.0
        
        # Generate 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable Gaussian blur
        blurred = img_conv
        kernel_1d = kernel_1d.view(1, 1, -1, 1)
        blurred = F.conv2d(blurred, kernel_1d.expand(3, 1, -1, 1), 
                          padding=(kernel_size//2, 0), groups=3)
        
        kernel_1d = kernel_1d.view(1, 1, 1, -1)
        blurred = F.conv2d(blurred, kernel_1d.expand(3, 1, 1, -1), 
                          padding=(0, kernel_size//2), groups=3)
        
        # Calculate unsharp mask
        blurred = blurred.permute(0, 2, 3, 1)
        unsharp = img - blurred
        
        # Apply threshold
        if threshold > 0:
            mask = torch.abs(unsharp) > threshold
            unsharp = unsharp * mask.float()
        
        # Apply sharpening
        return torch.clamp(img + unsharp * strength, 0, 1)

    def _apply_glow(self, img, kwargs, glow_type):
        """Apply glow effect (small or large)"""
        if glow_type == 'small':
            intensity = kwargs.get('small_glow_intensity', 0.3)
            radius = kwargs.get('small_glow_radius', 2.0)
            threshold = kwargs.get('small_glow_threshold', 0.7)
        else:
            intensity = kwargs.get('large_glow_intensity', 0.2)
            radius = kwargs.get('large_glow_radius', 8.0)
            threshold = kwargs.get('large_glow_threshold', 0.8)
        
        if intensity == 0.0:
            return img
        
        # Extract bright areas
        brightness = torch.mean(img, dim=-1, keepdim=True)
        glow_mask = torch.clamp((brightness - threshold) / (1.0 - threshold), 0, 1)
        glow_source = img * glow_mask
        
        # Apply blur to create glow
        glow_source = glow_source.permute(0, 3, 1, 2)
        
        # Multiple blur passes for smooth glow
        kernel_size = max(3, int(radius * 2) | 1)
        sigma = radius / 2.0
        
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable Gaussian blur
        glow = glow_source
        kernel_1d = kernel_1d.view(1, 1, -1, 1)
        glow = F.conv2d(glow, kernel_1d.expand(3, 1, -1, 1), 
                       padding=(kernel_size//2, 0), groups=3)
        
        kernel_1d = kernel_1d.view(1, 1, 1, -1)
        glow = F.conv2d(glow, kernel_1d.expand(3, 1, 1, -1), 
                       padding=(0, kernel_size//2), groups=3)
        
        glow = glow.permute(0, 2, 3, 1)
        
        # Blend glow with original image
        return torch.clamp(img + glow * intensity, 0, 1)

    def _apply_glare(self, img, kwargs):
        """Apply star/anamorphic glare effects with improved quality and options"""
        logger.debug("Entering _apply_glare")
        glare_type = kwargs.get('glare_type', 'star_4')
        intensity = kwargs.get('glare_intensity', 0.65)
        length = kwargs.get('glare_length', 1.5)
        angle = kwargs.get('glare_angle', 0.0)
        threshold = kwargs.get('glare_threshold', 0.015)
        quality = kwargs.get('glare_quality', 16)
        
        if intensity == 0.0:
            return img
        
        # Extract bright areas
        brightness = torch.mean(img, dim=-1, keepdim=True)
        glare_mask = torch.clamp((brightness - threshold) / (1.0 - threshold), 0, 1)
        glare_source = img * glare_mask
        
        # Convert to BCHW for convolution
        glare_source = glare_source.permute(0, 3, 1, 2)
        h, w = glare_source.shape[2], glare_source.shape[3]
        
        if 'star' in glare_type:
            rays = int(glare_type.split('_')[1])
            glare_effect = self._create_star_glare(glare_source, rays, length, angle, h, w, quality)
        elif 'anamorphic' in glare_type:
            horizontal = 'h' in glare_type
            glare_effect = self._create_anamorphic_glare(glare_source, length, angle, horizontal, h, w, quality)
        elif glare_type == 'ring':
            glare_effect = self._create_ring_glare(glare_source, length, angle, h, w, quality)
        elif glare_type == 'bloom':
            glare_effect = self._create_bloom_glare(glare_source, length, angle, h, w, quality)
        else:  # cross
            glare_effect = self._create_cross_glare(glare_source, length, angle, h, w, quality)
        
        glare_effect = glare_effect.permute(0, 2, 3, 1)
        # Apply fade-out based on distance from bright areas
        dist_from_bright = 1.0 - glare_mask
        fade_factor = torch.exp(-dist_from_bright * 5.0)
        return torch.clamp(img + glare_effect * intensity * fade_factor, 0, 1)

    def _create_star_glare(self, source, rays, length, angle, h, w, quality):
        """Create star-shaped glare pattern with improved quality"""
        logger.debug(f"Creating star glare with rays: {rays}, length: {length}, angle: {angle}, quality: {quality}")
        glare_result = torch.zeros_like(source)
        
        # Create rays at different angles with higher quality sampling
        for i in range(rays):
            ray_angle = (angle + i * 180.0 / rays) * math.pi / 180.0
            kernel_size = max(3, int(length * min(h, w) * 0.1 * (quality / 16)) | 1)  # Scale with quality
            logger.debug(f"Kernel size for star glare: {kernel_size}")
            
            motion_kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=source.device)
            center = kernel_size // 2
            
            for j in range(kernel_size):
                offset = j - center
                x_pos = center + int(offset * math.cos(ray_angle))
                y_pos = center + int(offset * math.sin(ray_angle))
                if 0 <= x_pos < kernel_size and 0 <= y_pos < kernel_size:
                    motion_kernel[0, 0, y_pos, x_pos] = 1.0 / quality  # Distribute weight
            
            motion_kernel = motion_kernel / motion_kernel.sum()
            padding = (kernel_size//2, kernel_size//2)
            ray_blur = F.conv2d(source, motion_kernel.expand(3, 1, -1, -1), 
                               padding=padding, groups=3)
            
            glare_result += ray_blur / rays
        
        return glare_result

    def _create_anamorphic_glare(self, source, length, angle, horizontal, h, w, quality):
        """Create anamorphic lens flare with distinct horizontal/vertical streaks"""
        logger.debug(f"Creating anamorphic glare, horizontal: {horizontal}, length: {length}, quality: {quality}")
        blur_size = max(3, int(length * min(h, w) * 0.1 * (quality / 16)) | 1)
        logger.debug(f"Blur size for anamorphic glare: {blur_size}")
        
        # Create directional kernel with angle adjustment
        kernel = torch.zeros(1, 1, blur_size, blur_size, device=source.device)
        center = blur_size // 2
        angle_rad = angle * math.pi / 180.0
        
        for i in range(blur_size):
            for j in range(blur_size):
                dx = i - center
                dy = j - center
                if horizontal:
                    weight = math.exp(-abs(dx) / (blur_size / 4)) if abs(dx) < blur_size / 2 else 0
                else:
                    weight = math.exp(-abs(dy) / (blur_size / 4)) if abs(dy) < blur_size / 2 else 0
                kernel[0, 0, i, j] = weight * math.cos(angle_rad * (dx + dy) / blur_size)
        
        kernel = kernel / kernel.sum()
        padding = (blur_size//2, blur_size//2) if horizontal else (0, blur_size//2)
        flare = F.conv2d(source, kernel.expand(3, 1, -1, -1), 
                        padding=padding, groups=3)
        
        return flare

    def _create_ring_glare(self, source, length, angle, h, w, quality):
        """Create a ring-shaped glare effect"""
        logger.debug(f"Creating ring glare, length: {length}, quality: {quality}")
        kernel_size = max(3, int(length * min(h, w) * 0.1 * (quality / 16)) | 1)
        center = kernel_size // 2
        ring = torch.zeros(1, 1, kernel_size, kernel_size, device=source.device)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx = i - center
                dy = j - center
                dist = math.sqrt(dx**2 + dy**2)
                if 0.3 * kernel_size <= dist <= 0.7 * kernel_size:
                    ring[0, 0, i, j] = torch.exp(-dist / (kernel_size / 4))
        
        ring = ring / ring.sum()
        padding = (kernel_size//2, kernel_size//2)
        return F.conv2d(source, ring.expand(3, 1, -1, -1), 
                       padding=padding, groups=3)

    def _create_bloom_glare(self, source, length, angle, h, w, quality):
        """Create a soft bloom effect"""
        logger.debug(f"Creating bloom glare, length: {length}, quality: {quality}")
        kernel_size = max(5, int(length * min(h, w) * 0.1 * (quality / 16)) | 1)
        sigma = kernel_size / 6.0
        coords = torch.arange(kernel_size, dtype=torch.float32, device=source.device) - kernel_size // 2
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        padding = (kernel_size//2, kernel_size//2)
        return F.conv2d(source, kernel.view(1, 1, kernel_size, kernel_size).expand(3, 1, -1, -1), 
                       padding=padding, groups=3)

    def _create_cross_glare(self, source, length, angle, h, w, quality):
        """Create cross-shaped glare with improved quality"""
        logger.debug("Creating cross glare")
        h_flare = self._create_anamorphic_glare(source, length, angle, True, h, w, quality)
        v_flare = self._create_anamorphic_glare(source, length, angle + 90, False, h, w, quality)
        return (h_flare + v_flare) * 0.5

    def _apply_chromatic_aberration(self, img, kwargs):
        """Apply chromatic aberration effect on edges"""
        strength = kwargs.get('ca_strength', 0.002)
        edge_falloff = kwargs.get('ca_edge_falloff', 1.0)
        
        if strength == 0.0:
            return img
        
        h, w = img.shape[1:3]
        device = img.device
        
        # Create distance from center map
        y_coords = torch.linspace(-1, 1, h, device=device).view(-1, 1)
        x_coords = torch.linspace(-1, 1, w, device=device).view(1, -1)
        dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Apply edge falloff
        ca_amount = strength * (dist_from_center ** edge_falloff)
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                                       torch.linspace(-1, 1, w, device=device), indexing='ij')
        
        # Calculate displacement for each channel
        red_offset = ca_amount * 0.5
        blue_offset = -ca_amount * 0.5
        
        # Red channel (outward)
        red_grid_x = grid_x + red_offset * grid_x
        red_grid_y = grid_y + red_offset * grid_y
        red_grid = torch.stack([red_grid_x, red_grid_y], dim=-1).unsqueeze(0)
        
        # Blue channel (inward)
        blue_grid_x = grid_x + blue_offset * grid_x
        blue_grid_y = grid_y + blue_offset * grid_y
        blue_grid = torch.stack([blue_grid_x, blue_grid_y], dim=-1).unsqueeze(0)
        
        # Sample channels
        img_permuted = img.permute(0, 3, 1, 2)
        
        red_channel = F.grid_sample(img_permuted[:, 0:1], red_grid, 
                                   mode='bilinear', padding_mode='border', align_corners=True)
        green_channel = img_permuted[:, 1:2]  # No displacement
        blue_channel = F.grid_sample(img_permuted[:, 2:3], blue_grid, 
                                    mode='bilinear', padding_mode='border', align_corners=True)
        
        # Combine channels
        result = torch.cat([red_channel, green_channel, blue_channel], dim=1)
        return result.permute(0, 2, 3, 1)

    def _apply_vignette(self, img, kwargs):
        """Apply vignette effect"""
        strength = kwargs.get('vignette_strength', 0.3)
        radius = kwargs.get('vignette_radius', 1.0)
        softness = kwargs.get('vignette_softness', 0.5)
        
        if strength == 0.0:
            return img
        
        h, w = img.shape[1:3]
        device = img.device
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, h, device=device).view(-1, 1)
        x_coords = torch.linspace(-1, 1, w, device=device).view(1, -1)
        
        # Calculate distance from center
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Create vignette mask
        vignette_mask = 1.0 - torch.clamp((dist - radius) / softness, 0, 1)
        vignette_mask = 1.0 - strength * (1.0 - vignette_mask)
        
        # Apply vignette
        return img * vignette_mask.unsqueeze(-1)

    def _apply_zoom_blur(self, img, kwargs):
        """Apply zoom blur effect"""
        strength = kwargs.get('radial_blur_strength', 0.05)
        center_x_norm = kwargs.get('radial_blur_center_x', 0.5)
        center_y_norm = kwargs.get('radial_blur_center_y', 0.5)
        falloff = kwargs.get('radial_blur_falloff', 0.3)
        num_samples = kwargs.get('radial_blur_samples', 16)
        if strength == 0.0:
            return img
        b, h, w, c = img.shape
        device = img.device
        y_norm = torch.linspace(0, 1, h, device=device).view(-1, 1).repeat(1, w)
        x_norm = torch.linspace(0, 1, w, device=device).view(1, -1).repeat(h, 1)
        dx_n = x_norm - center_x_norm
        dy_n = y_norm - center_y_norm
        dist_n = torch.sqrt(dx_n**2 + dy_n**2).clamp_min(1e-6)
        dir_x_n = dx_n / dist_n
        dir_y_n = dy_n / dist_n
        blur_disp_n = strength * dist_n
        # Calculate distance to nearest edge
        dist_to_left = x_norm
        dist_to_right = 1.0 - x_norm
        dist_to_top = y_norm
        dist_to_bottom = 1.0 - y_norm
        dist_from_edge = torch.stack([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom], dim=0).min(dim=0)[0]
        result = torch.zeros_like(img)
        img_p = img.permute(0, 3, 1, 2)
        for i in range(num_samples):
            t = (i / (num_samples - 1.0)) * 2.0 - 1.0 if num_samples > 1 else 0.0
            curr_disp_n = blur_disp_n * t
            samp_x_n = (x_norm + curr_disp_n * dir_x_n) * 2.0 - 1.0
            samp_y_n = (y_norm + curr_disp_n * dir_y_n) * 2.0 - 1.0
            grid = torch.stack([samp_x_n, samp_y_n], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            result += F.grid_sample(img_p, grid, mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1)
        result /= num_samples
        # Falloff starts from edges (stronger blur at center)
        falloff_mask = torch.exp(-dist_from_edge**2 / (2.0 * (falloff + 1e-6)**2))
        falloff_mask = falloff_mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, c)
        return torch.clamp(img * (1.0 - falloff_mask) + result * falloff_mask, 0, 1)

    def _apply_radial_spin_blur(self, img, kwargs):
        """Apply radial spin blur effect"""
        strength_rad = kwargs.get('radial_blur_strength', 0.05)
        center_x_norm = kwargs.get('radial_blur_center_x', 0.5)
        center_y_norm = kwargs.get('radial_blur_center_y', 0.5)
        falloff = kwargs.get('radial_blur_falloff', 0.3)
        num_samples = kwargs.get('radial_blur_samples', 16)
        if strength_rad == 0.0:
            return img
        b, h, w, c = img.shape
        device = img.device
        y_norm = torch.linspace(0, 1, h, device=device).view(-1, 1).repeat(1, w)
        x_norm = torch.linspace(0, 1, w, device=device).view(1, -1).repeat(h, 1)
        rel_x_n = x_norm - center_x_norm
        rel_y_n = y_norm - center_y_norm
        dist_n = torch.sqrt(rel_x_n**2 + rel_y_n**2)
        # Calculate distance to nearest edge
        dist_to_left = x_norm
        dist_to_right = 1.0 - x_norm
        dist_to_top = y_norm
        dist_to_bottom = 1.0 - y_norm
        dist_from_edge = torch.stack([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom], dim=0).min(dim=0)[0]
        result = torch.zeros_like(img)
        img_p = img.permute(0, 3, 1, 2)
        for i in range(num_samples):
            t = (i / (num_samples - 1.0)) * 2.0 - 1.0 if num_samples > 1 else 0.0
            curr_angle_rad = strength_rad * t
            cos_a = math.cos(curr_angle_rad)
            sin_a = math.sin(curr_angle_rad)
            rot_rel_x_n = rel_x_n * cos_a - rel_y_n * sin_a
            rot_rel_y_n = rel_x_n * sin_a + rel_y_n * cos_a
            samp_x_n = (rot_rel_x_n + center_x_norm) * 2.0 - 1.0
            samp_y_n = (rot_rel_y_n + center_y_norm) * 2.0 - 1.0
            grid = torch.stack([samp_x_n, samp_y_n], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            result += F.grid_sample(img_p, grid, mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1)
        result /= num_samples
        # Falloff starts from edges (stronger blur at center)
        falloff_mask = torch.exp(-dist_from_edge**2 / (2.0 * (falloff + 1e-6)**2))
        falloff_mask = falloff_mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, c)
        return torch.clamp(img * (1.0 - falloff_mask) + result * falloff_mask, 0, 1)

    def _apply_radial_blur(self, img, kwargs):
        """Apply radial blur effect similar to DaVinci Resolve"""
        strength = kwargs.get('radial_blur_strength', 0.05)
        center_x = kwargs.get('radial_blur_center_x', 0.5)
        center_y = kwargs.get('radial_blur_center_y', 0.5)
        falloff = kwargs.get('radial_blur_falloff', 2.0)
        num_samples = kwargs.get('radial_blur_samples', 16)
        
        if strength == 0.0:
            return img
        
        h, w = img.shape[1:3]
        device = img.device
        
        # Create normalized coordinate grid in [0, 1] space
        y_coords = torch.linspace(0, 1, h, device=device).view(-1, 1)
        x_coords = torch.linspace(0, 1, w, device=device).view(1, -1)
        
        # Calculate direction vectors from center to each pixel
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = torch.sqrt(dx**2 + dy**2)
        
        # Normalize direction vectors (avoid division by zero)
        direction_x = torch.where(dist > 0, dx / dist, torch.zeros_like(dx))
        direction_y = torch.where(dist > 0, dy / dist, torch.zeros_like(dy))
        
        # Compute blur amount based on distance and strength
        # The blur amount increases with distance from the center
        blur_amount = strength * dist
        
        # Apply falloff to control the intensity of the blur
        falloff_factor = torch.exp(-dist**2 / (2 * falloff**2))
        blur_amount = blur_amount * falloff_factor
        
        # Create the blurred image by sampling along the radial direction
        result = torch.zeros_like(img)
        
        # Sample along the radial direction for each pixel
        for i in range(num_samples):
            # Linearly interpolate sample positions from -blur_amount to +blur_amount
            t = (i / (num_samples - 1)) * 2 - 1  # t ranges from -1 to 1
            offset_x = blur_amount * direction_x * t
            offset_y = blur_amount * direction_y * t
            
            # Compute sampling coordinates in [-1, 1] space for grid_sample
            sample_x = (x_coords + offset_x) * 2 - 1
            sample_y = (y_coords + offset_y) * 2 - 1
            
            # Create sampling grid
            grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
            
            # Sample the image
            img_permuted = img.permute(0, 3, 1, 2)
            sampled = F.grid_sample(img_permuted, grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)
            result += sampled.permute(0, 2, 3, 1)
        
        # Normalize the result by the number of samples
        result = result / num_samples
        
        # Blend with the original image using the falloff factor
        # This preserves sharpness near the center or edges based on falloff
        blend_mask = falloff_factor.unsqueeze(-1)
        return torch.clamp(img * (1 - blend_mask) + result * blend_mask, 0, 1)

    def _apply_lens_distortion(self, img, kwargs):
        """Apply barrel distortion with offset and crop for positive values"""
        barrel = kwargs.get('barrel_distortion', 0.0)
        
        if barrel == 0.0:
            return img
        
        h, w = img.shape[1:3]
        device = img.device
        
        # Create normalized coordinate grid
        y_coords = torch.linspace(-1, 1, h, device=device).view(-1, 1)
        x_coords = torch.linspace(-1, 1, w, device=device).view(1, -1)
        
        # Calculate radius from center
        r = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Apply barrel distortion: r' = r * (1 + k * r^2)
        r_distorted = r * (1.0 + barrel * r**2)
        
        # Avoid division by zero
        scale_factor = torch.where(r > 0.001, r_distorted / r, torch.ones_like(r))
        
        # Offset and crop for positive barrel distortion
        if barrel > 0:
            # Calculate maximum distortion at edges
            max_dist = r_distorted.max()
            crop_factor = (max_dist - 1.0) / 2.0  # Half the difference to center the image
            crop_amount = int(crop_factor * min(h, w))
            if crop_amount > 0:
                img = img[:, crop_amount:h-crop_amount, crop_amount:w-crop_amount, :]
        
        # Apply distortion to coordinates
        x_distorted = x_coords * scale_factor
        y_distorted = y_coords * scale_factor
        
        # Create grid for sampling
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Sample distorted image
        img_permuted = img.permute(0, 3, 1, 2)
        distorted = F.grid_sample(img_permuted, grid, mode='bilinear', 
                                 padding_mode='border', align_corners=True)
        
        return distorted.permute(0, 2, 3, 1)

    def _apply_film_grain(self, img, kwargs):
        """Apply film grain texture"""
        intensity = kwargs.get('grain_intensity', 0.1)
        size = kwargs.get('grain_size', 1.0)
        color_amount = kwargs.get('grain_color_amount', 0.0)
        
        if intensity == 0.0:
            return img
        
        h, w = img.shape[1:3]
        device = img.device
        
        # Generate grain at different scale if needed
        grain_h = max(1, int(h / size))
        grain_w = max(1, int(w / size))
        
        # Generate noise
        if color_amount > 0:
            # Colored grain
            grain = torch.randn(img.shape[0], grain_h, grain_w, 3, device=device)
        else:
            # Monochrome grain
            grain = torch.randn(img.shape[0], grain_h, grain_w, 1, device=device)
            grain = grain.expand(-1, -1, -1, 3)
        
        # Resize grain to match image size if needed
        if size != 1.0:
            grain = grain.permute(0, 3, 1, 2)
            grain = F.interpolate(grain, size=(h, w), mode='bilinear', align_corners=False)
            grain = grain.permute(0, 2, 3, 1)
        
        # Apply grain based on image brightness (film characteristic)
        brightness = torch.mean(img, dim=-1, keepdim=True)
        grain_mask = brightness * (1.0 - brightness) * 4.0  # Parabolic response
        
        # Mix color and monochrome grain
        if color_amount < 1.0:
            mono_grain = torch.mean(grain, dim=-1, keepdim=True).expand_as(grain)
            grain = mono_grain * (1.0 - color_amount) + grain * color_amount
        
        # Apply grain
        grain_effect = grain * intensity * grain_mask
        return torch.clamp(img + grain_effect, 0, 1)

# JavaScript extension for custom slider UI
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CRTPostProcess": CRTPostProcessNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTPostProcess": "CRT Post-Process Suite"
}
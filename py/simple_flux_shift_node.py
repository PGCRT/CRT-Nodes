import math
import json
import hashlib

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

class SimpleFluxShiftNode:
    """Ultra-simple preset-based FLUX shift node."""
    _cache = {}
    _max_cache_size = 10
    
    INTENSITY_PRESETS = {
        "-3": (0.1, 0.4, "Maximum shift effects (very low values)"),
        "-2": (0.2, 0.6, "Strong shift effects (low values)"),
        "-1": (0.35, 0.9, "Moderate shift effects (below default)"),
        "0": (0.5, 1.15, "Standard FLUX shift (default)"),
        "+1": (0.65, 1.4, "Moderate shift effects (above default)"),
        "+2": (0.8, 1.8, "Strong shift effects (high values)"),
        "+3": (1.0, 2.5, "Maximum shift effects (very high values)"),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        intensity_options = list(cls.INTENSITY_PRESETS.keys())
        return {
            "required": {
                "model": ("MODEL",),
                "intensity": (intensity_options, {
                    "default": "0",
                    "tooltip": "Distance from 0 determines effect strength"
                }),
                "resolution_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Multiply detected resolution for shift calculation (1.0=use as-is, 2.0=double, etc.)"
                }),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "Primary resolution source - overrides image input"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Fallback resolution source when latent not connected"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_shift_with_multiplier"
    CATEGORY = "CRT/Model"
    DESCRIPTION = "Dead simple FLUX shift - pick intensity, resolution from image!"
    
    @staticmethod
    def _create_cache_key(model, intensity, resolution_multiplier, width, height):
        """Create a cache key from the inputs."""
        model_id = id(model)
        
        cache_dict = {
            "model_id": model_id,
            "intensity": intensity,
            "resolution_multiplier": resolution_multiplier,
            "width": width,
            "height": height
        }
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    @staticmethod
    def _create_change_key(intensity, resolution_multiplier, width, height):
        """Create a deterministic key for IS_CHANGED that doesn't include model reference."""
        change_dict = {
            "intensity": intensity,
            "resolution_multiplier": resolution_multiplier,
            "width": width,
            "height": height
        }
        change_str = json.dumps(change_dict, sort_keys=True)
        return hashlib.md5(change_str.encode()).hexdigest()
    
    @classmethod
    def _manage_cache_size(cls):
        """Keep cache size under control."""
        if len(cls._cache) > cls._max_cache_size:
            keys_to_remove = list(cls._cache.keys())[:-cls._max_cache_size]
            for key in keys_to_remove:
                del cls._cache[key]
    
    @staticmethod
    def get_latent_dimensions(latent):
        """Extract width and height from ComfyUI latent tensor."""
        samples = latent["samples"]
        height, width = samples.shape[2] * 8, samples.shape[3] * 8
        return width, height
    
    @staticmethod
    def get_image_dimensions(image):
        """Extract width and height from ComfyUI image tensor."""
        if len(image.shape) == 4:
            height, width = image.shape[1], image.shape[2]
        elif len(image.shape) == 3:
            height, width = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        return width, height
    
    @staticmethod
    def calculate_shift(width, height, shift_strength, reference_resolution=1024):
        """Calculate shift based on resolution and strength. (Legacy - not used in current implementation)"""
        current_resolution = math.sqrt(width * height)
        reference_resolution = math.sqrt(reference_resolution * reference_resolution)
        resolution_factor = current_resolution / reference_resolution
        
        base_shift = 0.5 + (resolution_factor * 0.5)
        final_shift = base_shift * shift_strength
        
        return max(0.1, min(final_shift, 3.0))
    
    def apply_shift_with_multiplier(self, model, intensity, resolution_multiplier, latent=None, image=None):
        """Apply shift using intensity preset, resolution from latent/image, and resolution multiplier."""
        if intensity not in self.INTENSITY_PRESETS:
            raise ValueError(f"Unknown intensity: {intensity}")
        if latent is not None:
            width, height = self.get_latent_dimensions(latent)
            source_type = "latent"
        elif image is not None:
            width, height = self.get_image_dimensions(image)
            source_type = "image"
        else:
            raise ValueError("Either latent or image input must be provided")
        cache_key = self._create_cache_key(model, intensity, resolution_multiplier, width, height)
        
        if cache_key in self._cache:
            colored_print(f"🚀 [Cache Hit] Reusing cached model for '{intensity}' intensity", Colors.GREEN)
            colored_print(f"  📐 Resolution source: {source_type} ({width}x{height})", Colors.BLUE)
            return (self._cache[cache_key],)
        colored_print(f"🔧 [Cache Miss] Computing '{intensity}' intensity", Colors.YELLOW)
        
        base_shift, max_shift, description = self.INTENSITY_PRESETS[intensity]
        effective_width = int(width * resolution_multiplier)
        effective_height = int(height * resolution_multiplier)
        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (effective_width * effective_height / (8 * 8 * 2 * 2)) * mm + b
        import comfy.model_sampling
        
        m = model.clone()
        
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST
        
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        self._cache[cache_key] = m
        self._manage_cache_size()
        
        colored_print(f"  📊 Preset values: base_shift={base_shift}, max_shift={max_shift}", Colors.CYAN)
        colored_print(f"  📐 Resolution source: {source_type} ({width}x{height})", Colors.BLUE)
        colored_print(f"  📈 Effective resolution: {effective_width}x{effective_height} (x{resolution_multiplier})", Colors.BLUE)
        colored_print(f"  ✨ Final calculated shift: {shift:.3f}", Colors.GREEN)
        
        return (m,)
    
    @classmethod
    def IS_CHANGED(cls, intensity, resolution_multiplier, latent=None, image=None, **kwargs):
        """Return a stable hash that only changes when meaningful inputs change."""
        try:
            if latent is not None:
                width, height = cls.get_latent_dimensions(cls, latent)
            elif image is not None:
                width, height = cls.get_image_dimensions(cls, image)
            else:
                return "no_dimension_source"
            change_key = cls._create_change_key(cls, intensity, resolution_multiplier, width, height)
            return change_key
            
        except Exception as e:
            colored_print(f"⚠️ IS_CHANGED error: {e}, falling back to constant", Colors.YELLOW)
            return "fallback_constant"
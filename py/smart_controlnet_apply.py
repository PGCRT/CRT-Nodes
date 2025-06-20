import torch
import comfy.utils
import comfy.model_management as mm
import folder_paths
from nodes import ControlNetApplyAdvanced
import time
import hashlib
import weakref
from pathlib import Path

# Comprehensive list of all known preprocessors
ALL_PREPROCESSORS = [
    "none",
    "CannyEdgePreprocessor",
    "DepthAnythingV2Preprocessor",
    "DepthAnythingPreprocessor", 
    "MiDaS-DepthMapPreprocessor",
    "MiDaS-NormalMapPreprocessor",
    "LeReS-DepthMapPreprocessor",
    "Zoe-DepthMapPreprocessor",
    "BAE-NormalMapPreprocessor",
    "DSINE-NormalMapPreprocessor",
    "LineArtPreprocessor",
    "LineartStandardPreprocessor",
    "Manga2Anime_LineArt_Preprocessor",
    "M-LSDPreprocessor",
    "OpenposePreprocessor",
    "DWPreprocessor",
    "AnimalPosePreprocessor",
    "FacialPartColoringFromPoseKps",
    "UpperBodyTrackingFromPoseKps",
    "RenderPeopleKps",
    "RenderAnimalKps",
    "MLSDPreprocessor",
    "HEDPreprocessor",
    "FakeScribblePreprocessor",
    "ScribblePreprocessor",
    "PiDiNetPreprocessor",
    "NormalBAEPreprocessor",
    "SAMPreprocessor",
    "ColorPreprocessor",
    "MediaPipe-FaceMeshPreprocessor",
    "MediaPipe-HandPosePreprocessor",
    "DensePosePreprocessor",
    "OpenPosePreprocessor",
    "TilePreprocessor",
    "SegmentAnythingPreprocessor",
    "SemSegPreprocessor",
    "UniFormer-SemSegPreprocessor",
    "OneFormer-COCO-SemSegPreprocessor",
    "OneFormer-ADE20K-SemSegPreprocessor",
    "BinaryPreprocessor",
    "InpaintPreprocessor",
    "ShufflePreprocessor",
    "ContentShufflePreprocessor",
    "ColorizePreprocessor",
    "RecolorPreprocessor",
    "ImageLuminanceDetector",
    "ImageIntensityDetector",
    "TEED_Preprocessor",
    "Metric3D-DepthMapPreprocessor",
    "Metric3D-NormalMapPreprocessor",
    "DepthAnything-DepthMapPreprocessor",
    "DepthAnythingV2-DepthMapPreprocessor"
]

# Try to import ControlNet Aux
try:
    from comfyui_controlnet_aux import AUX_NODE_MAPPINGS, PREPROCESSOR_OPTIONS
    CONTROLNET_AUX_AVAILABLE = True
    # Use the actual available preprocessors from the aux package
    AVAILABLE_PREPROCESSORS = PREPROCESSOR_OPTIONS
    print(f"✅ ControlNet Aux loaded with {len(PREPROCESSOR_OPTIONS)} preprocessors")
except ImportError as e:
    print(f"⚠️ ControlNet Aux not available: {e}")
    CONTROLNET_AUX_AVAILABLE = False
    AUX_NODE_MAPPINGS = {}
    # Use comprehensive list so all preprocessors show up in the UI
    AVAILABLE_PREPROCESSORS = ALL_PREPROCESSORS
    print(f"🔄 Using comprehensive preprocessor list: {len(ALL_PREPROCESSORS)} options")

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored_print(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

class PreprocessorCache:
    def __init__(self, max_size=10, max_memory_mb=512):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size  # Reduced from 50 to 10
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self._strong_refs = {}
    
    def _generate_key(self, image, preprocessor, resolution, **kwargs):
        """Generate cache key from image and parameters."""
        if isinstance(image, torch.Tensor):
            image_hash = hashlib.md5(image.cpu().numpy().tobytes()).hexdigest()[:16]
        else:
            image_hash = str(hash(str(image)))[:16]
        param_str = f"{preprocessor}_{resolution}_{str(sorted(kwargs.items()))}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        
        return f"{image_hash}_{param_hash}"
    
    def get(self, image, preprocessor, resolution, **kwargs):
        """Get cached result if available."""
        if preprocessor == "none":
            return None
            
        key = self._generate_key(image, preprocessor, resolution, **kwargs)
        
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            weak_ref = self.cache[key]
            result = weak_ref()
            if result is not None:
                colored_print(f"   💾 Cache HIT for {preprocessor} (key: {key[:8]}...)", Colors.GREEN)
                return result
            else:
                if key in self._strong_refs:
                    result = self._strong_refs[key]
                    colored_print(f"   💾 Cache HIT (strong ref) for {preprocessor} (key: {key[:8]}...)", Colors.GREEN)
                    return result
                else:
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
        
        colored_print(f"   💾 Cache MISS for {preprocessor} (key: {key[:8]}...)", Colors.YELLOW)
        return None
    
    def put(self, image, preprocessor, resolution, result, **kwargs):
        """Store result in cache."""
        if preprocessor == "none":
            return
            
        key = self._generate_key(image, preprocessor, resolution, **kwargs)
        while len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            if oldest_key in self._strong_refs:
                del self._strong_refs[oldest_key]
        
        self.cache[key] = weakref.ref(result)
        self._strong_refs[key] = result.clone() if hasattr(result, 'clone') else result
        self.access_order.append(key)
        
        colored_print(f"   💾 Cached result for {preprocessor} (cache size: {len(self.cache)})", Colors.CYAN)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.access_order.clear()
        self._strong_refs.clear()
        colored_print("   💾 Cache cleared", Colors.BLUE)

_preprocessor_cache = PreprocessorCache()

class SmartControlNetApply:
    """
    Smart ControlNet node that combines preprocessing and application with intelligent bypassing.
    Skips preprocessing and ControlNet application when strength is 0 to avoid unnecessary computation.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preprocessor": (AVAILABLE_PREPROCESSORS, {"default": "none"}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "union_type": (["auto"] + list(getattr(__import__('comfy.cldm.control_types', fromlist=['UNION_CONTROLNET_TYPES']), 'UNION_CONTROLNET_TYPES', {}).keys()), {"default": "auto"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "processed_image")
    FUNCTION = "apply_smart_controlnet"
    CATEGORY = "🎨 CRT Nodes/ControlNet"

    def __init__(self):
        self.controlnet_applier = ControlNetApplyAdvanced()
    
    def _set_union_controlnet_type(self, control_net, union_type):
        """Set union ControlNet type if specified."""
        if union_type == "auto":
            return control_net
        
        try:
            from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
            control_net = control_net.copy()
            type_number = UNION_CONTROLNET_TYPES.get(union_type, -1)
            if type_number >= 0:
                control_net.set_extra_arg("control_type", [type_number])
            else:
                control_net.set_extra_arg("control_type", [])
        except ImportError:
            colored_print("   ⚠️ Union ControlNet types not available", Colors.YELLOW)
        
        return control_net
    
    def _run_preprocessor(self, image, preprocessor, resolution):
        """Run the specified preprocessor on the image."""
        if preprocessor == "none":
            return image
        
        if not CONTROLNET_AUX_AVAILABLE:
            colored_print(f"   ⚠️ ControlNet Aux not available, cannot run '{preprocessor}' - returning original image", Colors.YELLOW)
            return image
        
        if preprocessor not in AUX_NODE_MAPPINGS:
            colored_print(f"   ❌ Preprocessor '{preprocessor}' not found in AUX_NODE_MAPPINGS", Colors.RED)
            available_list = list(AUX_NODE_MAPPINGS.keys())[:10]  # Show first 10
            colored_print(f"   Available preprocessors: {available_list}{'...' if len(AUX_NODE_MAPPINGS) > 10 else ''}", Colors.BLUE)
            return image
        
        start_time = time.time()
        
        try:
            aux_class = AUX_NODE_MAPPINGS[preprocessor]
            input_types = aux_class.INPUT_TYPES()
            input_types = {
                **input_types["required"],
                **(input_types["optional"] if "optional" in input_types else {})
            }
            params = {"image": image, "resolution": resolution}
            preprocessor_params = {}
            
            for name, input_type in input_types.items():
                if name in params:
                    continue
                
                default_value = None
                if len(input_type) == 2 and isinstance(input_type[1], dict) and "default" in input_type[1]:
                    default_value = input_type[1]["default"]
                elif isinstance(input_type[0], list) and len(input_type[0]) > 0:
                    default_value = input_type[0][0]
                else:
                    default_values = {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}
                    if input_type[0] in default_values:
                        default_value = default_values[input_type[0]]
                
                if default_value is not None:
                    params[name] = default_value
                    if name not in ["image", "resolution"]:
                        preprocessor_params[name] = default_value
            
            # Check cache first
            cached_result = _preprocessor_cache.get(image, preprocessor, resolution, **preprocessor_params)
            if cached_result is not None:
                return cached_result
            
            # Run preprocessor
            instance = aux_class()
            result = getattr(instance, aux_class.FUNCTION)(**params)
            
            if isinstance(result, tuple) and len(result) > 0:
                processed_image = result[0]
            else:
                processed_image = result
            
            # Cache result
            _preprocessor_cache.put(image, preprocessor, resolution, processed_image, **preprocessor_params)
            
            processing_time = time.time() - start_time
            colored_print(f"   🖼️ Preprocessed with {preprocessor} in {processing_time:.2f}s", Colors.GREEN)
            
            return processed_image
            
        except Exception as e:
            colored_print(f"   ❌ Preprocessing failed for '{preprocessor}': {str(e)}", Colors.RED)
            colored_print(f"   🔄 Returning original image instead", Colors.YELLOW)
            return image
    
    def _handle_inpainting_controlnet(self, control_net, image, mask, vae):
        """Handle inpainting ControlNet if mask is provided."""
        if mask is None or not hasattr(control_net, 'concat_mask') or not control_net.concat_mask:
            return image, []
            
        mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
        masked_image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
        
        colored_print("   🎭 Applied inpainting mask", Colors.CYAN)
        return masked_image, [mask]
    
    def apply_smart_controlnet(self, positive, negative, image, vae, control_net, strength, end_percent, 
                              preprocessor, resolution, union_type, mask=None):
        """
        Smart ControlNet application with preprocessing and intelligent bypassing.
        """
        colored_print(f"\n🎨 Smart ControlNet Processing", Colors.HEADER)
        colored_print(f"   📊 Strength: {strength:.3f} | End: {end_percent:.3f} | Preprocessor: {preprocessor}", Colors.BLUE)
        
        # Check if preprocessor is available and warn if not
        if not CONTROLNET_AUX_AVAILABLE and preprocessor != "none":
            colored_print(f"   ⚠️ ControlNet Aux not installed - '{preprocessor}' will be skipped", Colors.YELLOW)
        elif CONTROLNET_AUX_AVAILABLE and preprocessor not in AUX_NODE_MAPPINGS and preprocessor != "none":
            colored_print(f"   ⚠️ Preprocessor '{preprocessor}' not available in current installation", Colors.YELLOW)
        
        start_time = time.time()
        
        # Early exit if strength is 0
        if strength == 0.0:
            colored_print(f"   ⚡ BYPASSED - Strength is 0, skipping all processing", Colors.YELLOW)
            return (positive, negative, image)
        
        # Set union type
        control_net = self._set_union_controlnet_type(control_net, union_type)
        if union_type != "auto":
            colored_print(f"   🔗 Union type: {union_type}", Colors.CYAN)
        
        # Run preprocessing
        processed_image = self._run_preprocessor(image, preprocessor, resolution)
        
        # Handle inpainting
        extra_concat = []
        if mask is not None:
            processed_image, extra_concat = self._handle_inpainting_controlnet(
                control_net, processed_image, mask, vae
            )
        
        # Apply ControlNet
        colored_print(f"   🎮 Applying ControlNet...", Colors.GREEN)
        try:
            if extra_concat:
                result = self.controlnet_applier.apply_controlnet(
                    positive, negative, control_net, processed_image, 
                    strength, 0.0, end_percent, vae=vae, extra_concat=extra_concat
                )
            else:
                result = self.controlnet_applier.apply_controlnet(
                    positive, negative, control_net, processed_image, 
                    strength, 0.0, end_percent, vae=vae
                )
            
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart ControlNet completed in {total_time:.2f}s", Colors.GREEN)
            
            return (result[0], result[1], processed_image)
            
        except Exception as e:
            colored_print(f"   ❌ ControlNet application failed: {str(e)}", Colors.RED)
            return (positive, negative, processed_image)

class ClearPreprocessorCache:
    """Utility node to clear the preprocessor cache."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clear_cache": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear_cache"
    CATEGORY = "CRT"
    
    def clear_cache(self, clear_cache):
        if clear_cache:
            _preprocessor_cache.clear()
            return ("Cache cleared successfully",)
        return ("Cache not cleared",)

# Register the node mappings
NODE_CLASS_MAPPINGS = {
    "SmartControlNetApply": SmartControlNetApply,
    "ClearPreprocessorCache": ClearPreprocessorCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartControlNetApply": "🎨 Smart ControlNet Apply",
    "ClearPreprocessorCache": "🗑️ Clear Preprocessor Cache",
}
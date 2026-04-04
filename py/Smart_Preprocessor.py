import torch
import time
import hashlib
import weakref

# Try to import ControlNet Aux - check multiple possible import paths
CONTROLNET_AUX_AVAILABLE = False
AUX_NODE_MAPPINGS = {}
AVAILABLE_PREPROCESSORS = ["none", "canny"]

# Try different import paths for ControlNet Aux
import_attempts = [
    ("comfyui_controlnet_aux", "AUX_NODE_MAPPINGS", "PREPROCESSOR_OPTIONS"),
    ("custom_nodes.comfyui_controlnet_aux", "AUX_NODE_MAPPINGS", "PREPROCESSOR_OPTIONS"),
    ("comfyui_controlnet_aux.__init__", "AUX_NODE_MAPPINGS", "PREPROCESSOR_OPTIONS"),
]

for module_path, mappings_attr, options_attr in import_attempts:
    try:
        module = __import__(module_path, fromlist=[mappings_attr, options_attr])
        AUX_NODE_MAPPINGS = getattr(module, mappings_attr, {})
        PREPROCESSOR_OPTIONS = getattr(module, options_attr, [])

        if AUX_NODE_MAPPINGS and PREPROCESSOR_OPTIONS:
            CONTROLNET_AUX_AVAILABLE = True
            AVAILABLE_PREPROCESSORS = PREPROCESSOR_OPTIONS
            print(f"✅ ControlNet Aux loaded from '{module_path}' with {len(PREPROCESSOR_OPTIONS)} preprocessors")
            break
    except (ImportError, AttributeError) as e:
        continue

# If all import attempts failed, try to detect any controlnet aux installation
if not CONTROLNET_AUX_AVAILABLE:
    try:
        import sys
        import os

        # Look for any controlnet aux in sys.modules
        aux_modules = [name for name in sys.modules.keys() if 'controlnet' in name.lower() and 'aux' in name.lower()]

        if aux_modules:
            print(f"🔍 Found ControlNet Aux modules: {aux_modules}")
            print("   But couldn't import AUX_NODE_MAPPINGS - checking alternative paths...")

            # Try to find the actual mappings in any of these modules
            for module_name in aux_modules:
                try:
                    module = sys.modules[module_name]
                    if hasattr(module, 'AUX_NODE_MAPPINGS'):
                        AUX_NODE_MAPPINGS = getattr(module, 'AUX_NODE_MAPPINGS', {})
                        PREPROCESSOR_OPTIONS = getattr(module, 'PREPROCESSOR_OPTIONS', list(AUX_NODE_MAPPINGS.keys()))
                        CONTROLNET_AUX_AVAILABLE = True
                        AVAILABLE_PREPROCESSORS = PREPROCESSOR_OPTIONS
                        print(
                            f"✅ ControlNet Aux loaded from existing module '{module_name}' with {len(PREPROCESSOR_OPTIONS)} preprocessors"
                        )
                        break
                except Exception:
                    continue

    except Exception as e:
        pass

if not CONTROLNET_AUX_AVAILABLE:
    print("\n" + "=" * 80)
    print("🚨 IMPORTANT: **comfyui_controlnet_aux** is not installed or not detectable!")
    print("   Install it via ComfyUI Manager for full preprocessor support.")
    print("   Fallback: Only 'none' and 'canny' preprocessors available.")
    print("   GitHub: https://github.com/Fannovel16/comfyui_controlnet_aux")
    print("=" * 80 + "\n")


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
    def __init__(self, max_size=10):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
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


class SmartPreprocessor:
    """
    Smart preprocessor that can bypass processing when ControlNet strength is 0.
    This prevents unnecessary preprocessing when the ControlNet won't be applied.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (AVAILABLE_PREPROCESSORS, {"default": "none"}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "enable_bypass": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "smart_preprocess"
    CATEGORY = "CRT/Image"

    def _run_canny_fallback(self, image, resolution):
        """Fallback Canny edge detection when ControlNet Aux is not available."""
        import cv2
        import numpy as np

        try:
            # Convert torch tensor to numpy
            if isinstance(image, torch.Tensor):
                # Convert from (batch, height, width, channels) to numpy
                img_np = image.cpu().numpy()
                if img_np.ndim == 4:
                    img_np = img_np[0]  # Take first image from batch

                # Convert to uint8 format for OpenCV
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                if img_np.shape[-1] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_np = image

            # Resize if needed
            if resolution != img_np.shape[1] or resolution != img_np.shape[0]:
                img_np = cv2.resize(img_np, (resolution, resolution))

            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Convert back to RGB
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Convert back to torch tensor
            edges_tensor = torch.from_numpy(edges_rgb.astype(np.float32) / 255.0)
            edges_tensor = edges_tensor.unsqueeze(0)  # Add batch dimension

            colored_print("   🔧 Applied fallback Canny edge detection", Colors.CYAN)
            return edges_tensor

        except Exception as e:
            colored_print(f"   ❌ Fallback Canny failed: {str(e)}", Colors.RED)
            return image

    def _run_preprocessor(self, image, preprocessor, resolution):
        """Run the specified preprocessor on the image."""
        if preprocessor == "none":
            return image

        # Handle fallback Canny when ControlNet Aux is not available
        if not CONTROLNET_AUX_AVAILABLE:
            if preprocessor == "canny":
                return self._run_canny_fallback(image, resolution)
            else:
                colored_print(
                    f"   ⚠️ ControlNet Aux not available, cannot run '{preprocessor}' - returning original image",
                    Colors.YELLOW,
                )
                return image

        if preprocessor not in AUX_NODE_MAPPINGS:
            colored_print(f"   ❌ Preprocessor '{preprocessor}' not found in AUX_NODE_MAPPINGS", Colors.RED)
            available_list = list(AUX_NODE_MAPPINGS.keys())[:10]  # Show first 10
            colored_print(
                f"   Available preprocessors: {available_list}{'...' if len(AUX_NODE_MAPPINGS) > 10 else ''}",
                Colors.BLUE,
            )
            return image

        start_time = time.time()

        try:
            aux_class = AUX_NODE_MAPPINGS[preprocessor]
            input_types = aux_class.INPUT_TYPES()
            input_types = {**input_types["required"], **(input_types["optional"] if "optional" in input_types else {})}
            params = {"image": image, "resolution": resolution}
            preprocessor_params = {}

            # Set default parameters for the preprocessor
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
            colored_print("   🔄 Returning original image instead", Colors.YELLOW)
            return image

    def smart_preprocess(self, image, preprocessor, resolution, controlnet_strength, enable_bypass):
        """
        Smart preprocessing with strength-based bypassing.
        """
        colored_print("\n🔧 Smart Preprocessor", Colors.HEADER)
        colored_print(
            f"   📊 Preprocessor: {preprocessor} | Resolution: {resolution} | CN Strength: {controlnet_strength:.3f}",
            Colors.BLUE,
        )

        # Check if we should bypass preprocessing
        if enable_bypass and controlnet_strength == 0.0:
            colored_print("   ⚡ BYPASSED - ControlNet strength is 0, skipping preprocessing", Colors.YELLOW)
            return (image,)

        # Check if preprocessor is available and warn if not
        if not CONTROLNET_AUX_AVAILABLE and preprocessor not in ["none", "canny"]:
            colored_print(
                f"   ⚠️ ControlNet Aux not installed - '{preprocessor}' not available, returning original image",
                Colors.YELLOW,
            )
            return (image,)
        elif (
            CONTROLNET_AUX_AVAILABLE and preprocessor not in AUX_NODE_MAPPINGS and preprocessor not in ["none", "canny"]
        ):
            colored_print(f"   ⚠️ Preprocessor '{preprocessor}' not available in current installation", Colors.YELLOW)
            return (image,)

        # Run preprocessing
        start_time = time.time()
        processed_image = self._run_preprocessor(image, preprocessor, resolution)

        total_time = time.time() - start_time
        colored_print(f"   ✅ Smart preprocessing completed in {total_time:.2f}s", Colors.GREEN)

        return (processed_image,)


# Register the node mappings
NODE_CLASS_MAPPINGS = {
    "SmartPreprocessor": SmartPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartPreprocessor": "🔧 Smart Preprocessor",
}

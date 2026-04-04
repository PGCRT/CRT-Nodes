import torch
import time
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
    UNDERLINE = '\033[4m'


def colored_print(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")


class StyleModelDualCache:
    def __init__(self):
        self.cache = None
        self.cache_key = None

    def _generate_key(self, style_model, image_1, image_2, strength_1, strength_2, strength_type, crop):
        """Generate cache key from style model, images, and parameters."""
        # Use a model-specific identifier if available, otherwise fall back to id()
        model_id = getattr(style_model, 'model_id', id(style_model))
        # Use image data for cache key to ensure uniqueness
        image_1_hash = hashlib.md5(image_1.cpu().numpy().tobytes()).hexdigest()[:16]
        image_2_hash = hashlib.md5(image_2.cpu().numpy().tobytes()).hexdigest()[:16] if image_2 is not None else "none"
        param_str = f"{model_id}_{image_1_hash}_{image_2_hash}_{strength_1}_{strength_2}_{strength_type}_{crop}"
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def get(self, style_model, image_1, image_2, strength_1, strength_2, strength_type, crop):
        """Get cached result if available."""
        key = self._generate_key(style_model, image_1, image_2, strength_1, strength_2, strength_type, crop)

        if self.cache_key == key and self.cache is not None:
            colored_print(f"   💾 Dual Style Cache HIT (key: {key[:8]}...)", Colors.GREEN)
            return self.cache
        return None

    def put(self, style_model, image_1, image_2, strength_1, strength_2, strength_type, crop, result):
        """Store result in cache, replacing any existing entry."""
        key = self._generate_key(style_model, image_1, image_2, strength_1, strength_2, strength_type, crop)
        self.cache = result  # Store list directly, no weakref
        self.cache_key = key
        colored_print(f"   💾 Cached dual style result (key: {key[:8]}...)", Colors.CYAN)

    def clear(self):
        """Clear cached result."""
        if self.cache is not None:
            cleared = True
            self.cache = None
            self.cache_key = None
            colored_print("   💾 Dual Style cache cleared", Colors.BLUE)
            return cleared
        return False


_style_dual_cache = StyleModelDualCache()


class SmartStyleModelApplyDual:
    """
    Dual Smart Style Model Apply node with integrated CLIP Vision encoding.
    Uses shared settings for both styles. Second style is optional and automatically
    activated when image_2 is connected. Caches only the most recent result.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image_1": ("IMAGE",),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_type": (["multiply", "attn_bias"],),
                "crop": (["center", "none"],),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_dual_smart_stylemodel"
    CATEGORY = "CRT/Conditioning"

    def _encode_clip_vision(self, clip_vision, image, crop):
        """Encode image using CLIP Vision model."""
        try:
            crop_image = crop == "center"
            return clip_vision.encode_image(image, crop=crop_image)
        except Exception as e:
            colored_print(f"   ❌ CLIP Vision encoding failed: {str(e)}", Colors.RED)
            raise

    def _apply_style_conditioning(self, conditioning, cond, strength, strength_type, style_num):
        """Apply style conditioning to the input conditioning with proper attention mask handling."""
        try:
            if strength_type == "multiply":
                cond *= strength

            n = cond.shape[1]
            c_out = []

            for t in conditioning:
                txt, keys = t
                keys = keys.copy()
                if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                    attn_bias = torch.log(torch.tensor([strength if strength_type == "attn_bias" else 1.0]))
                    mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                    n_ref = mask_ref_size[0] * mask_ref_size[1]
                    n_txt = txt.shape[1]
                    mask = keys.get("attention_mask", None)
                    if mask is None:
                        mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                    if mask.dtype == torch.bool:
                        mask = torch.log(mask.to(dtype=torch.float16))
                    new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                    new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                    new_mask[:, :n_txt, n_txt + n :] = mask[:, :n_txt, n_txt:]
                    new_mask[:, n_txt + n :, :n_txt] = mask[:, n_txt:, :n_txt]
                    new_mask[:, n_txt + n :, n_txt + n :] = mask[:, n_txt:, n_txt:]
                    new_mask[:, :n_txt, n_txt : n_txt + n] = attn_bias
                    new_mask[:, n_txt + n :, n_txt : n_txt + n] = attn_bias

                    keys["attention_mask"] = new_mask.to(txt.device)
                    keys["attention_mask_img_shape"] = mask_ref_size

                c_out.append([torch.cat((txt, cond), dim=1), keys])

            colored_print(f"   ✨ Style {style_num} conditioning applied", Colors.CYAN)
            return c_out
        except Exception as e:
            colored_print(f"   ❌ Style {style_num} conditioning failed: {str(e)}", Colors.RED)
            raise

    def _process_single_style(self, conditioning, style_model, clip_vision_output, strength, strength_type, style_num):
        """Process a single style model application."""
        if strength == 0.0:
            colored_print(f"   ⚡ Style {style_num} BYPASSED - Strength is 0", Colors.YELLOW)
            return conditioning

        colored_print(f"   🎭 Generating Style {style_num} conditioning...", Colors.CYAN)
        try:
            model_start = time.time()
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            model_time = time.time() - model_start
            colored_print(f"   🔄 Style {style_num} model inference in {model_time:.3f}s", Colors.BLUE)
            return self._apply_style_conditioning(conditioning, cond, strength, strength_type, style_num)
        except Exception as e:
            colored_print(f"   ❌ Style {style_num} model application failed: {str(e)}", Colors.RED)
            raise

    def apply_dual_smart_stylemodel(
        self, conditioning, style_model, clip_vision, image_1, strength_1, strength_2, strength_type, crop, image_2=None
    ):
        """
        Dual Smart Style Model application with integrated CLIP Vision encoding.
        Second style is optional and automatically activated when image_2 is provided.
        Caches only the most recent result based on images, strengths, and crop.
        """
        # Input validation
        if not isinstance(image_1, torch.Tensor):
            colored_print("   ❌ Invalid input: image_1 must be a torch.Tensor", Colors.RED)
            raise ValueError("image_1 must be a torch.Tensor")
        if image_2 is not None and not isinstance(image_2, torch.Tensor):
            colored_print("   ❌ Invalid input: image_2 must be a torch.Tensor", Colors.RED)
            raise ValueError("image_2 must be a torch.Tensor")

        # Determine if dual mode is active
        dual_mode = image_2 is not None

        if dual_mode:
            colored_print("\n🎨🎨 Dual Smart Style Model Processing", Colors.HEADER)
            colored_print(
                f"   📊 Style 1: {strength_1:.3f} | Style 2: {strength_2:.3f} | Type: {strength_type} | Crop: {crop}",
                Colors.BLUE,
            )
        else:
            colored_print("\n🎨 Single Smart Style Model Processing (Image 2 not connected)", Colors.HEADER)
            colored_print(f"   📊 Style 1: {strength_1:.3f} | Type: {strength_type} | Crop: {crop}", Colors.BLUE)

        start_time = time.time()

        # Check bypass conditions first
        if not dual_mode and strength_1 == 0.0:
            colored_print("   ⚡ SINGLE BYPASS - Strength 1 is 0, returning original conditioning", Colors.YELLOW)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Single Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            return (conditioning,)

        if dual_mode and strength_1 == 0.0 and strength_2 == 0.0:
            colored_print("   ⚡ DUAL BYPASS - Both strengths are 0, returning original conditioning", Colors.YELLOW)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Dual Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            return (conditioning,)

        # Check cache after bypass
        cached_cond = _style_dual_cache.get(style_model, image_1, image_2, strength_1, strength_2, strength_type, crop)
        if cached_cond is not None:
            colored_print("   💾 Using cached conditioning", Colors.GREEN)
            total_time = time.time() - start_time
            if dual_mode:
                colored_print(f"   🎉 Dual Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            else:
                colored_print(f"   🎉 Single Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            return (cached_cond,)

        # Always encode Image 1
        colored_print("   👁️ Encoding Image 1 with CLIP Vision...", Colors.CYAN)
        vision_start_1 = time.time()
        clip_vision_output_1 = self._encode_clip_vision(clip_vision, image_1, crop)
        vision_time_1 = time.time() - vision_start_1
        colored_print(f"   👁️ Image 1 CLIP Vision encoding completed in {vision_time_1:.3f}s", Colors.BLUE)

        # Encode Image 2 only if provided
        clip_vision_output_2 = None
        vision_time_2 = 0
        if dual_mode:
            colored_print("   👁️ Encoding Image 2 with CLIP Vision...", Colors.CYAN)
            vision_start_2 = time.time()
            clip_vision_output_2 = self._encode_clip_vision(clip_vision, image_2, crop)
            vision_time_2 = time.time() - vision_start_2
            colored_print(f"   👁️ Image 2 CLIP Vision encoding completed in {vision_time_2:.3f}s", Colors.BLUE)

        # Apply first style model
        colored_print("   🎨 Processing Style Model with Image 1...", Colors.HEADER)
        style_1_start = time.time()
        intermediate_conditioning = self._process_single_style(
            conditioning, style_model, clip_vision_output_1, strength_1, strength_type, 1
        )
        style_1_time = time.time() - style_1_start
        colored_print(f"   ✅ Style 1 completed in {style_1_time:.3f}s", Colors.GREEN)

        # Apply second style model only if in dual mode and Image 2 is provided
        style_2_time = 0
        final_conditioning = intermediate_conditioning

        if dual_mode:
            colored_print("   🎨 Processing Style Model with Image 2 (chained from Style 1)...", Colors.HEADER)
            style_2_start = time.time()
            final_conditioning = self._process_single_style(
                intermediate_conditioning, style_model, clip_vision_output_2, strength_2, strength_type, 2
            )
            style_2_time = time.time() - style_2_start
            colored_print(f"   ✅ Style 2 completed in {style_2_time:.3f}s", Colors.GREEN)

        # Cache the final result
        _style_dual_cache.put(
            style_model, image_1, image_2, strength_1, strength_2, strength_type, crop, final_conditioning
        )

        total_time = time.time() - start_time

        if dual_mode:
            colored_print(f"   🎉 Dual Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            colored_print(
                f"   📈 Breakdown: Vision1={vision_time_1:.3f}s, Vision2={vision_time_2:.3f}s, Style1={style_1_time:.3f}s, Style2={style_2_time:.3f}s",
                Colors.BLUE,
            )
        else:
            colored_print(f"   🎉 Single Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            colored_print(f"   📈 Breakdown: Vision1={vision_time_1:.3f}s, Style1={style_1_time:.3f}s", Colors.BLUE)

        return (final_conditioning,)


class ClearStyleModelDualCache:
    """Utility node to clear the dual style model cache."""

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
    CATEGORY = "CRT/Conditioning"

    def clear_cache(self, clear_cache):
        if clear_cache:
            cleared = _style_dual_cache.clear()
            if cleared:
                return ("Dual Style cache cleared successfully",)
            return ("Dual Style cache was already empty",)
        return ("Dual Style cache not cleared",)


# Export classes for external import
__all__ = ["SmartStyleModelApplyDual", "ClearStyleModelDualCache"]

NODE_CLASS_MAPPINGS = {
    "SmartStyleModelApplyDual": SmartStyleModelApplyDual,
    "ClearStyleModelDualCache": ClearStyleModelDualCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartStyleModelApplyDual": "Smart Style Model Apply DUAL (CRT)",
    "ClearStyleModelDualCache": "Clear Dual Style Cache (CRT)",
}

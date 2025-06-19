import torch
import time
import hashlib
import weakref

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
    def __init__(self, max_size=5):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def _generate_key(self, style_model, strength, strength_type):
        """Generate cache key from style model and parameters only (not image-dependent)."""
        model_id = id(style_model)
        param_str = f"{model_id}_{strength}_{strength_type}"
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def get(self, style_model, strength, strength_type):
        """Get cached result if available."""
        key = self._generate_key(style_model, strength, strength_type)
        
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            weak_ref = self.cache[key]
            result = weak_ref()
            if result is not None:
                colored_print(f"   💾 Dual Style Cache HIT (key: {key[:8]}...)", Colors.GREEN)
                return result
            else:
                del self.cache[key]
                self.access_order.remove(key)
        
        return None
    
    def put(self, style_model, strength, strength_type, result):
        """Store result in cache."""
        key = self._generate_key(style_model, strength, strength_type)
        while len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                colored_print(f"   💾 Evicted old dual style cache entry", Colors.BLUE)
        self.cache[key] = weakref.ref(result)
        self.access_order.append(key)
        
        colored_print(f"   💾 Cached dual style result (cache size: {len(self.cache)})", Colors.CYAN)
    
    def clear(self):
        """Clear all cached results."""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        colored_print(f"   💾 Dual Style cache cleared ({cleared_count} entries)", Colors.BLUE)

_style_dual_cache = StyleModelDualCache()

class SmartStyleModelApplyDual:
    """
    Dual Smart Style Model Apply node with integrated CLIP Vision encoding.
    Uses shared settings for both styles. Second style is optional and automatically
    activated when image_2 is connected.
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
                "enable_caching": ("BOOLEAN", {"default": False, "tooltip": "Enable caching (use with caution - may cause OOM with many different images)"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_dual_smart_stylemodel"
    CATEGORY = "CRT"

    def _encode_clip_vision(self, clip_vision, image, crop):
        """Encode image using CLIP Vision model."""
        crop_image = True
        if crop != "center":
            crop_image = False
        return clip_vision.encode_image(image, crop=crop_image)

    def _apply_style_conditioning(self, conditioning, cond, strength, strength_type, style_num):
        """Apply style conditioning to the input conditioning with proper attention mask handling."""
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
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
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        colored_print(f"   ✨ Style {style_num} conditioning applied", Colors.CYAN)
        return c_out

    def _process_single_style(self, conditioning, style_model, clip_vision_output, strength, strength_type, style_num, enable_caching):
        """Process a single style model application."""
        if strength == 0.0:
            colored_print(f"   ⚡ Style {style_num} BYPASSED - Strength is 0", Colors.YELLOW)
            return conditioning

        cached_cond = None
        if enable_caching:
            cached_cond = _style_dual_cache.get(style_model, strength, strength_type)

        if cached_cond is not None:
            colored_print(f"   💾 Using cached conditioning for Style {style_num}", Colors.GREEN)
            return self._apply_style_conditioning(conditioning, cached_cond, strength, strength_type, style_num)

        colored_print(f"   🎭 Generating Style {style_num} conditioning...", Colors.CYAN)
        try:
            model_start = time.time()
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            model_time = time.time() - model_start
            colored_print(f"   🔄 Style {style_num} model inference in {model_time:.3f}s", Colors.BLUE)

            if enable_caching:
                _style_dual_cache.put(style_model, strength, strength_type, cond)

            return self._apply_style_conditioning(conditioning, cond, strength, strength_type, style_num)

        except Exception as e:
            colored_print(f"   ❌ Style {style_num} model application failed: {str(e)}", Colors.RED)
            return conditioning

    def apply_dual_smart_stylemodel(self, conditioning, style_model, clip_vision, image_1, strength_1, strength_2, strength_type, crop, enable_caching, image_2=None):
        """
        Dual Smart Style Model application with integrated CLIP Vision encoding.
        Second style is optional and automatically activated when image_2 is provided.
        """
        # Determine if dual mode is active
        dual_mode = image_2 is not None
        
        if dual_mode:
            colored_print(f"\n🎨🎨 Dual Smart Style Model Processing", Colors.HEADER)
            colored_print(f"   📊 Style 1: {strength_1:.3f} | Style 2: {strength_2:.3f} | Type: {strength_type} | Crop: {crop}", Colors.BLUE)
        else:
            colored_print(f"\n🎨 Single Smart Style Model Processing (Image 2 not connected)", Colors.HEADER)
            colored_print(f"   📊 Style 1: {strength_1:.3f} | Type: {strength_type} | Crop: {crop}", Colors.BLUE)
        
        colored_print(f"   📊 Caching: {enable_caching}", Colors.BLUE)
        
        start_time = time.time()
        
        # Always encode Image 1
        colored_print(f"   👁️  Encoding Image 1 with CLIP Vision...", Colors.CYAN)
        vision_start_1 = time.time()
        clip_vision_output_1 = self._encode_clip_vision(clip_vision, image_1, crop)
        vision_time_1 = time.time() - vision_start_1
        colored_print(f"   👁️  Image 1 CLIP Vision encoding completed in {vision_time_1:.3f}s", Colors.BLUE)
        
        # Encode Image 2 only if provided
        clip_vision_output_2 = None
        vision_time_2 = 0
        if dual_mode:
            colored_print(f"   👁️  Encoding Image 2 with CLIP Vision...", Colors.CYAN)
            vision_start_2 = time.time()
            clip_vision_output_2 = self._encode_clip_vision(clip_vision, image_2, crop)
            vision_time_2 = time.time() - vision_start_2
            colored_print(f"   👁️  Image 2 CLIP Vision encoding completed in {vision_time_2:.3f}s", Colors.BLUE)
        
        # Check bypass conditions
        if not dual_mode and strength_1 == 0.0:
            colored_print(f"   ⚡ SINGLE BYPASS - Strength 1 is 0, returning original conditioning", Colors.YELLOW)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Single Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            return (conditioning,)
        
        if dual_mode and strength_1 == 0.0 and strength_2 == 0.0:
            colored_print(f"   ⚡ DUAL BYPASS - Both strengths are 0, returning original conditioning", Colors.YELLOW)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Dual Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            return (conditioning,)
        
        if enable_caching:
            colored_print(f"   ⚠️  WARNING: Caching enabled - may cause OOM with many images", Colors.YELLOW)
        
        # Apply first style model
        colored_print(f"   🎨 Processing Style Model with Image 1...", Colors.HEADER)
        style_1_start = time.time()
        intermediate_conditioning = self._process_single_style(
            conditioning, style_model, clip_vision_output_1, 
            strength_1, strength_type, 1, enable_caching
        )
        style_1_time = time.time() - style_1_start
        colored_print(f"   ✅ Style 1 completed in {style_1_time:.3f}s", Colors.GREEN)
        
        # Apply second style model only if in dual mode and Image 2 is provided
        style_2_time = 0
        final_conditioning = intermediate_conditioning
        
        if dual_mode:
            colored_print(f"   🎨 Processing Style Model with Image 2 (chained from Style 1)...", Colors.HEADER)
            style_2_start = time.time()
            final_conditioning = self._process_single_style(
                intermediate_conditioning, style_model, clip_vision_output_2,
                strength_2, strength_type, 2, enable_caching
            )
            style_2_time = time.time() - style_2_start
            colored_print(f"   ✅ Style 2 completed in {style_2_time:.3f}s", Colors.GREEN)
        
        total_time = time.time() - start_time
        
        if dual_mode:
            colored_print(f"   🎉 Dual Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            colored_print(f"   📈 Breakdown: Vision1={vision_time_1:.3f}s, Vision2={vision_time_2:.3f}s, Style1={style_1_time:.3f}s, Style2={style_2_time:.3f}s", Colors.BLUE)
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
    CATEGORY = "CRT"
    
    def clear_cache(self, clear_cache):
        if clear_cache:
            _style_dual_cache.clear()
            return ("Dual Style cache cleared successfully",)
        return ("Dual Style cache not cleared",)

# Export classes for external import
__all__ = ["SmartStyleModelApplyDual", "ClearStyleModelDualCache"]

NODE_CLASS_MAPPINGS = {
    "SmartStyleModelApplyDual": SmartStyleModelApplyDual,
    "ClearStyleModelDualCache": ClearStyleModelDualCache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartStyleModelApplyDual": "Smart Style Model Apply DUAL (CRT)",
    "ClearStyleModelDualCache": "Clear Dual Style Cache (CRT)"
}
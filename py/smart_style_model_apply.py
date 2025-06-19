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

class StyleModelCache:
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
                colored_print(f"   💾 Style Cache HIT (key: {key[:8]}...)", Colors.GREEN)
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
                colored_print(f"   💾 Evicted old cache entry", Colors.BLUE)
        self.cache[key] = weakref.ref(result)
        self.access_order.append(key)
        
        colored_print(f"   💾 Cached style result (cache size: {len(self.cache)})", Colors.CYAN)
    
    def clear(self):
        """Clear all cached results."""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        colored_print(f"   💾 Style cache cleared ({cleared_count} entries)", Colors.BLUE)

_style_cache = StyleModelCache()

class SmartStyleModelApply:
    """
    Smart Style Model Apply node that avoids computation when strength is 0.
    Note: Caching disabled for image-dependent operations to prevent OOM.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_type": (["multiply", "attn_bias"],),
                "enable_caching": ("BOOLEAN", {"default": False, "tooltip": "Enable caching (use with caution - may cause OOM with many different images)"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_smart_stylemodel"
    CATEGORY = "CRT"

    def _apply_style_conditioning(self, conditioning, cond, strength, strength_type):
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

        return c_out

    def apply_smart_stylemodel(self, conditioning, style_model, clip_vision_output, strength, strength_type, enable_caching):
        """
        Smart Style Model application with intelligent bypassing and optional caching.
        """
        colored_print(f"\n🎨 Smart Style Model Processing", Colors.HEADER)
        colored_print(f"   📊 Strength: {strength:.3f} | Type: {strength_type} | Caching: {enable_caching}", Colors.BLUE)
        
        start_time = time.time()
        if strength == 0.0:
            colored_print(f"   ⚡ BYPASSED - Strength is 0, returning original conditioning", Colors.YELLOW)
            return (conditioning,)
        cached_cond = None
        if enable_caching:
            colored_print(f"   ⚠️  WARNING: Caching enabled - may cause OOM with many images", Colors.YELLOW)
            cached_cond = _style_cache.get(style_model, strength, strength_type)
        
        if cached_cond is not None:
            result = self._apply_style_conditioning(conditioning, cached_cond, strength, strength_type)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart Style Model completed (cached) in {total_time:.3f}s", Colors.GREEN)
            return (result,)
        colored_print(f"   🎭 Generating style conditioning (no cache)...", Colors.CYAN)
        try:
            model_start = time.time()
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            model_time = time.time() - model_start
            colored_print(f"   🔄 Style model inference in {model_time:.3f}s", Colors.BLUE)
            if enable_caching:
                _style_cache.put(style_model, strength, strength_type, cond)
            result = self._apply_style_conditioning(conditioning, cond, strength, strength_type)
            
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart Style Model completed in {total_time:.3f}s", Colors.GREEN)
            
            return (result,)
            
        except Exception as e:
            colored_print(f"   ❌ Style model application failed: {str(e)}", Colors.RED)
            return (conditioning,)

class SmartStyleModelApplyWithVision:
    """
    Alternative Smart Style Model Apply node that includes CLIP Vision encoding.
    Combines CLIP vision encoding and style model application in one node.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_type": (["multiply", "attn_bias"],),
                "crop": (["center", "none"],),
                "enable_caching": ("BOOLEAN", {"default": False, "tooltip": "Enable caching (use with caution - may cause OOM with many different images)"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CLIP_VISION_OUTPUT")
    RETURN_NAMES = ("conditioning", "clip_vision_output")
    FUNCTION = "apply_smart_stylemodel_with_vision"
    CATEGORY = "CRT"

    def _encode_clip_vision(self, clip_vision, image, crop):
        """Encode image using CLIP Vision model."""
        crop_image = True
        if crop != "center":
            crop_image = False
        return clip_vision.encode_image(image, crop=crop_image)

    def _apply_style_conditioning(self, conditioning, cond, strength, strength_type):
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

        return c_out

    def apply_smart_stylemodel_with_vision(self, conditioning, style_model, clip_vision, image, strength, strength_type, crop, enable_caching):
        """
        Smart Style Model application with integrated CLIP Vision encoding.
        """
        colored_print(f"\n🎨 Smart Style Model with Vision Processing", Colors.HEADER)
        colored_print(f"   📊 Strength: {strength:.3f} | Type: {strength_type} | Crop: {crop} | Caching: {enable_caching}", Colors.BLUE)
        
        start_time = time.time()
        
        # Always encode the image first (needed for output)
        vision_start = time.time()
        colored_print(f"   👁️  Encoding image with CLIP Vision...", Colors.CYAN)
        clip_vision_output = self._encode_clip_vision(clip_vision, image, crop)
        vision_time = time.time() - vision_start
        colored_print(f"   👁️  CLIP Vision encoding completed in {vision_time:.3f}s", Colors.BLUE)
        
        if strength == 0.0:
            colored_print(f"   ⚡ BYPASSED - Strength is 0, returning original conditioning", Colors.YELLOW)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart Style Model with Vision completed in {total_time:.3f}s", Colors.GREEN)
            return (conditioning, clip_vision_output)
        
        cached_cond = None
        if enable_caching:
            colored_print(f"   ⚠️  WARNING: Caching enabled - may cause OOM with many images", Colors.YELLOW)
            cached_cond = _style_cache.get(style_model, strength, strength_type)
        
        if cached_cond is not None:
            result = self._apply_style_conditioning(conditioning, cached_cond, strength, strength_type)
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart Style Model with Vision completed (cached) in {total_time:.3f}s", Colors.GREEN)
            return (result, clip_vision_output)
        
        colored_print(f"   🎭 Generating style conditioning (no cache)...", Colors.CYAN)
        try:
            model_start = time.time()
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            model_time = time.time() - model_start
            colored_print(f"   🔄 Style model inference in {model_time:.3f}s", Colors.BLUE)
            
            if enable_caching:
                _style_cache.put(style_model, strength, strength_type, cond)
            
            result = self._apply_style_conditioning(conditioning, cond, strength, strength_type)
            
            total_time = time.time() - start_time
            colored_print(f"   ✅ Smart Style Model with Vision completed in {total_time:.3f}s", Colors.GREEN)
            
            return (result, clip_vision_output)
            
        except Exception as e:
            colored_print(f"   ❌ Style model application failed: {str(e)}", Colors.RED)
            return (conditioning, clip_vision_output)

class ClearStyleModelCache:
    """Utility node to clear the style model cache."""
    
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
            _style_cache.clear()
            return ("Style cache cleared successfully",)
        return ("Style cache not cleared",)

# Export classes for external import
__all__ = ["SmartStyleModelApply", "SmartStyleModelApplyWithVision", "ClearStyleModelCache"]

NODE_CLASS_MAPPINGS = {
    "SmartStyleModelApply": SmartStyleModelApply,
    "SmartStyleModelApplyWithVision": SmartStyleModelApplyWithVision,
    "ClearStyleModelCache": ClearStyleModelCache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartStyleModelApply": "Smart Style Model Apply (CRT)",
    "SmartStyleModelApplyWithVision": "Smart Style Model Apply with Vision (CRT)",
    "ClearStyleModelCache": "Clear Style Model Cache (CRT)"
}
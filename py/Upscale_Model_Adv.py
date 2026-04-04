import torch
import torch.nn.functional as F
import comfy
import comfy.utils
import comfy.model_management as mm
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
import folder_paths
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


class CRT_UpscaleModelAdv:
    """Advanced upscale node with tiling, smart memory management, and output multiplier control."""

    _cache = {}
    _max_cache_size = 5

    precision_options = ["auto", "fp32", "fp16", "bf16"]
    tile_count_options = ["1", "4", "8", "16"]

    def __init__(self):
        self.upscale_loader = UpscaleModelLoader()
        self.image_upscaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = folder_paths.get_filename_list("upscale_models")
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model_name": (
                    upscale_models,
                    {"tooltip": "Select upscale model from models/upscale_models folder"},
                ),
                "use_fixed_resolution": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable to use fixed width/height instead of a multiplier"},
                ),
                "output_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.25,
                        "tooltip": "Final output size multiplier relative to input (1.0=same as input, 2.0=double input size)",
                    },
                ),
                "fixed_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Target width for the upscaled image",
                    },
                ),
                "fixed_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Target height for the upscaled image",
                    },
                ),
                "tile_count": (
                    cls.tile_count_options,
                    {
                        "default": "1",
                        "tooltip": "Number of tiles per side (1=no tiling, 4=4x4=16 tiles, 8=8x8=64 tiles, etc.)",
                    },
                ),
                "precision": (
                    cls.precision_options,
                    {"default": "auto", "tooltip": "Processing precision (auto=fp16 on CUDA, fp32 on CPU)"},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Number of images to process simultaneously",
                    },
                ),
                "offload_model": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Offload model to CPU after processing to save VRAM"},
                ),
                "disable_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Disable caching to always reprocess images (useful for testing or varying results)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "upscale_advanced"
    CATEGORY = "CRT/Image"
    DESCRIPTION = "Advanced upscaling with tiling, smart memory management, and precise output control"

    @staticmethod
    def _create_cache_key(
        image,
        upscale_model_name,
        use_fixed_resolution,
        output_multiplier,
        fixed_width,
        fixed_height,
        tile_count,
        precision,
        batch_size,
    ):
        """Create cache key from parameters (excluding image tensor for efficiency)."""
        img_signature = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "device": str(image.device),
            "sample_hash": hashlib.md5(image[0, :8, :8, :].cpu().numpy().tobytes()).hexdigest(),
        }

        cache_dict = {
            "image_signature": img_signature,
            "upscale_model_name": upscale_model_name,
            "use_fixed_resolution": use_fixed_resolution,
            "output_multiplier": output_multiplier,
            "fixed_width": fixed_width,
            "fixed_height": fixed_height,
            "tile_count": tile_count,
            "precision": precision,
            "batch_size": batch_size,
        }

        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    @classmethod
    def _manage_cache_size(cls):
        """Keep cache size under control."""
        if len(cls._cache) > cls._max_cache_size:
            keys_to_remove = list(cls._cache.keys())[: -cls._max_cache_size]
            for key in keys_to_remove:
                colored_print("  🗑️ Removing old cache entry", Colors.YELLOW)
                del cls._cache[key]

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _determine_dtype(self, precision, device):
        """Determine optimal dtype based on precision setting and device."""
        import torch  # Explicit import to fix scope issue

        if precision == "auto":
            if device.type == "cuda":
                return torch.float16
            elif hasattr(torch, 'bfloat16') and device.type in ["mps", "cpu"]:
                return torch.bfloat16
            else:
                return torch.float32
        elif precision == "fp16":
            return torch.float16 if device.type == "cuda" else torch.bfloat16
        elif precision == "bf16":
            return torch.bfloat16
        else:  # fp32
            return torch.float32

    def _get_model_scale_factor(self, model_name):
        """Detect model scale factor from filename."""
        model_name_lower = model_name.lower()
        if "4x" in model_name_lower or "x4" in model_name_lower:
            return 4
        elif "2x" in model_name_lower or "x2" in model_name_lower:
            return 2
        elif "8x" in model_name_lower or "x8" in model_name_lower:
            return 8
        else:
            return 4

    def _split_image_into_tiles(self, image, tile_size):
        """Split image into tiles for processing (no overlap)."""
        B, H, W, C = image.shape
        tiles = []
        positions = []

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)

                tile = image[:, y:y_end, x:x_end, :]
                tiles.append(tile)
                positions.append((y, y_end, x, x_end))

        return tiles, positions

    def _merge_tiles(self, tiles, positions, output_shape, model_scale):
        """Merge processed tiles back into full image (no overlap blending)."""
        B, H, W, C = output_shape
        result = torch.zeros((B, H, W, C), dtype=tiles[0].dtype, device=tiles[0].device)

        for tile, (y_start, y_end, x_start, x_end) in zip(tiles, positions):
            out_y_start = y_start * model_scale
            out_y_end = y_end * model_scale
            out_x_start = x_start * model_scale
            out_x_end = x_end * model_scale
            tile_h, tile_w = tile.shape[1], tile.shape[2]
            actual_h = min(tile_h, H - out_y_start)
            actual_w = min(tile_w, W - out_x_start)
            result[:, out_y_start : out_y_start + actual_h, out_x_start : out_x_start + actual_w, :] = tile[
                :, :actual_h, :actual_w, :
            ]

        return result

    def _process_with_tiling(self, model, image, tile_size_px, model_scale, dtype, device):
        """Process image using tiling for memory efficiency (no overlap)."""
        colored_print(f"🧩 Processing with {tile_size_px}px tiles (no overlap)...", Colors.CYAN)
        tiles, positions = self._split_image_into_tiles(image, tile_size_px)
        processed_tiles = []

        colored_print(f"  📋 Split into {len(tiles)} tiles", Colors.BLUE)
        for i, tile in enumerate(tiles):
            colored_print(f"  🔄 Processing tile {i+1}/{len(tiles)} ({tile.shape[1]}x{tile.shape[2]})", Colors.YELLOW)

            tile = tile.to(dtype)
            with torch.no_grad():
                if dtype in [torch.float16, torch.bfloat16]:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        processed_tile = self.image_upscaler.upscale(model, tile)[0]
                else:
                    processed_tile = self.image_upscaler.upscale(model, tile)[0]

            processed_tiles.append(processed_tile)
        B, H, W, C = image.shape
        output_shape = (B, H * model_scale, W * model_scale, C)
        colored_print(f"  🔗 Merging tiles into {output_shape[1]}x{output_shape[2]} image...", Colors.CYAN)
        result = self._merge_tiles(processed_tiles, positions, output_shape, model_scale)

        return result

    def upscale_advanced(
        self,
        image,
        upscale_model_name,
        use_fixed_resolution,
        output_multiplier,
        fixed_width,
        fixed_height,
        tile_count,
        precision,
        batch_size,
        offload_model,
        disable_cache,
    ):
        """Advanced upscaling with all features."""
        cache_key = self._create_cache_key(
            image,
            upscale_model_name,
            use_fixed_resolution,
            output_multiplier,
            fixed_width,
            fixed_height,
            tile_count,
            precision,
            batch_size,
        )

        # Check cache only if caching is enabled
        if not disable_cache and cache_key in self._cache:
            colored_print("🚀 [Cache Hit] Reusing cached upscale result", Colors.GREEN)
            cached_result = self._cache[cache_key]
            cached_width = cached_result.shape[2]
            cached_height = cached_result.shape[1]
            colored_print(f"  📄 Model: {upscale_model_name}", Colors.BLUE)
            colored_print(f"  📐 Cached resolution: {cached_width}x{cached_height}", Colors.BLUE)
            return (cached_result, cached_width, cached_height)

        if disable_cache:
            colored_print("🔧 [Cache Disabled] Processing upscale without caching", Colors.YELLOW)
        else:
            colored_print("🔧 [Cache Miss] Processing upscale", Colors.YELLOW)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device() if offload_model else device

        colored_print("🚀 [CRT Upscale Advanced] Starting upscale process", Colors.HEADER)
        colored_print(f"  📄 Model: {upscale_model_name}", Colors.BLUE)
        colored_print(f"  📐 Input resolution: {image.shape[2]}x{image.shape[1]}", Colors.BLUE)
        try:
            upscale_model = self.upscale_loader.load_model(upscale_model_name)[0]
            colored_print("  ✅ Model loaded successfully", Colors.GREEN)
        except Exception as e:
            colored_print(f"  ❌ Failed to load model: {e}", Colors.RED)
            raise

        model_scale = self._get_model_scale_factor(upscale_model_name)
        colored_print(f"  🔍 Detected model scale: {model_scale}x", Colors.CYAN)

        input_h, input_w = image.shape[1], image.shape[2]

        if use_fixed_resolution:
            target_h, target_w = fixed_height, fixed_width
            colored_print(f"  🎯 Target resolution (fixed): {target_w}x{target_h}", Colors.CYAN)
        else:
            target_h = int(input_h * output_multiplier)
            target_w = int(input_w * output_multiplier)
            colored_print(
                f"  🎯 Target resolution (multiplier {output_multiplier}x): {target_w}x{target_h}", Colors.CYAN
            )

        tile_count_int = int(tile_count)
        tile_size_px = max(input_h, input_w) // tile_count_int  # Calculate tile size based on image and count

        use_tiling = tile_count_int > 1

        if use_tiling:
            colored_print(
                f"  🧩 Using tiled processing ({tile_count}x{tile_count} = {tile_count_int*tile_count_int} tiles)",
                Colors.YELLOW,
            )
            colored_print(f"    📋 Each tile: ~{tile_size_px}px (no overlap)", Colors.BLUE)
        else:
            colored_print("  🖼️ Using single-pass processing (no tiling)", Colors.GREEN)

        colored_print(f"  📦 Batch processing: {batch_size} image(s) per batch", Colors.BLUE)
        dtype = self._determine_dtype(precision, device)
        colored_print(f"  🎛️ Processing precision: {str(dtype).split('.')[-1]}", Colors.BLUE)
        try:
            upscale_model.to(device)
            upscale_model = upscale_model.to(dtype)
            image_batches = torch.split(image, batch_size)
            processed_batches = []

            colored_print(f"  📦 Processing {len(image_batches)} batch(es) of size {batch_size}", Colors.BLUE)

            for batch_idx, batch in enumerate(image_batches):
                colored_print(f"  🔄 Processing batch {batch_idx + 1}/{len(image_batches)}", Colors.YELLOW)

                if use_tiling:
                    processed_batch = self._process_with_tiling(
                        upscale_model, batch, tile_size_px, model_scale, dtype, device
                    )
                else:
                    batch = batch.to(dtype)
                    with torch.no_grad():
                        if dtype in [torch.float16, torch.bfloat16]:
                            with torch.autocast(device_type=device.type, dtype=dtype):
                                processed_batch = self.image_upscaler.upscale(upscale_model, batch)[0]
                        else:
                            processed_batch = self.image_upscaler.upscale(upscale_model, batch)[0]

                processed_batches.append(processed_batch)
            full_upscaled = torch.cat(processed_batches, dim=0)

        finally:
            if offload_model and device != offload_device:
                upscale_model.to(offload_device)
            self._cleanup_memory()
        current_h, current_w = full_upscaled.shape[1], full_upscaled.shape[2]

        if current_h != target_h or current_w != target_w:
            colored_print(f"  🔄 Resizing from {current_w}x{current_h} to {target_w}x{target_h} (Lanczos)", Colors.CYAN)
            full_upscaled = comfy.utils.lanczos(full_upscaled.permute(0, 3, 1, 2), target_w, target_h).permute(
                0, 2, 3, 1
            )

        final_result = torch.clamp(full_upscaled, 0, 1).to(image.dtype).to(image.device)

        # Only cache if caching is enabled
        if not disable_cache:
            self._cache[cache_key] = final_result.clone()
            self._manage_cache_size()

        final_width = final_result.shape[2]
        final_height = final_result.shape[1]

        colored_print(f"  ✅ Upscaling complete! Output: {final_width}x{final_height}", Colors.GREEN)

        if not disable_cache:
            colored_print("  💾 Result cached for future use", Colors.CYAN)
        else:
            colored_print("  🚫 Caching disabled - result not stored", Colors.YELLOW)

        if offload_model:
            colored_print("  💾 Model offloaded to save VRAM", Colors.BLUE)

        return (final_result, final_width, final_height)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {"CRT_UpscaleModelAdv": CRT_UpscaleModelAdv}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_UpscaleModelAdv": "Upscale using model adv (CRT)"}

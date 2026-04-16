import re
from pathlib import Path

import numpy as np
import torch
import comfy.utils
from PIL import Image


VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp'}


class CRT_ImageLoaderCrawlBatch:
    def __init__(self):
        self.cache = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def natural_sort_key(p):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', p.name)]

    def _resize_to_megapixels(self, t, megapixels, quantize=8):
        """Resize [1, H, W, C] to target megapixels preserving AR, quantized to `quantize`."""
        _, H, W, _ = t.shape
        scale = (megapixels * 1_000_000 / (H * W)) ** 0.5
        new_H = max(quantize, round(H * scale / quantize) * quantize)
        new_W = max(quantize, round(W * scale / quantize) * quantize)
        if new_H == H and new_W == W:
            return t
        return comfy.utils.common_upscale(
            t.movedim(-1, 1), new_W, new_H, "lanczos", "disabled"
        ).movedim(1, -1)

    def _center_crop(self, t, target_H, target_W):
        """Center-crop [1, H, W, C] to (target_H, target_W)."""
        _, H, W, _ = t.shape
        top  = (H - target_H) // 2
        left = (W - target_W) // 2
        return t[:, top:top + target_H, left:left + target_W, :]

    def _resize_exact(self, t, target_H, target_W):
        """Resize [1, H, W, C] to exact (target_H, target_W)."""
        if t.shape[1] == target_H and t.shape[2] == target_W:
            return t
        return comfy.utils.common_upscale(
            t.movedim(-1, 1), target_W, target_H, "lanczos", "disabled"
        ).movedim(1, -1)

    # ── Node definition ───────────────────────────────────────────────────────

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "batch_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Number of images to load. Window starts at seed × batch_count and wraps around the folder.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Selects the starting image: index = (seed × batch_count) % total_images.",
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Target resolution in megapixels. Each image is resized (lanczos) "
                                   "preserving its aspect ratio. If the batch contains mixed aspect ratios, "
                                   "images are center-cropped to the average AR before batching.",
                    },
                ),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "remove_extension": ("BOOLEAN", {"default": False}),
                "print_index": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Print each selected file index and name to the console."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "file_names", "file_paths", "batch_count", "total_images")
    FUNCTION = "load_batch"
    CATEGORY = "CRT/Load"

    # ── Main ──────────────────────────────────────────────────────────────────

    def load_batch(self, folder_path, batch_count, seed, megapixels, crawl_subfolders, remove_extension, print_index):
        TAG = "[CRT Image Loader Crawl Batch]"

        def blank():
            return torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        if not folder_path or not folder_path.strip():
            return (blank(), "Error: folder path is empty", "", 0, 0)

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"{TAG} ERROR: Folder '{folder}' not found.")
            return (blank(), "Error: folder not found", "", 0, 0)

        # ── Cache ─────────────────────────────────────────────────────────────
        cache_key   = str(folder.resolve()) + ("_sub" if crawl_subfolders else "")
        cur_mtime   = folder.stat().st_mtime

        if cache_key not in self.cache or self.cache[cache_key]["mtime"] != cur_mtime:
            print(f"{TAG} Scanning '{folder}'...")
            try:
                it    = folder.rglob("*") if crawl_subfolders else folder.glob("*")
                files = sorted(
                    [p for p in it if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS],
                    key=self.natural_sort_key,
                )
                self.cache[cache_key] = {"files": files, "mtime": cur_mtime}
                print(f"{TAG} Found {len(files)} images.")
            except Exception as e:
                print(f"{TAG} ERROR scanning: {e}")
                self.cache.pop(cache_key, None)
                return (blank(), f"Error: {e}", "", 0, 0)

        files = self.cache[cache_key]["files"]
        total = len(files)

        if total == 0:
            print(f"{TAG} No images found in '{folder}'.")
            return (blank(), "No images found", "", 0, 0)

        # ── Select batch window ───────────────────────────────────────────────
        start           = (seed * batch_count) % total
        selected_indices = [(start + i) % total for i in range(batch_count)]

        # ── Load & resize ─────────────────────────────────────────────────────
        tensors = []
        names   = []
        paths   = []

        for idx in selected_indices:
            f = files[idx]
            try:
                with Image.open(f) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    t = torch.from_numpy(
                        np.array(img).astype(np.float32) / 255.0
                    ).unsqueeze(0)   # [1, H, W, C]

                t = self._resize_to_megapixels(t, megapixels)
                tensors.append(t)

                name = f.stem if remove_extension else f.name
                names.append(name)
                paths.append(str(f.resolve()))

                if print_index:
                    print(f"{TAG} [{idx + 1}/{total}] {name}")

            except Exception as e:
                print(f"{TAG} ERROR loading '{f}': {e}")
                # Insert blank so batch count stays correct
                tensors.append(blank())
                names.append(f"Error: {f.name}")
                paths.append(str(f.resolve()))

        # ── Mixed aspect ratio fallback ───────────────────────────────────────
        shapes = [(t.shape[1], t.shape[2]) for t in tensors]   # (H, W)

        if len(set(shapes)) > 1:
            print(
                f"{TAG} Mixed resolutions detected — computing average AR "
                f"and center-cropping to a uniform size."
            )

            ars     = [W / H for H, W in shapes]
            avg_ar  = sum(ars) / len(ars)

            # Target size: ~megapixels MP at avg_ar, quantized to 8
            H_tgt = max(8, round((megapixels * 1_000_000 / avg_ar) ** 0.5 / 8) * 8)
            W_tgt = max(8, round(H_tgt * avg_ar / 8) * 8)

            unified = []
            for t in tensors:
                _, H, W, _ = t.shape
                cur_ar = W / H

                if cur_ar > avg_ar:
                    # Too wide → crop width to match avg_ar
                    crop_W = max(1, min(round(H * avg_ar), W))
                    t = self._center_crop(t, H, crop_W)
                elif cur_ar < avg_ar:
                    # Too tall → crop height to match avg_ar
                    crop_H = max(1, min(round(W / avg_ar), H))
                    t = self._center_crop(t, crop_H, W)

                # Final exact resize so every tensor is identical
                t = self._resize_exact(t, H_tgt, W_tgt)
                unified.append(t)

            tensors = unified

        # ── Stack ─────────────────────────────────────────────────────────────
        batch = torch.cat(tensors, dim=0)   # [B, H, W, C]

        return (
            batch,
            "\n".join(names),
            "\n".join(paths),
            len(tensors),
            total,
        )


NODE_CLASS_MAPPINGS       = {"CRT_ImageLoaderCrawlBatch": CRT_ImageLoaderCrawlBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"CRT_ImageLoaderCrawlBatch": "Image Loader Crawl Batch (CRT)"}

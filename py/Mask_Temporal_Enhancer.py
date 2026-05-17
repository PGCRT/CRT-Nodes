import numpy as np
import torch
import torch.nn.functional as F


class MaskTemporalEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "processing_device": (["auto", "input", "cuda", "cpu"], {"default": "auto"}),
                "auto_repair": (
                    ["off", "conservative", "balanced", "aggressive"],
                    {"default": "balanced"},
                ),
                "repair_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_repair_gap": (
                    "INT",
                    {"default": 3, "min": 1, "max": 24, "step": 1},
                ),
                "temporal_window": (
                    "INT",
                    {"default": 5, "min": 1, "max": 31, "step": 2},
                ),
                "temporal_strength": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "edge_strength": (
                    "FLOAT",
                    {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "edge_radius": (
                    "INT",
                    {"default": 3, "min": 1, "max": 32, "step": 1},
                ),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "hole_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "cleanup_mode": (
                    ["off", "close gaps", "open speckles", "close then open"],
                    {"default": "close gaps"},
                ),
                "cleanup_radius": (
                    "INT",
                    {"default": 1, "min": 0, "max": 16, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "enhance"
    CATEGORY = "CRT/Mask"

    _REPAIR_PROFILES = {
        "conservative": {
            "endpoint_iou": 0.55,
            "missing_area_ratio": 0.30,
            "outlier_iou": 0.12,
            "area_delta": 0.75,
        },
        "balanced": {
            "endpoint_iou": 0.35,
            "missing_area_ratio": 0.50,
            "outlier_iou": 0.20,
            "area_delta": 0.55,
        },
        "aggressive": {
            "endpoint_iou": 0.20,
            "missing_area_ratio": 0.72,
            "outlier_iou": 0.32,
            "area_delta": 0.35,
        },
    }

    def enhance(
        self,
        mask,
        processing_device,
        auto_repair,
        repair_strength,
        max_repair_gap,
        temporal_window,
        temporal_strength,
        edge_strength,
        edge_radius,
        fill_holes,
        hole_threshold,
        cleanup_mode,
        cleanup_radius,
    ):
        input_device = mask.device if isinstance(mask, torch.Tensor) else torch.device("cpu")
        device = self._resolve_device(input_device, processing_device)
        mask = self._normalize_mask(mask).to(device=device)

        if fill_holes:
            mask = self._fill_holes(mask, hole_threshold)

        if mask.shape[0] > 1 and auto_repair != "off" and repair_strength > 0.0:
            mask = self._repair_temporal_glitches(
                mask,
                auto_repair,
                repair_strength,
                hole_threshold,
                max_repair_gap,
            )

        mask = self._spatial_cleanup(mask, cleanup_mode, cleanup_radius)

        if mask.shape[0] > 1 and temporal_strength > 0.0 and edge_strength > 0.0:
            mask = self._stabilize_edges(
                mask,
                temporal_window,
                temporal_strength,
                edge_strength,
                edge_radius,
            )

        return (torch.clamp(mask, 0.0, 1.0).to(device=input_device),)

    @staticmethod
    def _resolve_device(input_device, processing_device):
        if processing_device == "cpu":
            return torch.device("cpu")
        if processing_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Mask Temporal Enhancer (CRT): CUDA requested but not available.")
            return torch.device("cuda")
        if processing_device == "input":
            return input_device
        if input_device.type == "cuda":
            return input_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return input_device

    @staticmethod
    def _normalize_mask(mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))

        mask = torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0)

        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        elif mask.dim() == 4:
            mask = mask.mean(dim=-1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if mask.dim() != 3:
            raise ValueError(
                f"Mask Temporal Enhancer (CRT): MASK must be [H,W] or [B,H,W], got {tuple(mask.shape)}"
            )

        return torch.clamp(mask, 0.0, 1.0)

    @staticmethod
    def _pool(mask, radius, mode):
        radius = int(radius)
        if radius <= 0:
            return mask

        kernel = radius * 2 + 1
        x = F.pad(mask.unsqueeze(1), (radius, radius, radius, radius), mode="replicate")

        if mode == "dilate":
            return F.max_pool2d(x, kernel_size=kernel, stride=1).squeeze(1)

        return (-F.max_pool2d(-x, kernel_size=kernel, stride=1)).squeeze(1)

    @classmethod
    def _dilate(cls, mask, radius):
        return cls._pool(mask, radius, "dilate")

    @classmethod
    def _erode(cls, mask, radius):
        return cls._pool(mask, radius, "erode")

    @classmethod
    def _close(cls, mask, radius):
        return cls._erode(cls._dilate(mask, radius), radius)

    @classmethod
    def _open(cls, mask, radius):
        return cls._dilate(cls._erode(mask, radius), radius)

    @classmethod
    def _spatial_cleanup(cls, mask, cleanup_mode, cleanup_radius):
        radius = int(cleanup_radius)
        if cleanup_mode == "off" or radius <= 0:
            return mask

        if cleanup_mode == "close gaps":
            return cls._close(mask, radius)
        if cleanup_mode == "open speckles":
            return cls._open(mask, radius)

        return cls._open(cls._close(mask, radius), radius)

    @classmethod
    def _stabilize_edges(cls, mask, temporal_window, temporal_strength, edge_strength, edge_radius):
        smoothed = cls._temporal_triangle_blur(mask, temporal_window)

        outer = cls._dilate(mask, edge_radius)
        inner = cls._erode(mask, edge_radius)
        edge_band = torch.clamp(outer - inner, 0.0, 1.0)

        local_delta = cls._temporal_triangle_blur(torch.abs(mask - smoothed), temporal_window)
        consistency = torch.clamp(1.0 - local_delta * 3.0, 0.0, 1.0)

        blend = float(temporal_strength) * float(edge_strength) * edge_band
        blend = blend * (0.25 + 0.75 * consistency)

        return torch.lerp(mask, smoothed, torch.clamp(blend, 0.0, 1.0))

    @staticmethod
    def _temporal_triangle_blur(mask, temporal_window):
        batch = int(mask.shape[0])
        window = max(1, int(temporal_window))
        if batch <= 1 or window <= 1:
            return mask

        if window % 2 == 0:
            window += 1

        half = window // 2
        weights = torch.arange(1, half + 2, device=mask.device, dtype=mask.dtype)
        weights = torch.cat((weights, weights[:-1].flip(0)))
        weights = weights / weights.sum()

        x = mask.unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (0, 0, 0, 0, half, half), mode="replicate")
        kernel = weights.view(1, 1, window, 1, 1)
        return F.conv3d(x, kernel).squeeze(0).squeeze(0)

    @classmethod
    def _repair_temporal_glitches(cls, mask, profile_name, repair_strength, threshold, max_repair_gap):
        profile = cls._REPAIR_PROFILES.get(profile_name, cls._REPAIR_PROFILES["balanced"])
        repaired = mask.clone()
        batch, height, width = repaired.shape
        total_pixels = float(height * width)
        min_valid_area = max(8.0, total_pixels * 0.0005)

        binary = repaired >= float(threshold)
        areas = [float(binary[i].sum().item()) for i in range(batch)]
        touched = [False] * batch

        max_gap = max(1, min(int(max_repair_gap), batch - 2))
        for gap_len in range(max_gap, 0, -1):
            start = 1
            while start + gap_len < batch:
                end = start + gap_len - 1
                left = start - 1
                right = end + 1

                if any(touched[start : end + 1]):
                    start += 1
                    continue

                endpoint_area = (areas[left] + areas[right]) * 0.5
                if endpoint_area < min_valid_area:
                    start += 1
                    continue

                endpoint_iou = cls._binary_iou(binary[left], binary[right])
                max_missing_area = endpoint_area * profile["missing_area_ratio"]
                gap_is_missing = all(areas[i] <= max_missing_area for i in range(start, end + 1))

                if endpoint_iou >= profile["endpoint_iou"] and gap_is_missing:
                    span = float(right - left)
                    for frame in range(start, end + 1):
                        alpha = float(frame - left) / span
                        interpolated = torch.lerp(repaired[left], repaired[right], alpha)
                        repaired[frame] = torch.lerp(repaired[frame], interpolated, float(repair_strength))
                        touched[frame] = True
                    start = end + 1
                else:
                    start += 1

        binary = repaired >= float(threshold)
        areas = [float(binary[i].sum().item()) for i in range(batch)]

        for index in range(1, batch - 1):
            if touched[index]:
                continue

            endpoint_area = (areas[index - 1] + areas[index + 1]) * 0.5
            if endpoint_area < min_valid_area:
                continue

            endpoint_iou = cls._binary_iou(binary[index - 1], binary[index + 1])
            if endpoint_iou < profile["endpoint_iou"]:
                continue

            previous_iou = cls._binary_iou(binary[index], binary[index - 1])
            next_iou = cls._binary_iou(binary[index], binary[index + 1])
            area_delta = abs(areas[index] - endpoint_area) / max(endpoint_area, 1.0)
            missing = areas[index] <= endpoint_area * profile["missing_area_ratio"]
            wrong_shape = max(previous_iou, next_iou) <= profile["outlier_iou"]
            unstable_area = area_delta >= profile["area_delta"] and min(previous_iou, next_iou) < 0.45

            if missing or wrong_shape or unstable_area:
                interpolated = (repaired[index - 1] + repaired[index + 1]) * 0.5
                repaired[index] = torch.lerp(repaired[index], interpolated, float(repair_strength))

        return torch.clamp(repaired, 0.0, 1.0)

    @staticmethod
    def _binary_iou(a, b):
        union = torch.logical_or(a, b).sum().item()
        if union == 0:
            return 1.0

        intersection = torch.logical_and(a, b).sum().item()
        return float(intersection) / float(union)

    @classmethod
    def _fill_holes(cls, mask, threshold):
        frames = []
        device = mask.device
        dtype = mask.dtype

        for frame in mask:
            binary = frame.detach().cpu().numpy() >= float(threshold)
            holes = cls._find_holes(binary)

            if holes.any():
                holes_tensor = torch.from_numpy(holes).to(device=device, dtype=dtype)
                frame = torch.where(holes_tensor > 0.5, torch.ones_like(frame), frame)

            frames.append(frame)

        return torch.stack(frames, dim=0)

    @staticmethod
    def _find_holes(foreground):
        try:
            import cv2

            background = np.logical_not(foreground).astype(np.uint8)
            labels_count, labels = cv2.connectedComponents(background, connectivity=4)
            if labels_count <= 1:
                return np.zeros_like(foreground, dtype=bool)

            border_labels = np.unique(
                np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
            )
            outside = np.isin(labels, border_labels)
            return np.logical_and(background.astype(bool), np.logical_not(outside))
        except Exception:
            return MaskTemporalEnhancer._find_holes_torch_fallback(foreground)

    @staticmethod
    def _find_holes_torch_fallback(foreground):
        fg = torch.from_numpy(foreground.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        bg = 1.0 - fg
        outside = torch.zeros_like(bg)
        outside[:, :, 0, :] = bg[:, :, 0, :]
        outside[:, :, -1, :] = bg[:, :, -1, :]
        outside[:, :, :, 0] = bg[:, :, :, 0]
        outside[:, :, :, -1] = bg[:, :, :, -1]

        for _ in range(int(foreground.shape[0] + foreground.shape[1])):
            grown = F.max_pool2d(F.pad(outside, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
            grown = grown * bg
            if torch.equal(grown, outside):
                break
            outside = grown

        holes = (bg > 0.5) & (outside < 0.5)
        return holes.squeeze(0).squeeze(0).numpy().astype(bool)

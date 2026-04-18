import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils


def load_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


class MaskCensor:
    def __init__(self):
        self.detectors = {}

    @classmethod
    def INPUT_TYPES(cls):
        try:
            segm_files = folder_paths.get_filename_list("ultralytics_segm")
        except Exception:
            segm_files = []

        return {
            "required": {
                "image": ("IMAGE",),
                "face_segm_model": ((["segm/" + x for x in segm_files] or ["segm/"]),),
                "segm_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "censor_type": (["blur", "pixelate"], {"default": "blur"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 1.0,
                    },
                ),
                "mask_expand": (
                    "INT",
                    {
                        "default": 10,
                        "min": -64,
                        "max": 64,
                        "step": 1,
                    },
                ),
                "mask_blur": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.5,
                    },
                ),
                "processing_device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto"},
                ),
                "chunk_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "mask_override": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("filtered_image",)
    FUNCTION = "execute"
    CATEGORY = "CRT/Mask"

    @staticmethod
    def _apply_blur(images, strength):
        kernel = int(round(strength * 2.0)) + 1
        kernel = max(1, min(kernel, 51))
        if kernel % 2 == 0:
            kernel += 1

        if kernel <= 1:
            return images

        nchw = images.movedim(-1, 1)
        pad = kernel // 2
        padded = F.pad(nchw, (pad, pad, pad, pad), mode="replicate")
        blurred = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
        return torch.clamp(blurred.movedim(1, -1), 0.0, 1.0)

    @staticmethod
    def _apply_pixelate(images, strength):
        h = images.shape[1]
        w = images.shape[2]
        block = max(1, min(int(round(strength)), 64))
        if block <= 1:
            return images

        down_h = max(1, h // block)
        down_w = max(1, w // block)

        nchw = images.movedim(-1, 1)
        small = F.interpolate(nchw, size=(down_h, down_w), mode="nearest")
        pixelated = F.interpolate(small, size=(h, w), mode="nearest")
        return torch.clamp(pixelated.movedim(1, -1), 0.0, 1.0)

    @staticmethod
    def _expand_and_blur_mask(mask_2d, expand, blur):
        m = mask_2d.unsqueeze(0).unsqueeze(0)

        exp_px = int(expand)
        if exp_px != 0:
            k = max(1, exp_px * 2 + 1)
            if exp_px > 0:
                m = F.max_pool2d(m, kernel_size=k, stride=1, padding=exp_px)
            else:
                p = abs(exp_px)
                k = max(1, p * 2 + 1)
                m = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=p)

        blur_px = int(round(float(blur)))
        if blur_px > 0:
            k = max(1, blur_px * 2 + 1)
            if k % 2 == 0:
                k += 1
            p = k // 2
            m = F.avg_pool2d(F.pad(m, (p, p, p, p), mode="replicate"), k, stride=1)

        return torch.clamp(m.squeeze(0).squeeze(0), 0.0, 1.0)

    @staticmethod
    def _resolve_device(image_tensor, processing_device):
        if processing_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Mask Censor (CRT): CUDA requested but not available.")
            return torch.device("cuda")

        if processing_device == "cpu":
            return torch.device("cpu")

        if image_tensor.is_cuda:
            return image_tensor.device

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def _align_mask_batch(mask, batch):
        current = int(mask.shape[0])

        if current == batch:
            return mask

        if current == 1:
            return mask.expand(batch, -1, -1)

        raise ValueError(
            "Mask Censor (CRT): MASK batch does not match IMAGE batch. "
            f"Got MASK={current}, IMAGE={batch}. "
            "Provide [B,H,W] or a single [H,W]/[1,H,W] override mask."
        )

    @staticmethod
    def _normalize_mask_batch(mask, batch, height, width, device, dtype):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))

        mask = mask.float()
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if mask.dim() != 3:
            raise ValueError(
                f"Mask Censor (CRT): MASK must be [H,W] or [B,H,W], got shape {tuple(mask.shape)}"
            )

        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[1] != height or mask.shape[2] != width:
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        mask = MaskCensor._align_mask_batch(mask, batch)

        return torch.clamp(mask, 0.0, 1.0)

    @staticmethod
    def _frame_to_numpy_uint8(frame):
        arr = frame.detach().float().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    def _get_detector(self, face_segm_model):
        segm_filename_only = face_segm_model.split("/")[-1]
        segm_full_path = folder_paths.get_full_path("ultralytics_segm", segm_filename_only)
        if not segm_full_path:
            raise ValueError(f"Mask Censor (CRT): could not resolve segm model '{face_segm_model}'.")

        if face_segm_model not in self.detectors:
            self.detectors[face_segm_model] = load_yolo(segm_full_path)

        return self.detectors[face_segm_model]

    def _detect_mask_single(self, model, frame, segm_threshold, height, width, device, dtype):
        frame_np = self._frame_to_numpy_uint8(frame)
        pred = model(frame_np, conf=float(segm_threshold), verbose=False)
        if (
            not pred
            or not hasattr(pred[0], "masks")
            or pred[0].masks is None
            or pred[0].masks.data is None
            or pred[0].masks.data.nelement() == 0
        ):
            return torch.zeros((height, width), dtype=dtype, device=device)

        segms = pred[0].masks.data.to(device=device, dtype=dtype)
        segms = F.interpolate(
            segms.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return torch.clamp(segms.max(dim=0).values, 0.0, 1.0)

    def _detect_mask_batch(self, image, face_segm_model, segm_threshold, device, dtype):
        model = self._get_detector(face_segm_model)
        batch, height, width = image.shape[:3]
        masks = []
        pbar = comfy.utils.ProgressBar(batch)

        for b in range(batch):
            masks.append(
                self._detect_mask_single(
                    model,
                    image[b],
                    segm_threshold,
                    height,
                    width,
                    device,
                    dtype,
                )
            )
            pbar.update(1)

        return torch.stack(masks, dim=0)

    def execute(
        self,
        image,
        face_segm_model,
        segm_threshold,
        censor_type,
        strength,
        mask_expand,
        mask_blur,
        processing_device,
        chunk_size,
        mask_override=None,
    ):
        target_device = self._resolve_device(image, processing_device)
        output = image.clone()
        batch, height, width = output.shape[:3]
        pbar = comfy.utils.ProgressBar(batch)

        model = None
        if mask_override is None:
            model = self._get_detector(face_segm_model)

        mask_batch = None
        if mask_override is not None:
            mask_batch = self._normalize_mask_batch(
                mask_override,
                batch,
                height,
                width,
                output.device,
                output.dtype,
            )

        chunk_size = max(1, int(chunk_size))

        for start in range(0, batch, chunk_size):
            end = min(batch, start + chunk_size)
            chunk = output[start:end].to(device=target_device)

            for local_idx, b in enumerate(range(start, end)):
                if mask_batch is not None:
                    frame_mask = mask_batch[b if mask_batch.shape[0] > 1 else 0]
                    frame_mask = frame_mask.to(device=target_device, dtype=chunk.dtype)
                else:
                    frame_mask = self._detect_mask_single(
                        model,
                        image[b],
                        segm_threshold,
                        height,
                        width,
                        target_device,
                        chunk.dtype,
                    )

                frame_mask = self._expand_and_blur_mask(frame_mask, mask_expand, mask_blur)
                nz = torch.nonzero(frame_mask > 1e-6, as_tuple=False)
                if nz.numel() == 0:
                    pbar.update(1)
                    continue

                y1 = int(nz[:, 0].min().item())
                y2 = int(nz[:, 0].max().item()) + 1
                x1 = int(nz[:, 1].min().item())
                x2 = int(nz[:, 1].max().item()) + 1

                frame_roi = chunk[local_idx : local_idx + 1, y1:y2, x1:x2, :]
                mask_roi = frame_mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(-1)

                if censor_type == "pixelate":
                    filtered_roi = self._apply_pixelate(frame_roi, strength)
                else:
                    filtered_roi = self._apply_blur(frame_roi, strength)

                chunk[local_idx : local_idx + 1, y1:y2, x1:x2, :] = (
                    frame_roi * (1.0 - mask_roi) + filtered_roi * mask_roi
                )
                pbar.update(1)

            output[start:end] = chunk.to(device=output.device)
            del chunk

            if target_device.type == "cuda":
                torch.cuda.empty_cache()

        return (output,)


NODE_CLASS_MAPPINGS = {"MaskCensor": MaskCensor}

NODE_DISPLAY_NAME_MAPPINGS = {"MaskCensor": "Mask Censor (CRT)"}

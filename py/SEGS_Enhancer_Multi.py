import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import folder_paths
import node_helpers
import comfy.utils
import latent_preview
import comfy.sd
import comfy.sample
import comfy.samplers
import scipy.ndimage
import cv2

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate as F_interpolate


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def colored_print(message, color=Colors.ENDC):
    safe_message = str(message).encode("ascii", "ignore").decode("ascii")
    print(f"{color}{safe_message}{Colors.ENDC}")


def load_yolo(model_path: str):
    colored_print(f"🔄 Loading YOLO model from: {model_path}", Colors.CYAN)
    from ultralytics import YOLO
    import ultralytics.nn.tasks as nn_tasks

    original_torch_safe_load = nn_tasks.torch_safe_load

    def unsafe_pt_loader(weight, map_location="cpu"):
        ckpt = torch.load(weight, map_location=map_location, weights_only=False)
        return ckpt, weight

    try:
        nn_tasks.torch_safe_load = unsafe_pt_loader
        model = YOLO(model_path)
        colored_print("✅ YOLO model loaded successfully!", Colors.GREEN)
    finally:
        nn_tasks.torch_safe_load = original_torch_safe_load
    return model


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def combine_masks(segmasks):
    if not segmasks:
        return None
    combined_mask = np.zeros_like(segmasks[0][1], dtype=np.float32)
    for _, mask, _ in segmasks:
        combined_mask += mask
    return torch.from_numpy(np.clip(combined_mask, 0, 1))


def dilate_masks(segmasks, dilation_factor):
    if dilation_factor == 0:
        return segmasks
    dilated_segmasks = []
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    colored_print(
        f"   🔧 Applying dilation with {dilation_factor}x{dilation_factor} kernel",
        Colors.BLUE,
    )
    for bbox, mask, confidence in segmasks:
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_segmasks.append((bbox, dilated_mask, confidence))
    return dilated_segmasks


def create_segmasks(results):
    if not results or not results[1]:
        return []
    bboxs, segms, confidence = results[1], results[2], results[3]
    output = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        output.append(item)
    colored_print(f"   📊 Created {len(output)} segmentation mask(s)", Colors.BLUE)
    return output


def inference_bbox(model, image: Image.Image, confidence: float = 0.3):
    colored_print(
        f"   🔍 Running bbox detection (confidence: {confidence:.2f})", Colors.BLUE
    )
    pred = model(image, conf=confidence)
    if (
        not pred
        or not hasattr(pred[0], "boxes")
        or pred[0].boxes is None
        or pred[0].boxes.xyxy.nelement() == 0
    ):
        colored_print("   ⚠️ No detections found", Colors.YELLOW)
        return [[], [], [], []]

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    colored_print(f"   ✅ Found {len(bboxes)} bounding box(es)", Colors.GREEN)

    cv2_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        segms.append(cv2_mask.astype(bool))
    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())
    return results


def inference_segm(model, image: Image.Image, confidence: float = 0.3):
    colored_print(
        f"   🔍 Running segmentation detection (confidence: {confidence:.2f})",
        Colors.BLUE,
    )
    pred = model(image, conf=confidence)
    if (
        not pred
        or not hasattr(pred[0], "masks")
        or pred[0].masks is None
        or pred[0].masks.data.nelement() == 0
    ):
        colored_print("   ⚠️ No segmentation masks found", Colors.YELLOW)
        return [[], [], [], []]

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    segms = pred[0].masks.data.cpu().numpy()
    colored_print(f"   ✅ Found {len(bboxes)} segmentation mask(s)", Colors.GREEN)

    results = [[], [], [], []]
    for i in range(bboxes.shape[0]):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        mask = torch.from_numpy(segms[i])
        scaled_mask = F_interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(image.size[1], image.size[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())
    return results


class UltraDetector:
    def __init__(self, model_path, detection_type):
        colored_print(
            f"🤖 Initializing {detection_type.upper()} detector...", Colors.CYAN
        )
        self.model = load_yolo(model_path)
        self.type = detection_type
        colored_print(f"✅ {detection_type.upper()} detector ready!", Colors.GREEN)

    def detect_combined(self, image, threshold, dilation):
        colored_print(f"🔍 Running {self.type.upper()} detection...", Colors.CYAN)
        pil_image = tensor2pil(image)
        colored_print(f"   📐 Detection image size: {pil_image.size}", Colors.BLUE)

        if self.type == "bbox":
            detected_results = inference_bbox(self.model, pil_image, threshold)
        else:
            detected_results = inference_segm(self.model, pil_image, threshold)

        if not detected_results or not detected_results[0]:
            colored_print("   ❌ No faces detected", Colors.YELLOW)
            return torch.zeros((pil_image.height, pil_image.width), dtype=torch.float32)

        segmasks = create_segmasks(detected_results)
        if not segmasks:
            colored_print("   ❌ No valid segmentation masks created", Colors.YELLOW)
            return torch.zeros((pil_image.height, pil_image.width), dtype=torch.float32)

        if dilation > 0:
            colored_print(f"   🔧 Applying dilation factor: {dilation}", Colors.BLUE)
            segmasks = dilate_masks(segmasks, dilation)

        final_mask = combine_masks(segmasks)
        mask_coverage = final_mask.sum().item() / (final_mask.numel())
        colored_print(
            f"   📊 Final mask coverage: {mask_coverage * 100:.2f}%", Colors.GREEN
        )

        return final_mask


class ImageResize:
    def execute(self, image, width, height, method="stretch", interpolation="lanczos"):
        _, oh, ow, _ = image.shape
        colored_print(f"🔄 Resizing image from {ow}x{oh}...", Colors.CYAN)

        if method == "keep proportion":
            if ow > oh:
                ratio = width / ow
            else:
                ratio = height / oh
            width, height = round(ow * ratio), round(oh * ratio)
            colored_print(
                f"   📐 Keeping proportions - Target: {width}x{height} (ratio: {ratio:.3f})",
                Colors.BLUE,
            )
        else:
            width, height = (width if width > 0 else ow), (height if height > 0 else oh)
            colored_print(f"   📐 Direct resize to: {width}x{height}", Colors.BLUE)

        outputs = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos":
            colored_print("   🎯 Using Lanczos interpolation", Colors.BLUE)
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            colored_print(f"   🎯 Using {interpolation} interpolation", Colors.BLUE)
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        colored_print(f"✅ Image resize completed: {width}x{height}", Colors.GREEN)
        return torch.clamp(outputs.permute(0, 2, 3, 1), 0, 1)


class ColorMatch:
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        colored_print(
            f"🎨 Applying color matching (method: {method}, strength: {strength:.2f})...",
            Colors.CYAN,
        )

        try:
            from color_matcher import ColorMatcher
        except Exception:
            colored_print("❌ ERROR: 'color-matcher' library not found.", Colors.RED)
            raise Exception(
                "ColorMatch requires 'color-matcher'. Please 'pip install color-matcher'"
            )

        cm, out = ColorMatcher(), []
        image_ref, image_target = image_ref.cpu(), image_target.cpu()

        batch_size = image_target.size(0)
        colored_print(f"   🔄 Processing {batch_size} image(s)...", Colors.BLUE)

        for i in range(batch_size):
            target_np, ref_np = (
                image_target[i].numpy(),
                image_ref[
                    i if image_ref.size(0) == image_target.size(0) else 0
                ].numpy(),
            )
            target_mean = np.mean(target_np, axis=(0, 1))
            ref_mean = np.mean(ref_np, axis=(0, 1))

            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            result = target_np + strength * (result - target_np)

            final_mean = np.mean(result, axis=(0, 1))
            colored_print(
                f"   📊 Image {i + 1}: Target RGB({target_mean[0]:.3f},{target_mean[1]:.3f},{target_mean[2]:.3f}) → Final RGB({final_mean[0]:.3f},{final_mean[1]:.3f},{final_mean[2]:.3f})",
                Colors.BLUE,
            )

            out.append(torch.from_numpy(result))

        colored_print("✅ Color matching completed!", Colors.GREEN)
        return (torch.stack(out, dim=0).to(torch.float32).clamp_(0, 1),)


class GrowMaskWithBlur:
    def expand_mask(self, mask, expand, tapered_corners, blur_radius, taper_radius=0):
        colored_print(
            f"🎭 Processing mask (expand: {expand}, blur: {blur_radius}, taper: {taper_radius})...",
            Colors.CYAN,
        )

        c, out = (0 if tapered_corners else 1), []
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])

        for m in mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu():
            output = m.numpy().astype(np.float32)
            original_coverage = np.sum(output) / output.size

            if expand != 0:
                colored_print(
                    f"   🔧 Applying {abs(expand)} iterations of {'dilation' if expand > 0 else 'erosion'}",
                    Colors.BLUE,
                )
                for _ in range(abs(round(expand))):
                    if expand < 0:
                        output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                    else:
                        output = scipy.ndimage.grey_dilation(output, footprint=kernel)

            final_coverage = np.sum(output) / output.size
            colored_print(
                f"   📊 Mask coverage: {original_coverage * 100:.2f}% → {final_coverage * 100:.2f}%",
                Colors.BLUE,
            )

            out.append(torch.from_numpy(output))

        final_mask = torch.stack(out, dim=0)

        # Apply Border Taper
        if taper_radius > 0:
            h, w = final_mask.shape[-2], final_mask.shape[-1]
            t = min(int(taper_radius), h // 2, w // 2)

            if t > 0:
                colored_print(
                    f"   🖼️ Applying border taper fade (radius: {t}px)", Colors.BLUE
                )

                v_fade = torch.ones((h, 1), dtype=torch.float32)
                v_fade[:t, 0] = torch.linspace(0, 1, t)
                v_fade[-t:, 0] = torch.linspace(1, 0, t)

                h_fade = torch.ones((1, w), dtype=torch.float32)
                h_fade[0, :t] = torch.linspace(0, 1, t)
                h_fade[0, -t:] = torch.linspace(1, 0, t)

                border_mask = v_fade * h_fade
                final_mask = final_mask * border_mask.to(final_mask.device)

        if blur_radius > 0:
            colored_print(
                f"   🌫️ Applying Gaussian blur (radius: {blur_radius})", Colors.BLUE
            )
            pil_img = to_pil_image(final_mask)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
            final_mask = to_tensor(pil_img)

        colored_print("✅ Mask processing completed!", Colors.GREEN)
        return (final_mask, 1.0 - final_mask)


class BoundedImageCropWithMask:
    def crop(self, image, mask, top, bottom, left, right):
        colored_print(
            f"✂️ Cropping image with padding (t:{top}, b:{bottom}, l:{left}, r:{right})...",
            Colors.CYAN,
        )

        image, mask = (
            (image.unsqueeze(0) if image.dim() == 3 else image),
            (mask.unsqueeze(0) if mask.dim() == 2 else mask),
        )
        cropped_images, all_bounds = [], []

        for i in range(len(image)):
            current_mask = mask[i if len(mask) == len(image) else 0]
            if current_mask.sum() == 0:
                colored_print(
                    f"   ⚠️ Empty mask for image {i + 1}, skipping", Colors.YELLOW
                )
                continue

            rows, cols = torch.any(current_mask, dim=1), torch.any(current_mask, dim=0)

            if not torch.any(rows) or not torch.any(cols):
                colored_print(
                    f"   ⚠️ No valid mask regions for image {i + 1}, skipping",
                    Colors.YELLOW,
                )
                continue

            rmin_t, rmax_t = torch.where(rows)[0][[0, -1]]
            cmin_t, cmax_t = torch.where(cols)[0][[0, -1]]

            rmin = max(rmin_t.item() - top, 0)
            rmax = min(rmax_t.item() + bottom, current_mask.shape[0] - 1)
            cmin = max(cmin_t.item() - left, 0)
            cmax = min(cmax_t.item() + right, current_mask.shape[1] - 1)

            if rmax < rmin or cmax < cmin:
                colored_print(
                    f"   ❌ Invalid crop region for image {i + 1} (zero or negative size), skipping",
                    Colors.RED,
                )
                continue

            crop_w, crop_h = cmax - cmin + 1, rmax - rmin + 1
            colored_print(
                f"   📐 Crop {i + 1}: {crop_w}x{crop_h} at ({cmin},{rmin})", Colors.BLUE
            )

            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin : rmax + 1, cmin : cmax + 1, :])

        if cropped_images:
            colored_print(
                f"✅ Successfully cropped {len(cropped_images)} region(s)!",
                Colors.GREEN,
            )
            return (torch.stack(cropped_images), all_bounds)
        else:
            colored_print("❌ No valid crops produced", Colors.RED)
            return (None, None)


def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade
    return multipliers


def get_dd_schedule_value(sigma, sigmas, dd_schedule):
    sched_len = len(dd_schedule)
    if (
        sched_len < 1
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0

    if sched_len == 1:
        return dd_schedule[0].item()

    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        return dd_schedule[idx].item()

    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        return dd_schedule[idxlow]

    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def detail_daemon_sampler(
    model,
    x,
    sigmas,
    *,
    dds_wrapped_sampler,
    dds_make_schedule,
    dds_cfg_scale_override,
    **kwargs,
):
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )

    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = (
            get_dd_schedule_value(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        )
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in ("inner_model", "sigmas"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))

    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


def split_sigmas_denoise_stateless(sigmas, denoise):
    steps = max(sigmas.shape[-1] - 1, 0)
    total_steps = round(steps * denoise)
    if total_steps <= 0:
        return sigmas[:1], sigmas
    high = sigmas[:-(total_steps)]
    low = sigmas[-(total_steps + 1) :]
    return high, low


def multiply_sigmas_stateless(sigmas, factor, start, end):
    out = sigmas.clone()
    total_sigmas = len(out)
    start_idx = int(start * total_sigmas)
    end_idx = int(end * total_sigmas)
    for i in range(start_idx, end_idx):
        out[i] *= factor
    return out


def build_dd_sampler(
    base_sampler,
    amount,
    start,
    end,
    bias,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
    cfg_scale_override,
):
    def dds_make_schedule(steps):
        return make_detail_daemon_schedule(
            steps,
            start,
            end,
            bias,
            amount,
            exponent,
            start_offset,
            end_offset,
            fade,
            smooth,
        )

    return comfy.samplers.KSAMPLER(
        detail_daemon_sampler,
        extra_options={
            "dds_wrapped_sampler": base_sampler,
            "dds_make_schedule": dds_make_schedule,
            "dds_cfg_scale_override": cfg_scale_override,
        },
    )


class FaceEnhancementWithInjectionSEGS:
    def __init__(self):
        colored_print(
            "🐎 Face Enhancement with Injection SEGS initialized!", Colors.HEADER
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "lcm"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Classifier Free Guidance scale. Higher values follow the prompt more closely.",
                    },
                ),
                "steps": ("INT", {"default": 2, "min": 1, "max": 100}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Amount of denoising to apply. 1.0 = full denoising (txt2img), 0.5-0.8 typical for img2img.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "edit_model_flux2klein": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable per-crop reference latent conditioning for edit models like flux2klein",
                    },
                ),
                "upscale_megapixel": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target megapixels used to upscale each crop before enhancement",
                    },
                ),
                "resize_back_to_original": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If disabled, the whole input is upscaled so refined crops can keep higher detail",
                    },
                ),
                "multi_face_resolution_strategy": (
                    ["fit to smallest", "fit to largest", "optimal"],
                    {
                        "default": "optimal",
                        "tooltip": "Used only when resize_back_to_original is False to decide global upscale for multi-face inputs",
                    },
                ),
                "pre_crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Tightens each SEGS crop around its center before enhancement. Lower is tighter.",
                    },
                ),
                "post_mask_expand": (
                    "INT",
                    {"default": 10, "min": -64, "max": 64, "step": 1},
                ),
                "post_mask_blur": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.0, "max": 64.0, "step": 0.5},
                ),
                "post_mask_taper_borders": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Fades the mask edges to black to prevent hard clipping lines",
                    },
                ),
                "stage1_sigma_factor": (
                    "FLOAT",
                    {"default": 1.010, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage2_sigma_factor": (
                    "FLOAT",
                    {"default": 0.990, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage1_sigma_start": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage2_sigma_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "details_amount_stage1": (
                    "FLOAT",
                    {"default": 0.05, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "details_amount_stage2": (
                    "FLOAT",
                    {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "enable_noise_injection": (
                    ["disable", "enable"],
                    {
                        "default": "enable",
                        "tooltip": "Enable noise injection during sampling",
                    },
                ),
                "injection_point": (
                    "FLOAT",
                    {
                        "default": 0.50,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Percentage of steps after which to inject noise",
                    },
                ),
                "injection_strength": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.01,
                        "tooltip": "Strength of injected noise",
                    },
                ),
                "normalize_injected_noise": (
                    ["enable", "disable"],
                    {
                        "default": "enable",
                        "tooltip": "Normalize injected noise to match latent statistics",
                    },
                ),
                "enhancement_mix": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Mix between original (0.0) and enhanced (1.0) face. 0.5 = 50/50 blend",
                    },
                ),
                "color_match_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength of color matching between original and enhanced face. 0.0 = disabled, 1.0 = full matching",
                    },
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "enhanced_image",
        "enhanced_face",
        "cropped_face_before",
        "enhanced_face_alpha",
        "base_face_alpha",
    )
    FUNCTION = "execute"
    CATEGORY = "CRT/Sampling"

    def execute(
        self,
        image,
        segs,
        model,
        positive,
        cfg,
        edit_model_flux2klein,
        upscale_megapixel,
        resize_back_to_original,
        multi_face_resolution_strategy,
        pre_crop_factor,
        post_mask_expand,
        post_mask_blur,
        post_mask_taper_borders,
        sampler_name,
        steps,
        denoise,
        scheduler,
        stage1_sigma_factor,
        stage1_sigma_start,
        stage2_sigma_factor,
        stage2_sigma_end,
        details_amount_stage1,
        details_amount_stage2,
        seed,
        enhancement_mix,
        enable_noise_injection,
        injection_point,
        injection_strength,
        normalize_injected_noise,
        color_match_strength,
        negative=None,
        vae=None,
        stage1_sigma_end=1.0,
        stage2_sigma_start=0.0,
        detail_daemon_enabled=True,
        dd_start=0.0,
        dd_end=1.0,
        dd_bias=0.5,
        dd_exponent=1.0,
        dd_start_offset=0.0,
        dd_end_offset=0.0,
        dd_fade_stage1=0.0,
        dd_fade_stage2=0.5,
        dd_smooth=True,
        dd_cfg_scale_override=1.0,
    ):

        def to_2d_mask(mask_data):
            if isinstance(mask_data, torch.Tensor):
                mask_tensor = mask_data.detach().float().cpu()
            else:
                mask_tensor = torch.from_numpy(np.array(mask_data)).float()

            while mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze(0)

            if mask_tensor.numel() == 0:
                return None

            if mask_tensor.max().item() > 1.0:
                mask_tensor = mask_tensor / 255.0

            return torch.clamp(mask_tensor, 0.0, 1.0)

        def scale_region(region, sx, sy, max_w, max_h):
            x1, y1, x2, y2 = [int(v) for v in region]
            x1 = max(0, min(int(round(x1 * sx)), max_w - 1))
            y1 = max(0, min(int(round(y1 * sy)), max_h - 1))
            x2 = max(x1 + 1, min(int(round(x2 * sx)), max_w))
            y2 = max(y1 + 1, min(int(round(y2 * sy)), max_h))
            return [x1, y1, x2, y2]

        def pack_preview(images, fallback):
            if not images:
                return fallback
            h0, w0 = images[0].shape[1:3]
            if all(img.shape[1] == h0 and img.shape[2] == w0 for img in images):
                return torch.cat(images, dim=0)
            return images[0]

        def attach_reference_latent(conditioning, ref_samples):
            if conditioning is None or isinstance(conditioning, str):
                return conditioning
            return node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": [ref_samples]},
                append=False,
            )

        def zero_conditioning_like(conditioning):
            if conditioning is None or isinstance(conditioning, str):
                return conditioning
            out = []
            for t in conditioning:
                d = t[1].copy()
                pooled_output = d.get("pooled_output", None)
                if pooled_output is not None:
                    d["pooled_output"] = torch.zeros_like(pooled_output)
                conditioning_lyrics = d.get("conditioning_lyrics", None)
                if conditioning_lyrics is not None:
                    d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
                out.append([torch.zeros_like(t[0]), d])
            return out

        def run_custom_sample(
            latent,
            sampler_obj,
            sigmas,
            cond_pos,
            cond_neg,
            seed_value,
            disable_noise,
        ):
            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(
                model,
                latent_image,
                latent.get("downscale_ratio_spacial", None),
            )

            if disable_noise:
                noise = torch.zeros(
                    latent_image.size(),
                    dtype=latent_image.dtype,
                    layout=latent_image.layout,
                    device="cpu",
                )
            else:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = comfy.sample.prepare_noise(latent_image, seed_value, batch_inds)

            noise_mask = latent.get("noise_mask", None)
            callback = latent_preview.prepare_callback(
                model, max(0, sigmas.shape[-1] - 1)
            )
            sampled = comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler_obj,
                sigmas,
                cond_pos,
                cond_neg,
                latent_image,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=seed_value,
            )
            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out["samples"] = sampled
            return out

        def quantize_to_multiple(value, multiple=1):
            q = int(round(float(value) / float(multiple))) * multiple
            return max(multiple, q)

        def quantize_dim_for_canvas(value, canvas_dim, multiple=1):
            if canvas_dim < multiple:
                return max(1, canvas_dim)
            max_multiple = (canvas_dim // multiple) * multiple
            q = (int(value) // multiple) * multiple
            if q < multiple:
                q = multiple
            return min(q, max_multiple)

        def quantized_size_preserve_aspect(
            src_w, src_h, scale=1.0, multiple=1, min_dim=2
        ):
            src_w = max(1.0, float(src_w))
            src_h = max(1.0, float(src_h))
            scaled_w = src_w * float(scale)
            out_w = max(min_dim, quantize_to_multiple(scaled_w, multiple))

            uniform_scale = out_w / src_w
            out_h = max(min_dim, int(round(src_h * uniform_scale)))
            if multiple > 1 and (out_h % multiple) != 0:
                out_h += multiple - (out_h % multiple)

            return int(out_w), int(out_h)

        def quantize_region_centered(x1, y1, x2, y2, max_w, max_h, multiple=2):
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            qw = max(2, quantize_to_multiple(w, multiple))
            qh = max(2, quantize_to_multiple(h, multiple))

            nx1 = int(round(cx - qw / 2.0))
            ny1 = int(round(cy - qh / 2.0))
            nx2 = nx1 + qw
            ny2 = ny1 + qh

            if nx1 < 0:
                nx2 -= nx1
                nx1 = 0
            if ny1 < 0:
                ny2 -= ny1
                ny1 = 0
            if nx2 > max_w:
                shift = nx2 - max_w
                nx1 = max(0, nx1 - shift)
                nx2 = max_w
            if ny2 > max_h:
                shift = ny2 - max_h
                ny1 = max(0, ny1 - shift)
                ny2 = max_h

            return nx1, ny1, nx2, ny2

        colored_print(
            "\n🐎 Starting Face Enhancement with Injection SEGS...", Colors.HEADER
        )

        seed_shift = 0
        injection_seed_offset = 1
        detail_daemon_enabled = True

        if negative is None:
            negative = zero_conditioning_like(positive)

        if vae is None:
            raise ValueError("SEGS Enhancer (Multi): VAE input is required.")

        actual_seed = seed
        colored_print("🎲 Seed Configuration:", Colors.HEADER)
        colored_print(f"   Base Seed: {seed}", Colors.BLUE)
        colored_print(f"   Final Seed: {actual_seed}", Colors.GREEN)

        orig_h, orig_w = image.shape[1:3]
        colored_print(f"📐 Original image dimensions: {orig_w}x{orig_h}", Colors.BLUE)

        seg_entries = segs[1] if isinstance(segs, tuple) and len(segs) > 1 else []
        if not seg_entries:
            colored_print("❌ No SEGS provided! Returning original image.", Colors.RED)
            fallback = torch.zeros_like(image)
            return (image, fallback, fallback, fallback, fallback)

        target_area = max(0.1, float(upscale_megapixel)) * 1_000_000.0
        valid_segments = []
        scales = []

        for seg in seg_entries:
            x1, y1, x2, y2 = [int(v) for v in seg.crop_region]
            x1, y1, x2, y2 = quantize_region_centered(
                x1, y1, x2, y2, orig_w, orig_h, multiple=2
            )

            seg_w = max(1, x2 - x1)
            seg_h = max(1, y2 - y1)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            target_w, target_h = quantized_size_preserve_aspect(
                seg_w,
                seg_h,
                scale=pre_crop_factor,
                multiple=1,
                min_dim=2,
            )

            px1 = int(round(center_x - target_w / 2.0))
            py1 = int(round(center_y - target_h / 2.0))
            px2 = px1 + target_w
            py2 = py1 + target_h

            if px1 < 0:
                px2 -= px1
                px1 = 0
            if py1 < 0:
                py2 -= py1
                py1 = 0
            if px2 > orig_w:
                shift = px2 - orig_w
                px1 = max(0, px1 - shift)
                px2 = orig_w
            if py2 > orig_h:
                shift = py2 - orig_h
                py1 = max(0, py1 - shift)
                py2 = orig_h
            if px2 <= px1 or py2 <= py1:
                continue

            crop_w = px2 - px1
            crop_h = py2 - py1
            crop_area = max(1.0, float(crop_w * crop_h))
            crop_scale = max(1.0, math.sqrt(target_area / crop_area))

            valid_segments.append((seg, [px1, py1, px2, py2], [x1, y1, x2, y2]))
            scales.append(crop_scale)

        if not valid_segments:
            colored_print("❌ No valid SEGS crop regions available.", Colors.RED)
            fallback = torch.zeros_like(image)
            return (image, fallback, fallback, fallback, fallback)

        if resize_back_to_original:
            global_scale = 1.0
            colored_print(
                "📈 Using original canvas (resize_back_to_original=True)", Colors.BLUE
            )
        else:
            min_scale = min(scales)
            max_scale = max(scales)
            if multi_face_resolution_strategy == "fit to smallest":
                global_scale = max_scale
            elif multi_face_resolution_strategy == "fit to largest":
                global_scale = min_scale
            else:
                global_scale = (min_scale + max_scale) / 2.0

            colored_print("📈 Multi-face upscaling strategy:", Colors.HEADER)
            colored_print(f"   Strategy: {multi_face_resolution_strategy}", Colors.BLUE)
            colored_print(
                f"   Scale range: {min_scale:.3f}x - {max_scale:.3f}x", Colors.BLUE
            )
            colored_print(f"   Applied global scale: {global_scale:.3f}x", Colors.GREEN)

        canvas_w = max(orig_w, int(round(orig_w * global_scale)))
        canvas_h = max(orig_h, int(round(canvas_w * orig_h / max(1, orig_w))))
        resized_image = ImageResize().execute(
            image, canvas_w, canvas_h, method="stretch"
        )
        scale_x = canvas_w / max(1, orig_w)
        scale_y = canvas_h / max(1, orig_h)

        colored_print(f"🖼️ Working canvas: {canvas_w}x{canvas_h}", Colors.GREEN)
        colored_print("🎨 Starting SEGS enhancement phase...", Colors.HEADER)
        colored_print("   Enhancement Configuration:", Colors.BLUE)
        colored_print(
            f"     Edit model (flux2klein): {'on' if edit_model_flux2klein else 'off'}",
            Colors.BLUE,
        )
        colored_print(
            f"     Target Upscale: {upscale_megapixel:.2f} MP/crop", Colors.BLUE
        )
        colored_print(f"     Pre-crop Factor: {pre_crop_factor:.2f}", Colors.BLUE)
        colored_print(f"     Enhancement Mix: {enhancement_mix:.3f}", Colors.BLUE)
        colored_print(
            f"     Sampling Steps: {steps} | CFG: {cfg:.1f} | Denoise: {denoise:.2f}",
            Colors.BLUE,
        )
        colored_print(
            f"     Sampler: {sampler_name} | Scheduler: {scheduler}", Colors.BLUE
        )

        final_pil = tensor2pil(resized_image[0])
        enhanced_faces_for_output = []
        cropped_faces_for_output = []
        enhanced_alpha_for_output = []
        base_alpha_for_output = []

        base_sampler = comfy.samplers.sampler_object(sampler_name)
        if detail_daemon_enabled:
            sampler_stage1 = build_dd_sampler(
                base_sampler,
                details_amount_stage1,
                dd_start,
                dd_end,
                dd_bias,
                dd_exponent,
                dd_start_offset,
                dd_end_offset,
                dd_fade_stage1,
                dd_smooth,
                dd_cfg_scale_override,
            )
            sampler_stage2 = build_dd_sampler(
                base_sampler,
                details_amount_stage2,
                dd_start,
                dd_end,
                dd_bias,
                dd_exponent,
                dd_start_offset,
                dd_end_offset,
                dd_fade_stage2,
                dd_smooth,
                dd_cfg_scale_override,
            )
        else:
            sampler_stage1 = base_sampler
            sampler_stage2 = base_sampler

        ksampler_plan = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sigmas_full = ksampler_plan.sigmas.clone()

        split_ratio = injection_point

        stage1_sigmas, stage2_sigmas = split_sigmas_denoise_stateless(
            sigmas_full, split_ratio
        )
        stage1_sigmas = multiply_sigmas_stateless(
            stage1_sigmas,
            stage1_sigma_factor,
            stage1_sigma_start,
            stage1_sigma_end,
        )
        stage2_sigmas = multiply_sigmas_stateless(
            stage2_sigmas,
            stage2_sigma_factor,
            stage2_sigma_start,
            stage2_sigma_end,
        )

        if enable_noise_injection == "enable":
            colored_print("💉 Noise Injection Configuration:", Colors.HEADER)
            colored_print(
                f"   🎚️ Split Sigmas Denoise: {split_ratio:.2f} | Stage1 sigmas: {len(stage1_sigmas)} | Stage2 sigmas: {len(stage2_sigmas)}",
                Colors.CYAN,
            )
            colored_print(
                f"   🎲 Injection Seed: {actual_seed + injection_seed_offset} (offset: {injection_seed_offset:+d})",
                Colors.CYAN,
            )
            colored_print(
                f"   💪 Injection Strength: {injection_strength:.3f}", Colors.CYAN
            )
            colored_print(
                f"   📏 Normalize Noise: {normalize_injected_noise}", Colors.CYAN
            )
        for i, (seg, padded_region, original_region) in enumerate(valid_segments):
            seed_for_seg = actual_seed + i

            scaled_region = scale_region(
                padded_region, scale_x, scale_y, canvas_w, canvas_h
            )
            x1, y1, x2, y2 = scaled_region
            crop_w = x2 - x1
            crop_h = y2 - y1

            crop_w_q = quantize_dim_for_canvas(crop_w, canvas_w, 1)
            crop_h_q = quantize_dim_for_canvas(crop_h, canvas_h, 1)
            if crop_w_q != crop_w or crop_h_q != crop_h:
                x2 = x1 + crop_w_q
                y2 = y1 + crop_h_q
                if x2 > canvas_w:
                    x1 = max(0, canvas_w - crop_w_q)
                    x2 = canvas_w
                if y2 > canvas_h:
                    y1 = max(0, canvas_h - crop_h_q)
                    y2 = canvas_h
                crop_w = x2 - x1
                crop_h = y2 - y1

            if crop_w <= 1 or crop_h <= 1:
                colored_print(
                    f"   Segment {i + 1}: invalid scaled region, skipping",
                    Colors.YELLOW,
                )
                continue

            cropped_face_image = resized_image[:, y1:y2, x1:x2, :]
            if cropped_face_image.numel() == 0:
                colored_print(
                    f"   Segment {i + 1}: empty crop, skipping", Colors.YELLOW
                )
                continue

            ox1, oy1, ox2, oy2 = [int(v) for v in original_region]
            padded_x1, padded_y1, padded_x2, padded_y2 = [int(v) for v in padded_region]
            base_w = max(1, ox2 - ox1)
            base_h = max(1, oy2 - oy1)
            padded_w = max(1, padded_x2 - padded_x1)
            padded_h = max(1, padded_y2 - padded_y1)

            raw_mask = to_2d_mask(seg.cropped_mask)
            if raw_mask is None:
                raw_mask = torch.ones((base_h, base_w), dtype=torch.float32)

            raw_mask = (
                F_interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(base_h, base_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

            padded_mask = torch.zeros((padded_h, padded_w), dtype=torch.float32)
            overlap_x1 = max(ox1, padded_x1)
            overlap_y1 = max(oy1, padded_y1)
            overlap_x2 = min(ox2, padded_x2)
            overlap_y2 = min(oy2, padded_y2)

            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                src_x1 = overlap_x1 - ox1
                src_y1 = overlap_y1 - oy1
                src_x2 = src_x1 + (overlap_x2 - overlap_x1)
                src_y2 = src_y1 + (overlap_y2 - overlap_y1)

                dst_x1 = overlap_x1 - padded_x1
                dst_y1 = overlap_y1 - padded_y1
                dst_x2 = dst_x1 + (overlap_x2 - overlap_x1)
                dst_y2 = dst_y1 + (overlap_y2 - overlap_y1)

                padded_mask[dst_y1:dst_y2, dst_x1:dst_x2] = raw_mask[
                    src_y1:src_y2, src_x1:src_x2
                ]

            mask_canvas = (
                F_interpolate(
                    padded_mask.unsqueeze(0).unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

            feathered_mask, _ = GrowMaskWithBlur().expand_mask(
                mask_canvas.unsqueeze(0),
                expand=post_mask_expand,
                tapered_corners=True,
                blur_radius=post_mask_blur,
                taper_radius=post_mask_taper_borders,
            )

            local_area = max(1.0, float(crop_w * crop_h))
            local_scale = max(1.0, math.sqrt(target_area / local_area))
            target_w, target_h = quantized_size_preserve_aspect(
                crop_w,
                crop_h,
                scale=local_scale,
                multiple=16,
                min_dim=64,
            )
            upscaled_face = ImageResize().execute(
                cropped_face_image, target_w, target_h, method="stretch"
            )

            colored_print(
                f"   Segment {i + 1}: crop {crop_w}x{crop_h} -> process {upscaled_face.shape[2]}x{upscaled_face.shape[1]}",
                Colors.BLUE,
            )

            face_latent = {"samples": vae.encode(upscaled_face)}
            if edit_model_flux2klein:
                seg_positive = attach_reference_latent(positive, face_latent["samples"])
                seg_negative = attach_reference_latent(
                    negative,
                    torch.zeros_like(face_latent["samples"]),
                )
            else:
                seg_positive = positive
                seg_negative = negative

            if enable_noise_injection == "enable":
                stage1_latent = run_custom_sample(
                    face_latent,
                    sampler_stage1,
                    stage1_sigmas,
                    seg_positive,
                    seg_negative,
                    seed_for_seg,
                    disable_noise=False,
                )

                actual_injection_seed = seed_for_seg + injection_seed_offset
                injected_latent_samples = stage1_latent["samples"].clone()
                torch.manual_seed(actual_injection_seed)
                new_noise = torch.randn_like(injected_latent_samples)

                if normalize_injected_noise == "enable":
                    original_mean = injected_latent_samples.mean().item()
                    original_std = injected_latent_samples.std().item()
                    if original_std > 1e-6:
                        new_noise = new_noise * original_std + original_mean

                injected_latent_samples += new_noise * injection_strength
                injected_latent = stage1_latent.copy()
                injected_latent["samples"] = injected_latent_samples

                enhanced_latent = run_custom_sample(
                    injected_latent,
                    sampler_stage2,
                    stage2_sigmas,
                    seg_positive,
                    seg_negative,
                    seed_for_seg,
                    disable_noise=True,
                )
            else:
                full_sigmas = multiply_sigmas_stateless(
                    sigmas_full,
                    stage1_sigma_factor,
                    stage1_sigma_start,
                    stage1_sigma_end,
                )
                enhanced_latent = run_custom_sample(
                    face_latent,
                    sampler_stage1,
                    full_sigmas,
                    seg_positive,
                    seg_negative,
                    seed_for_seg,
                    disable_noise=False,
                )

            enhanced_face_image = vae.decode(enhanced_latent["samples"])

            if (
                enhanced_face_image.shape[1] != upscaled_face.shape[1]
                or enhanced_face_image.shape[2] != upscaled_face.shape[2]
            ):
                upscaled_face = ImageResize().execute(
                    upscaled_face,
                    enhanced_face_image.shape[2],
                    enhanced_face_image.shape[1],
                    method="stretch",
                    interpolation="lanczos",
                )

            if enhancement_mix < 1.0:
                enhanced_face_image = (
                    upscaled_face * (1.0 - enhancement_mix)
                    + enhanced_face_image * enhancement_mix
                )

            if color_match_strength > 0:
                color_matched_face = ColorMatch().colormatch(
                    upscaled_face,
                    enhanced_face_image,
                    method="mkl",
                    strength=color_match_strength,
                )[0]
            else:
                color_matched_face = enhanced_face_image

            enhanced_faces_for_output.append(color_matched_face.cpu())
            cropped_faces_for_output.append(upscaled_face.cpu())

            alpha_mask_enhanced = F_interpolate(
                feathered_mask.unsqueeze(1),
                size=(color_matched_face.shape[1], color_matched_face.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            alpha_mask_enhanced = torch.clamp(alpha_mask_enhanced, 0.0, 1.0).movedim(
                1, -1
            )

            alpha_mask_base = F_interpolate(
                feathered_mask.unsqueeze(1),
                size=(upscaled_face.shape[1], upscaled_face.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            alpha_mask_base = torch.clamp(alpha_mask_base, 0.0, 1.0).movedim(1, -1)

            enhanced_alpha_for_output.append((color_matched_face * alpha_mask_enhanced).cpu())
            base_alpha_for_output.append((upscaled_face * alpha_mask_base).cpu())

            enhanced_pil = tensor2pil(color_matched_face[0])
            paste_mask_pil = to_pil_image(feathered_mask.squeeze()).convert("L")

            enhanced_pil_resized = enhanced_pil.resize(
                (crop_w, crop_h), Image.Resampling.LANCZOS
            )
            paste_mask_pil_resized = paste_mask_pil.resize(
                (crop_w, crop_h), Image.Resampling.LANCZOS
            )
            final_pil.paste(enhanced_pil_resized, (x1, y1), paste_mask_pil_resized)

            del face_latent, enhanced_latent, enhanced_face_image
            if enable_noise_injection == "enable":
                del stage1_latent, injected_latent, injected_latent_samples, new_noise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        colored_print("🖼️ Compositing final image...", Colors.HEADER)
        final_image_tensor = pil2tensor(final_pil)
        colored_print("📏 Final image processing...", Colors.HEADER)
        if resize_back_to_original:
            final_image = ImageResize().execute(
                final_image_tensor, orig_w, orig_h, method="stretch"
            )
        else:
            final_image = final_image_tensor
        final_h, final_w = final_image.shape[1:3]

        preview_fallback = torch.zeros_like(image)
        enhanced_preview = pack_preview(enhanced_faces_for_output, preview_fallback)
        cropped_preview = pack_preview(cropped_faces_for_output, preview_fallback)
        enhanced_alpha_preview = pack_preview(
            enhanced_alpha_for_output, preview_fallback
        )
        base_alpha_preview = pack_preview(base_alpha_for_output, preview_fallback)

        colored_print(
            "\n✅ Face Enhancement with Injection SEGS completed successfully!",
            Colors.HEADER,
        )
        colored_print(
            f"   🧩 Segments processed: {len(enhanced_faces_for_output)}", Colors.GREEN
        )
        colored_print(f"   📐 Output size: {final_w}x{final_h}", Colors.GREEN)

        return (
            final_image,
            enhanced_preview,
            cropped_preview,
            enhanced_alpha_preview,
            base_alpha_preview,
        )


NODE_CLASS_MAPPINGS = {
    "FaceEnhancementWithInjectionSEGS": FaceEnhancementWithInjectionSEGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEnhancementWithInjectionSEGS": "SEGS Enhancer Multi (CRT)"
}

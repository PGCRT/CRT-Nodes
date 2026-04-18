import torch
import torch.nn.functional as F
import numpy as np
import os
import folder_paths
import comfy.utils
import comfy.sd
import comfy.sample
import comfy.controlnet
import comfy.samplers
import scipy.ndimage
import cv2
import inspect

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate as F_interpolate
from nodes import common_ksampler
from .SEGS_Enhancer_Multi import FaceEnhancementWithInjectionSEGS


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
    print(f"{color}{message}{Colors.ENDC}")


def load_yolo(model_path: str):
    colored_print(f"🔄 Loading YOLO model from: {model_path}", Colors.CYAN)
    from ultralytics import YOLO

    # Fixed: Compatibility layer for ultralytics 8.x
    try:
        # Try modern ultralytics approach first
        model = YOLO(model_path)
        colored_print("✅ YOLO model loaded successfully!", Colors.GREEN)
    except Exception as e:
        colored_print(
            f"⚠️ Modern YOLO loading failed, trying legacy approach: {e}", Colors.YELLOW
        )
        # Fallback for older ultralytics versions if nn.tasks exists
        try:
            import ultralytics.nn.tasks as nn_tasks

            original_torch_safe_load = nn_tasks.torch_safe_load

            def unsafe_pt_loader(weight, map_location="cpu"):
                ckpt = torch.load(weight, map_location=map_location, weights_only=False)
                return ckpt, weight

            try:
                nn_tasks.torch_safe_load = unsafe_pt_loader
                model = YOLO(model_path)
                colored_print(
                    "✅ YOLO model loaded successfully with legacy method!",
                    Colors.GREEN,
                )
            finally:
                nn_tasks.torch_safe_load = original_torch_safe_load
        except ImportError:
            # Final fallback - direct loading
            model = YOLO(model_path)
            colored_print("✅ YOLO model loaded with direct method!", Colors.GREEN)
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


def apply_controlnet_advanced(
    positive, negative, control_net, image, strength, start_percent, end_percent, vae
):
    colored_print(
        f"🎮 Applying ControlNet (strength: {strength:.3f}, range: {start_percent:.3f}-{end_percent:.3f})...",
        Colors.CYAN,
    )

    if strength == 0:
        colored_print("   🚫 ControlNet strength is 0, skipping", Colors.YELLOW)
        return (positive, negative)

    control_hint = image.movedim(-1, 1)
    cnets, out = {}, []

    conditioning_count = 0
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            conditioning_count += 1
            d = t[1].copy()
            prev_cnet = d.get("control", None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
                colored_print("   🔗 Reusing ControlNet instance", Colors.BLUE)
            else:
                c_net = control_net.copy().set_cond_hint(
                    control_hint, strength, (start_percent, end_percent), vae=vae
                )
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net
                colored_print("   🆕 Created new ControlNet instance", Colors.BLUE)
            d["control"], d["control_apply_to_uncond"] = c_net, False
            c.append([t[0], d])
        out.append(c)

    colored_print(
        f"✅ ControlNet applied to {conditioning_count} conditioning(s)!", Colors.GREEN
    )
    return (out[0], out[1])


class _SimpleSeg:
    def __init__(self, crop_region, cropped_mask):
        self.crop_region = crop_region
        self.cropped_mask = cropped_mask


def _mask_tensor_to_2d(mask, target_h, target_w):
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.array(mask))

    m = mask.detach().float().cpu()

    if m.dim() == 4 and m.shape[-1] == 1:
        m = m.squeeze(-1)
    if m.dim() == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        return None

    if m.shape[0] != target_h or m.shape[1] != target_w:
        m = (
            F_interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    return torch.clamp(m, 0.0, 1.0).numpy().astype(np.float32)


def _mask_to_seg_entries(mask_2d, grow_crop):
    if mask_2d is None:
        return []

    h, w = mask_2d.shape
    binary = (mask_2d > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours_info = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _img, contours, _hier = contours_info
    else:
        contours, _hier = contours_info

    seg_entries = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 1 or bh <= 1:
            continue

        x1 = x - grow_crop
        y1 = y - grow_crop
        x2 = x + bw + grow_crop
        y2 = y + bh + grow_crop

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        contour_mask = np.zeros((h, w), dtype=np.float32)
        cv2.drawContours(contour_mask, [contour], -1, 1.0, thickness=-1)
        cropped_mask = contour_mask[y1:y2, x1:x2]
        if cropped_mask.size == 0 or float(cropped_mask.max()) <= 0.0:
            continue

        seg_entries.append(_SimpleSeg([x1, y1, x2, y2], cropped_mask))

    return seg_entries


class FaceEnhancementPipelineWithInjection:
    def __init__(self):
        self.detectors = {}
        self.segs_enhancer = FaceEnhancementWithInjectionSEGS()
        self._segs_execute_sig = inspect.signature(self.segs_enhancer.execute)
        colored_print("Ultralytics Enhancer initialized!", Colors.HEADER)

    @classmethod
    def INPUT_TYPES(s):
        try:
            segm_files = folder_paths.get_filename_list("ultralytics_segm")
        except Exception:
            segm_files = []

        base = FaceEnhancementWithInjectionSEGS.INPUT_TYPES()
        required = {}
        for k, v in base["required"].items():
            if k == "segs":
                required["face_segm_model"] = (["segm/" + x for x in segm_files],)
                required["segm_threshold"] = (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                )
                continue
            if k == "edit_model_flux2klein":
                continue
            if k == "positive":
                required[k] = v
                required["control_net"] = ("CONTROL_NET",)
                continue
            if k == "seed":
                required[k] = v
                required["controlnet_strength"] = (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                )
                required["control_end"] = (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                )
                continue
            required[k] = v

        return {
            "required": required,
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

    def execute(self, **kwargs):
        image = kwargs["image"]
        face_segm_model = kwargs["face_segm_model"]
        segm_threshold = kwargs["segm_threshold"]
        control_net = kwargs["control_net"]
        controlnet_strength = kwargs["controlnet_strength"]
        control_end = kwargs["control_end"]

        positive = kwargs["positive"]
        negative = kwargs.get("negative", None)
        if negative is None:
            negative = []
            for t in positive:
                negative.append([torch.zeros_like(t[0]), t[1].copy()])

        vae = kwargs.get("vae", None)
        cnet_positive, cnet_negative = apply_controlnet_advanced(
            positive,
            negative,
            control_net,
            image,
            controlnet_strength,
            0.0,
            control_end,
            vae,
        )

        segm_filename_only = face_segm_model.split("/")[-1]
        segm_full_path = folder_paths.get_full_path(
            "ultralytics_segm", segm_filename_only
        )
        if face_segm_model not in self.detectors:
            self.detectors[face_segm_model] = UltraDetector(segm_full_path, "segm")
        segm_detector = self.detectors[face_segm_model]

        pil_image = tensor2pil(image[0])
        detected_results = inference_segm(
            segm_detector.model, pil_image, segm_threshold
        )
        segmasks = create_segmasks(detected_results)

        h, w = image.shape[1:3]
        seg_entries = []
        for bbox, mask, _confidence in segmasks:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_mask = mask[y1:y2, x1:x2]
            if cropped_mask.size == 0:
                continue
            seg_entries.append(_SimpleSeg([x1, y1, x2, y2], cropped_mask))

        segs = (None, seg_entries)

        call_kwargs = {}
        for name in self._segs_execute_sig.parameters:
            if name == "self":
                continue
            if name == "segs":
                call_kwargs[name] = segs
            elif name == "edit_model_flux2klein":
                call_kwargs[name] = False
            elif name == "positive":
                call_kwargs[name] = cnet_positive
            elif name == "negative":
                call_kwargs[name] = cnet_negative
            elif name in kwargs:
                call_kwargs[name] = kwargs[name]

        return self.segs_enhancer.execute(**call_kwargs)


class UltralyticsEnhancer:
    def __init__(self):
        self.detectors = {}
        self.segs_enhancer = FaceEnhancementWithInjectionSEGS()
        self._segs_execute_sig = inspect.signature(self.segs_enhancer.execute)
        colored_print("Ultralytics Enhancer initialized!", Colors.HEADER)

    @classmethod
    def INPUT_TYPES(s):
        try:
            segm_files = folder_paths.get_filename_list("ultralytics_segm")
        except Exception:
            segm_files = []

        base = FaceEnhancementWithInjectionSEGS.INPUT_TYPES()
        required = {}
        for k, v in base["required"].items():
            if k == "segs":
                required["face_segm_model"] = (["segm/" + x for x in segm_files],)
                required["segm_threshold"] = (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                )
                required["grow_crop"] = (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Expands each detected crop region by this many pixels before enhancement",
                    },
                )
                continue
            required[k] = v

        return {
            "required": required,
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

    def execute(self, **kwargs):
        image = kwargs["image"]
        face_segm_model = kwargs["face_segm_model"]
        segm_threshold = kwargs["segm_threshold"]
        grow_crop = int(kwargs.get("grow_crop", 0))
        h, w = image.shape[1:3]
        seg_entries = []

        segm_filename_only = face_segm_model.split("/")[-1]
        segm_full_path = folder_paths.get_full_path(
            "ultralytics_segm", segm_filename_only
        )
        if face_segm_model not in self.detectors:
            self.detectors[face_segm_model] = UltraDetector(segm_full_path, "segm")
        segm_detector = self.detectors[face_segm_model]

        pil_image = tensor2pil(image[0])
        detected_results = inference_segm(
            segm_detector.model, pil_image, segm_threshold
        )
        segmasks = create_segmasks(detected_results)

        for bbox, mask, _confidence in segmasks:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            if grow_crop > 0:
                x1 -= grow_crop
                y1 -= grow_crop
                x2 += grow_crop
                y2 += grow_crop
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_mask = mask[y1:y2, x1:x2]
            if cropped_mask.size == 0:
                continue
            seg_entries.append(_SimpleSeg([x1, y1, x2, y2], cropped_mask))

        segs = (None, seg_entries)

        call_kwargs = {}
        for name in self._segs_execute_sig.parameters:
            if name == "self":
                continue
            if name == "segs":
                call_kwargs[name] = segs
            elif name in kwargs:
                call_kwargs[name] = kwargs[name]

        return self.segs_enhancer.execute(**call_kwargs)


NODE_CLASS_MAPPINGS = {
    "FaceEnhancementPipelineWithInjection": FaceEnhancementPipelineWithInjection,
    "UltralyticsEnhancer": UltralyticsEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEnhancementPipelineWithInjection": "Flux.1 Cnet Ultralytics Enhancer (CRT)",
    "UltralyticsEnhancer": "Ultralytics Enhancer (CRT)",
}

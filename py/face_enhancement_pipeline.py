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

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate as F_interpolate
from nodes import common_ksampler

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
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def combine_masks(segmasks):
    if not segmasks: return None
    combined_mask = np.zeros_like(segmasks[0][1], dtype=np.float32)
    for _, mask, _ in segmasks: combined_mask += mask
    return torch.from_numpy(np.clip(combined_mask, 0, 1))

def dilate_masks(segmasks, dilation_factor):
    if dilation_factor == 0: return segmasks
    dilated_segmasks = []
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    colored_print(f"   🔧 Applying dilation with {dilation_factor}x{dilation_factor} kernel", Colors.BLUE)
    for bbox, mask, confidence in segmasks:
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_segmasks.append((bbox, dilated_mask, confidence))
    return dilated_segmasks

def create_segmasks(results):
    if not results or not results[1]: return []
    bboxs, segms, confidence = results[1], results[2], results[3]
    output = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        output.append(item)
    colored_print(f"   📊 Created {len(output)} segmentation mask(s)", Colors.BLUE)
    return output

def inference_bbox(model, image: Image.Image, confidence: float = 0.3):
    colored_print(f"   🔍 Running bbox detection (confidence: {confidence:.2f})", Colors.BLUE)
    pred = model(image, conf=confidence)
    if not pred or not hasattr(pred[0], 'boxes') or pred[0].boxes is None or pred[0].boxes.xyxy.nelement() == 0: 
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
    colored_print(f"   🔍 Running segmentation detection (confidence: {confidence:.2f})", Colors.BLUE)
    pred = model(image, conf=confidence)
    if not pred or not hasattr(pred[0], 'masks') or pred[0].masks is None or pred[0].masks.data.nelement() == 0: 
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
        scaled_mask = F_interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False).squeeze()
        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())
    return results

class UltraDetector:
    def __init__(self, model_path, detection_type):
        colored_print(f"🤖 Initializing {detection_type.upper()} detector...", Colors.CYAN)
        self.model = load_yolo(model_path)
        self.type = detection_type
        colored_print(f"✅ {detection_type.upper()} detector ready!", Colors.GREEN)
        
    def detect_combined(self, image, threshold, dilation):
        colored_print(f"🔍 Running {self.type.upper()} detection...", Colors.CYAN)
        pil_image = tensor2pil(image)
        colored_print(f"   📐 Detection image size: {pil_image.size}", Colors.BLUE)
        
        if self.type == "bbox": detected_results = inference_bbox(self.model, pil_image, threshold)
        else: detected_results = inference_segm(self.model, pil_image, threshold)
        
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
        colored_print(f"   📊 Final mask coverage: {mask_coverage*100:.2f}%", Colors.GREEN)
        
        return final_mask

class ImageResize:
    def execute(self, image, width, height, method="stretch", interpolation="lanczos"):
        _, oh, ow, _ = image.shape
        colored_print(f"🔄 Resizing image from {ow}x{oh}...", Colors.CYAN)
        
        if method == 'keep proportion':
            if ow > oh: ratio = width / ow
            else: ratio = height / oh
            width, height = round(ow * ratio), round(oh * ratio)
            colored_print(f"   📐 Keeping proportions - Target: {width}x{height} (ratio: {ratio:.3f})", Colors.BLUE)
        else:
            width, height = (width if width > 0 else ow), (height if height > 0 else oh)
            colored_print(f"   📐 Direct resize to: {width}x{height}", Colors.BLUE)
        
        outputs = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos": 
            colored_print(f"   🎯 Using Lanczos interpolation", Colors.BLUE)
            outputs = comfy.utils.lanczos(outputs, width, height)
        else: 
            colored_print(f"   🎯 Using {interpolation} interpolation", Colors.BLUE)
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)
        
        colored_print(f"✅ Image resize completed: {width}x{height}", Colors.GREEN)
        return torch.clamp(outputs.permute(0, 2, 3, 1), 0, 1)

class ColorMatch:
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        colored_print(f"🎨 Applying color matching (method: {method}, strength: {strength:.2f})...", Colors.CYAN)
        
        try: from color_matcher import ColorMatcher
        except: 
            colored_print("❌ ERROR: 'color-matcher' library not found.", Colors.RED)
            raise Exception("ColorMatch requires 'color-matcher'. Please 'pip install color-matcher'")
        
        cm, out = ColorMatcher(), []
        image_ref, image_target = image_ref.cpu(), image_target.cpu()
        
        batch_size = image_target.size(0)
        colored_print(f"   🔄 Processing {batch_size} image(s)...", Colors.BLUE)
        
        for i in range(batch_size):
            target_np, ref_np = image_target[i].numpy(), image_ref[i if image_ref.size(0) == image_target.size(0) else 0].numpy()
            target_mean = np.mean(target_np, axis=(0,1))
            ref_mean = np.mean(ref_np, axis=(0,1))
            
            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            result = target_np + strength * (result - target_np)
            
            final_mean = np.mean(result, axis=(0,1))
            colored_print(f"   📊 Image {i+1}: Target RGB({target_mean[0]:.3f},{target_mean[1]:.3f},{target_mean[2]:.3f}) → Final RGB({final_mean[0]:.3f},{final_mean[1]:.3f},{final_mean[2]:.3f})", Colors.BLUE)
            
            out.append(torch.from_numpy(result))
        
        colored_print("✅ Color matching completed!", Colors.GREEN)
        return (torch.stack(out, dim=0).to(torch.float32).clamp_(0, 1),)

class GrowMaskWithBlur:
    def expand_mask(self, mask, expand, tapered_corners, blur_radius):
        colored_print(f"🎭 Processing mask (expand: {expand}, blur: {blur_radius})...", Colors.CYAN)
        
        c, out = (0 if tapered_corners else 1), []
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
        
        for m in mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu():
            output = m.numpy().astype(np.float32)
            original_coverage = np.sum(output) / output.size
            
            if expand != 0:
                colored_print(f"   🔧 Applying {abs(expand)} iterations of {'dilation' if expand > 0 else 'erosion'}", Colors.BLUE)
                for _ in range(abs(round(expand))):
                    if expand < 0: output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                    else: output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            
            final_coverage = np.sum(output) / output.size
            colored_print(f"   📊 Mask coverage: {original_coverage*100:.2f}% → {final_coverage*100:.2f}%", Colors.BLUE)
            
            out.append(torch.from_numpy(output))
        
        final_mask = torch.stack(out, dim=0)
        
        if blur_radius > 0:
            colored_print(f"   🌫️ Applying Gaussian blur (radius: {blur_radius})", Colors.BLUE)
            pil_img = to_pil_image(final_mask)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
            final_mask = to_tensor(pil_img)
        
        colored_print("✅ Mask processing completed!", Colors.GREEN)
        return (final_mask, 1.0 - final_mask)

class BoundedImageCropWithMask:
    def crop(self, image, mask, top, bottom, left, right):
        colored_print(f"✂️ Cropping image with padding (t:{top}, b:{bottom}, l:{left}, r:{right})...", Colors.CYAN)
        
        image, mask = (image.unsqueeze(0) if image.dim() == 3 else image), (mask.unsqueeze(0) if mask.dim() == 2 else mask)
        cropped_images, all_bounds = [], []
        
        for i in range(len(image)):
            current_mask = mask[i if len(mask) == len(image) else 0]
            if current_mask.sum() == 0: 
                colored_print(f"   ⚠️ Empty mask for image {i+1}, skipping", Colors.YELLOW)
                continue
            
            rows, cols = torch.any(current_mask, dim=1), torch.any(current_mask, dim=0)
            
            if not torch.any(rows) or not torch.any(cols):
                colored_print(f"   ⚠️ No valid mask regions for image {i+1}, skipping", Colors.YELLOW)
                continue
                
            rmin_t, rmax_t = torch.where(rows)[0][[0, -1]]
            cmin_t, cmax_t = torch.where(cols)[0][[0, -1]]
            
            rmin = max(rmin_t.item() - top, 0)
            rmax = min(rmax_t.item() + bottom, current_mask.shape[0] - 1)
            cmin = max(cmin_t.item() - left, 0)
            cmax = min(cmax_t.item() + right, current_mask.shape[1] - 1)
            
            if rmax < rmin or cmax < cmin:
                colored_print(f"   ❌ Invalid crop region for image {i+1} (zero or negative size), skipping", Colors.RED)
                continue

            crop_w, crop_h = cmax - cmin + 1, rmax - rmin + 1
            colored_print(f"   📐 Crop {i+1}: {crop_w}x{crop_h} at ({cmin},{rmin})", Colors.BLUE)
            
            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin:rmax+1, cmin:cmax+1, :])
        
        if cropped_images:
            colored_print(f"✅ Successfully cropped {len(cropped_images)} region(s)!", Colors.GREEN)
            return (torch.stack(cropped_images), all_bounds)
        else:
            colored_print("❌ No valid crops produced", Colors.RED)
            return (None, None)

def apply_controlnet_advanced(positive, negative, control_net, image, strength, start_percent, end_percent, vae):
    colored_print(f"🎮 Applying ControlNet (strength: {strength:.3f}, range: {start_percent:.3f}-{end_percent:.3f})...", Colors.CYAN)
    
    if strength == 0: 
        colored_print("   🚫 ControlNet strength is 0, skipping", Colors.YELLOW)
        return (positive, negative)
    
    control_hint = image.movedim(-1,1)
    cnets, out = {}, []
    
    conditioning_count = 0
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            conditioning_count += 1
            d = t[1].copy()
            prev_cnet = d.get('control', None)
            if prev_cnet in cnets: 
                c_net = cnets[prev_cnet]
                colored_print(f"   🔗 Reusing ControlNet instance", Colors.BLUE)
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net
                colored_print(f"   🆕 Created new ControlNet instance", Colors.BLUE)
            d['control'], d['control_apply_to_uncond'] = c_net, False
            c.append([t[0], d])
        out.append(c)
    
    colored_print(f"✅ ControlNet applied to {conditioning_count} conditioning(s)!", Colors.GREEN)
    return (out[0], out[1])

class FaceEnhancementPipeline:
    def __init__(self):
        self.detectors = {}
        colored_print("🎭 Face Enhancement Pipeline initialized!", Colors.HEADER)

    @classmethod
    def INPUT_TYPES(s):
        try: bbox_files = folder_paths.get_filename_list("ultralytics_bbox")
        except: bbox_files = []
        try: segm_files = folder_paths.get_filename_list("ultralytics_segm")
        except: segm_files = []
        
        return {
            "required": {
                "image": ("IMAGE",), 
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
                "face_bbox_model": (["bbox/" + x for x in bbox_files] + ["segm/" + x for x in segm_files],),
                "face_segm_model": (["segm/" + x for x in segm_files],),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "segm_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "initial_upscale_resolution": ("INT", {"default": 2048, "min": 1024, "max": 4096, "step": 64, "tooltip": "Resolution to upscale the entire image to before processing"}),
                "upscale_resolution": ("INT", {"default": 1536, "min": 512, "max": 4096, "step": 64, "tooltip": "Resolution for face enhancement"}),
                
                "resize_back_to_original": ("BOOLEAN", {"default": True, "tooltip": "Whether to resize the final result back to original image dimensions"}),
                
                "padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "mask_expand": ("INT", {"default": 10, "min": -64, "max": 64, "step": 1}),
                "mask_blur": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "sampler_name": ("STRING", {"default": "dpmpp_2m_sde", "forceInput": True}),
                "scheduler": ("STRING", {"default": "karras", "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                
                "seed_shift": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1, "tooltip": "Offset added to the main seed for variation"}),
                
                "controlnet_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "control_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "enhancement_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mix between original (0.0) and enhanced (1.0) face. 0.5 = 50/50 blend"}),
            },
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced_image", "enhanced_face", "cropped_face_before")
    FUNCTION = "execute"
    CATEGORY = "CRT/Sampling"

    def execute(self, image, model, positive, vae, control_net, face_bbox_model, face_segm_model, 
                bbox_threshold, segm_threshold, initial_upscale_resolution, upscale_resolution, 
                resize_back_to_original, padding, mask_expand, mask_blur, 
                steps, sampler_name, scheduler, seed, seed_shift, controlnet_strength, control_end, color_match_strength, enhancement_mix):
        
        colored_print("\n🎭 Starting Face Enhancement Pipeline...", Colors.HEADER)
        
        cfg = 1.0
        control_start = 0.0 
        
        actual_seed = seed + seed_shift
        colored_print(f"🎲 Seed Configuration:", Colors.HEADER)
        colored_print(f"   Base Seed: {seed}", Colors.BLUE)
        colored_print(f"   Seed Shift: {seed_shift:+d}", Colors.BLUE)
        colored_print(f"   Final Seed: {actual_seed}", Colors.GREEN)
        
        orig_h, orig_w = image.shape[1:3]
        colored_print(f"📐 Original image dimensions: {orig_w}x{orig_h}", Colors.BLUE)
        colored_print("🔄 Creating negative conditioning...", Colors.CYAN)
        negative = []
        for t in positive:
            negative.append([torch.zeros_like(t[0]), t[1].copy()])
        colored_print(f"   ✅ Created {len(negative)} negative conditioning(s)", Colors.GREEN)
        bbox_filename_only = face_bbox_model.split('/')[-1]
        bbox_path_type = "ultralytics_bbox" if "bbox" in face_bbox_model else "ultralytics_segm"
        bbox_full_path = folder_paths.get_full_path(bbox_path_type, bbox_filename_only)
        segm_filename_only = face_segm_model.split('/')[-1]
        segm_full_path = folder_paths.get_full_path("ultralytics_segm", segm_filename_only)

        colored_print("🤖 Loading detection models...", Colors.HEADER)
        
        if face_bbox_model not in self.detectors: 
            colored_print(f"   📦 Loading bbox model: {bbox_filename_only}", Colors.CYAN)
            self.detectors[face_bbox_model] = UltraDetector(bbox_full_path, "bbox")
        else:
            colored_print(f"   🎯 [Cache Hit] Reusing bbox model: {bbox_filename_only}", Colors.GREEN)
            
        if face_segm_model not in self.detectors: 
            colored_print(f"   📦 Loading segm model: {segm_filename_only}", Colors.CYAN)
            self.detectors[face_segm_model] = UltraDetector(segm_full_path, "segm")
        else:
            colored_print(f"   🎯 [Cache Hit] Reusing segm model: {segm_filename_only}", Colors.GREEN)
        
        bbox_detector = self.detectors[face_bbox_model]
        segm_detector = self.detectors[face_segm_model]
        colored_print(f"📈 Initial image processing...", Colors.HEADER)
        colored_print(f"   Target resolution: {initial_upscale_resolution}px", Colors.BLUE)
        resized_image = ImageResize().execute(image, initial_upscale_resolution, initial_upscale_resolution, method="keep proportion")
        resized_h, resized_w = resized_image.shape[1:3]
        colored_print(f"   Result: {resized_w}x{resized_h}", Colors.GREEN)
        colored_print("🔍 Starting face detection phase...", Colors.HEADER)
        face_bbox_mask = bbox_detector.detect_combined(resized_image, bbox_threshold, 4)
        
        if face_bbox_mask.sum() == 0:
            colored_print("❌ No face detected in image!", Colors.RED)
            colored_print("   Returning original image without enhancement", Colors.YELLOW)
            fallback = torch.zeros_like(image)
            return (image, fallback, fallback)

        colored_print("✅ Face detection successful!", Colors.GREEN)
        colored_print("✂️ Cropping face region...", Colors.HEADER)
        cropped_face_image, face_bounds = BoundedImageCropWithMask().crop(resized_image, face_bbox_mask, padding, padding, padding, padding)
        
        if cropped_face_image is None:
            colored_print("❌ Face cropping failed after detection!", Colors.RED)
            colored_print("   Returning original image without enhancement", Colors.YELLOW)
            fallback = torch.zeros_like(image)
            return (image, fallback, fallback)

        crop_h, crop_w = cropped_face_image.shape[1:3]
        colored_print(f"✅ Face cropped successfully: {crop_w}x{crop_h}", Colors.GREEN)
        colored_print("🎨 Starting face enhancement phase...", Colors.HEADER)
        colored_print(f"   Enhancement Configuration:", Colors.BLUE)
        colored_print(f"     Target Resolution: {upscale_resolution}px", Colors.BLUE)
        colored_print(f"     ControlNet Strength: {controlnet_strength:.3f}", Colors.BLUE)
        colored_print(f"     Enhancement Mix: {enhancement_mix:.3f}", Colors.BLUE)
        colored_print(f"     Sampling Steps: {steps}", Colors.BLUE)
        colored_print(f"     Sampler: {sampler_name} | Scheduler: {scheduler}", Colors.BLUE)
        
        upscaled_face = ImageResize().execute(cropped_face_image, upscale_resolution, upscale_resolution, method='keep proportion')
        upscaled_h, upscaled_w = upscaled_face.shape[1:3]
        colored_print(f"   Upscaled face: {upscaled_w}x{upscaled_h}", Colors.GREEN)
        
        cnet_positive, cnet_negative = apply_controlnet_advanced(positive, negative, control_net, upscaled_face, controlnet_strength, control_start, control_end, vae)
        
        colored_print("🔄 Encoding face to latent space...", Colors.CYAN)
        face_latent = {"samples": vae.encode(upscaled_face)}
        latent_shape = face_latent["samples"].shape
        colored_print(f"   Latent shape: {latent_shape}", Colors.BLUE)
        
        colored_print("🔥 Running diffusion sampling...", Colors.GREEN)
        enhanced_latent_tuple = common_ksampler(model, actual_seed, steps, cfg, sampler_name, scheduler, cnet_positive, cnet_negative, face_latent, denoise=1.0)
        enhanced_latent = enhanced_latent_tuple[0]
        
        colored_print("🎨 Decoding enhanced latent...", Colors.CYAN)
        enhanced_face_image = vae.decode(enhanced_latent["samples"])
        enhanced_h, enhanced_w = enhanced_face_image.shape[1:3]
        colored_print(f"   Enhanced face: {enhanced_w}x{enhanced_h}", Colors.GREEN)
        colored_print("🎨 Applying color matching...", Colors.HEADER)
        color_matched_face = ColorMatch().colormatch(upscaled_face, enhanced_face_image, method='mkl', strength=color_match_strength)[0]
        if enhancement_mix < 1.0:
            colored_print(f"🎭 Blending original and enhanced face...", Colors.HEADER)
            colored_print(f"   Mix ratio: {(1.0-enhancement_mix)*100:.1f}% original + {enhancement_mix*100:.1f}% enhanced", Colors.BLUE)
            
            if upscaled_face.shape != color_matched_face.shape:
                colored_print(f"   ⚠️ Dimension mismatch detected!", Colors.YELLOW)
                colored_print(f"     Original: {upscaled_face.shape}", Colors.BLUE)
                colored_print(f"     Enhanced: {color_matched_face.shape}", Colors.BLUE)
                
                target_h, target_w = upscaled_face.shape[1], upscaled_face.shape[2]
                colored_print(f"   🔄 Resizing enhanced face to match: {target_w}x{target_h}", Colors.CYAN)
                
                color_matched_face = ImageResize().execute(
                    color_matched_face, 
                    target_w, 
                    target_h, 
                    method="stretch", 
                    interpolation="lanczos"
                )
                colored_print(f"   ✅ Resize completed: {color_matched_face.shape}", Colors.GREEN)
            
            color_matched_face = upscaled_face * (1.0 - enhancement_mix) + color_matched_face * enhancement_mix
            colored_print(f"✅ Enhancement blending completed!", Colors.GREEN)
        else:
            colored_print("🎯 Using 100% enhanced face (no blending)", Colors.BLUE)
        colored_print("🎭 Creating final composite mask...", Colors.HEADER)
        precise_face_mask = segm_detector.detect_combined(color_matched_face, segm_threshold, 0)
        feathered_mask, _ = GrowMaskWithBlur().expand_mask(precise_face_mask.unsqueeze(0), expand=mask_expand, tapered_corners=True, blur_radius=mask_blur)
        colored_print("🖼️ Compositing final image...", Colors.HEADER)
        
        if resized_image.dim() == 4:
            final_pil = tensor2pil(resized_image[0])
        else:
            final_pil = tensor2pil(resized_image)
            
        if color_matched_face.dim() == 4:
            enhanced_pil = tensor2pil(color_matched_face[0])
        else:
            enhanced_pil = tensor2pil(color_matched_face)
        
        colored_print(f"   Canvas size: {final_pil.size}", Colors.BLUE)
        colored_print(f"   Enhanced face size: {enhanced_pil.size}", Colors.BLUE)
            
        bounds = face_bounds[0]
        rmin, rmax, cmin, cmax = bounds
        crop_w, crop_h = cmax - cmin + 1, rmax - rmin + 1
        colored_print(f"   Paste region: ({cmin},{rmin}) size: {crop_w}x{crop_h}", Colors.BLUE)
        colored_print("   🎭 Preparing mask and face for composition...", Colors.CYAN)
        paste_mask_pil = to_pil_image(feathered_mask.squeeze()).resize(enhanced_pil.size, Image.Resampling.LANCZOS)
        enhanced_pil_resized = enhanced_pil.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        paste_mask_pil_resized = paste_mask_pil.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        
        colored_print(f"   📐 Final paste dimensions: {enhanced_pil_resized.size}", Colors.BLUE)
        colored_print(f"   🎭 Mask dimensions: {paste_mask_pil_resized.size}", Colors.BLUE)
        colored_print("   ✂️ Pasting enhanced face onto canvas...", Colors.CYAN)
        final_pil.paste(enhanced_pil_resized, (cmin, rmin), paste_mask_pil_resized)
        colored_print("   ✅ Face composition completed!", Colors.GREEN)
        final_image_tensor = pil2tensor(final_pil)
        colored_print("📏 Final image processing...", Colors.HEADER)
        if resize_back_to_original:
            colored_print(f"   🔄 Resizing back to original dimensions: {orig_w}x{orig_h}", Colors.CYAN)
            colored_print(f"   Current tensor shape: {final_image_tensor.shape}", Colors.BLUE)
            final_image = ImageResize().execute(final_image_tensor, orig_w, orig_h, method="stretch")
            final_h, final_w = final_image.shape[1:3]
            colored_print(f"   ✅ Final image shape: {final_image.shape} ({final_w}x{final_h})", Colors.GREEN)
        else:
            colored_print(f"   🎯 Keeping enhanced resolution", Colors.CYAN)
            colored_print(f"   Final tensor shape: {final_image_tensor.shape}", Colors.BLUE)
            final_image = final_image_tensor
            final_h, final_w = final_image.shape[1:3]
        colored_print(f"\n✅ Face Enhancement Pipeline completed successfully!", Colors.HEADER)
        colored_print(f"   🎯 Process Summary:", Colors.GREEN)
        colored_print(f"     Original: {orig_w}x{orig_h}", Colors.GREEN)
        colored_print(f"     Enhanced: {final_w}x{final_h}", Colors.GREEN)
        colored_print(f"     Face Region: {crop_w}x{crop_h}", Colors.GREEN)
        colored_print(f"     Enhancement Mix: {enhancement_mix*100:.1f}%", Colors.GREEN)
        colored_print(f"     Color Match Strength: {color_match_strength:.2f}", Colors.GREEN)
        colored_print(f"     Seed Used: {actual_seed}", Colors.GREEN)
        
        colored_print(f"   📊 Output Details:", Colors.BLUE)
        colored_print(f"     Enhanced Image: {final_image.shape}", Colors.BLUE)
        colored_print(f"     Enhanced Face: {color_matched_face.shape}", Colors.BLUE)
        colored_print(f"     Cropped Face (Before): {cropped_face_image.shape}", Colors.BLUE)

        return (final_image, color_matched_face, cropped_face_image)

NODE_CLASS_MAPPINGS = {
    "FaceEnhancementPipeline": FaceEnhancementPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEnhancementPipeline": "Face Enhancement Pipeline (CRT)"
}
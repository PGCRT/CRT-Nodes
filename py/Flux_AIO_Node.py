import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import comfy.model_patcher
import comfy.sd
import comfy.samplers
from comfy.ldm.modules import attention as comfy_attention
from comfy.model_management import get_torch_device, unet_offload_device, soft_empty_cache, InterruptProcessingException
import comfy.model_management as mm
from nodes import (
    VAELoader, DualCLIPLoader, LoraLoader, StyleModelLoader, CLIPVisionLoader,
    CLIPVisionEncode, StyleModelApply,
    ConditioningZeroOut, EmptyLatentImage, VAEDecode, VAEEncode,
    SaveImage, common_ksampler, ControlNetLoader, ControlNetApplyAdvanced
)
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
from safetensors.torch import load_file
import json
import copy
import re
import math
import os
import random
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as F_vision
import hashlib
import weakref
from contextlib import contextmanager
import gc
from typing import Dict, Any, Optional, Tuple, List
from unittest.mock import patch
import io
import base64
import scipy.ndimage
import cv2

try:
    from transformers import T5Tokenizer, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("\n[FluxAIO_CRT] WARNING: `transformers` library not found. Florence-2 functionality will be disabled.")
    TRANSFORMERS_AVAILABLE = False

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

class FaceEnhancementProcessor:
    def __init__(self, teacache_patcher):
        self.detectors = {}
        self.models = {}
        self.color_matcher = None
        self.teacache_patcher = teacache_patcher

    def _lazy_init_color_matcher(self):
        if self.color_matcher is None:
            try:
                from color_matcher import ColorMatcher
                self.color_matcher = ColorMatcher()
            except ImportError:
                raise Exception("Face Enhancement requires 'color-matcher'. Please 'pip install color-matcher'")
    
    def _load_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
            import ultralytics.nn.tasks as nn_tasks
        except ImportError:
            raise Exception("Face Enhancement requires 'ultralytics'. Please 'pip install ultralytics'")
        
        original_torch_safe_load = nn_tasks.torch_safe_load
        def unsafe_pt_loader(weight, map_location="cpu"):
            ckpt = torch.load(weight, map_location=map_location, weights_only=False)
            return ckpt, weight
        try:
            nn_tasks.torch_safe_load = unsafe_pt_loader
            model = YOLO(model_path)
        finally:
            nn_tasks.torch_safe_load = original_torch_safe_load
        return model

    class UltraDetector:
        def __init__(self, model_path, detection_type, yolo_loader_func):
            self.model = yolo_loader_func(model_path)
            self.type = detection_type
        
        @staticmethod
        def _tensor2pil(image: torch.Tensor) -> Image.Image:
            return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        @staticmethod
        def _create_segmasks(results):
            if not results or not results[1]: return []
            return [(bbox, segm.astype(np.float32), conf) for bbox, segm, conf in zip(results[1], results[2], results[3])]
        
        @staticmethod
        def _dilate_masks(segmasks, dilation_factor):
            if dilation_factor == 0: return segmasks
            kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
            return [(bbox, cv2.dilate(mask, kernel, iterations=1), conf) for bbox, mask, conf in segmasks]
        
        @staticmethod
        def _combine_masks(segmasks):
            if not segmasks: return None
            combined_mask = np.zeros_like(segmasks[0][1], dtype=np.float32)
            for _, mask, _ in segmasks: combined_mask += mask
            return torch.from_numpy(np.clip(combined_mask, 0, 1))

        def _inference_bbox(self, image: Image.Image, confidence: float):
            pred = self.model(image, conf=confidence)
            if not pred or not hasattr(pred[0], 'boxes') or pred[0].boxes is None or pred[0].boxes.xyxy.nelement() == 0: return [[], [], [], []]
            bboxes = pred[0].boxes.xyxy.cpu().numpy()
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

        def _inference_segm(self, image: Image.Image, confidence: float):
            pred = self.model(image, conf=confidence)
            if not pred or not hasattr(pred[0], 'masks') or pred[0].masks is None or pred[0].masks.data.nelement() == 0: return [[], [], [], []]
            bboxes, segms = pred[0].boxes.xyxy.cpu().numpy(), pred[0].masks.data.cpu().numpy()
            results = [[], [], [], []]
            for i in range(bboxes.shape[0]):
                results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
                results[1].append(bboxes[i])
                mask = torch.from_numpy(segms[i])
                scaled_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False).squeeze()
                results[2].append(scaled_mask.numpy())
                results[3].append(pred[0].boxes[i].conf.cpu().numpy())
            return results

        def detect_combined(self, image, threshold, dilation):
            pil_image = self._tensor2pil(image)
            inference_func = self._inference_bbox if self.type == "bbox" else self._inference_segm
            detected_results = inference_func(pil_image, threshold)
            if not detected_results or not detected_results[0]: return torch.zeros((pil_image.height, pil_image.width), dtype=torch.float32)
            segmasks = self._create_segmasks(detected_results)
            if not segmasks: return torch.zeros((pil_image.height, pil_image.width), dtype=torch.float32)
            if dilation > 0: segmasks = self._dilate_masks(segmasks, dilation)
            return self._combine_masks(segmasks)

    def _resize_proportional(self, image, target_long_side):
        _, oh, ow, _ = image.shape
        if oh == 0 or ow == 0: return image

        if ow > oh:
            ratio = target_long_side / ow
            new_w = target_long_side
            new_h = round(oh * ratio)
        else:
            ratio = target_long_side / oh
            new_h = target_long_side
            new_w = round(ow * ratio)

        new_w, new_h = max(new_w, 1), max(new_h, 1)
        
        outputs = image.permute(0, 3, 1, 2)
        outputs = comfy.utils.lanczos(outputs, new_w, new_h)
        return torch.clamp(outputs.permute(0, 2, 3, 1), 0, 1)

    def _resize_to_wh(self, image, width, height):
        outputs = image.permute(0, 3, 1, 2)
        outputs = comfy.utils.lanczos(outputs, width, height)
        return torch.clamp(outputs.permute(0, 2, 3, 1), 0, 1)

    def _bounded_crop(self, image, mask, padding):
        if mask.sum() == 0: return None, None
        rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
        
        rmin_t, rmax_t = torch.where(rows)[0][[0, -1]]
        cmin_t, cmax_t = torch.where(cols)[0][[0, -1]]
        
        rmin = max(rmin_t.item() - padding, 0)
        rmax = min(rmax_t.item() + padding, mask.shape[0] - 1)
        cmin = max(cmin_t.item() - padding, 0)
        cmax = min(cmax_t.item() + padding, mask.shape[1] - 1)

        bounds = (rmin, rmax, cmin, cmax)
        return image[0, rmin:rmax+1, cmin:cmax+1, :].unsqueeze(0), bounds

    def _grow_mask(self, mask, expand, blur_radius):
        if mask.dim() == 4: mask = mask.squeeze(0)
        
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        output = mask.squeeze(0).cpu().numpy().astype(np.float32)
        if expand != 0:
            op = scipy.ndimage.grey_dilation if expand > 0 else scipy.ndimage.grey_erosion
            for _ in range(abs(round(expand))): output = op(output, footprint=kernel)
        
        final_mask = torch.from_numpy(output).unsqueeze(0)
        
        if blur_radius > 0:
            pil_img = F_vision.to_pil_image(final_mask)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
            final_mask = F_vision.to_tensor(pil_img)

        return final_mask.squeeze(0)

    def enhance_face(self, image, model, positive, negative, vae, **kwargs):
        face_bbox_model = kwargs.get('face_bbox_model')
        face_segm_model = kwargs.get('face_segm_model')
        face_cnet_model = kwargs.get('flux_cnet_upscaler_model')

        if (not face_bbox_model or "None" in str(face_bbox_model) or "No models found" in str(face_bbox_model) or
            not face_segm_model or "None" in str(face_segm_model) or "No models found" in str(face_segm_model) or
            not face_cnet_model or "None" in str(face_cnet_model)):
            colored_print("🎭 Face Enhancement: Skipping. A required BBox, SEGM, or ControlNet model is set to 'None'.", Colors.YELLOW)
            return image
        
        self._lazy_init_color_matcher()
        
        sampler_name = kwargs['sampler_name']
        scheduler = kwargs['scheduler']
        resize_back = kwargs['face_resize_back']
        
        final_seed = kwargs['seed'] + kwargs.get('face_seed_shift', 1)
        cfg = 1.0

        bbox_filename_only = face_bbox_model.split('/')[-1]
        bbox_path_type = "ultralytics_bbox" if "bbox" in face_bbox_model else "ultralytics_segm"
        bbox_full_path = folder_paths.get_full_path(bbox_path_type, bbox_filename_only)

        segm_filename_only = face_segm_model.split('/')[-1]
        segm_full_path = folder_paths.get_full_path("ultralytics_segm", segm_filename_only)

        if face_bbox_model not in self.detectors: self.detectors[face_bbox_model] = self.UltraDetector(bbox_full_path, "bbox", self._load_yolo)
        if face_segm_model not in self.detectors: self.detectors[face_segm_model] = self.UltraDetector(segm_full_path, "segm", self._load_yolo)
        if face_cnet_model not in self.models: self.models[face_cnet_model] = ControlNetLoader().load_controlnet(face_cnet_model)[0]
        
        bbox_detector, segm_detector, cnet = self.detectors[face_bbox_model], self.detectors[face_segm_model], self.models[face_cnet_model]
        
        resized_image = self._resize_proportional(image, kwargs['face_initial_resize'])
        face_bbox_mask = bbox_detector.detect_combined(resized_image, kwargs['face_bbox_threshold'], 4)
        if face_bbox_mask.sum() == 0:
            colored_print("🎭 Face Enhancement: No face detected. Skipping.", Colors.YELLOW)
            return image
        
        cropped_face, face_bounds = self._bounded_crop(resized_image, face_bbox_mask, kwargs['face_padding'])
        if cropped_face is None:
            colored_print("🎭 Face Enhancement: Cropping failed. Skipping.", Colors.YELLOW)
            return image
            
        upscaled_face = self._resize_proportional(cropped_face, kwargs['face_upscale_res'])
        cnet_positive, cnet_negative = ControlNetApplyAdvanced().apply_controlnet(positive, negative, cnet, upscaled_face, kwargs['face_cnet_strength'], 0.0, kwargs['face_cnet_end'], vae)
        model_for_face_enhance = model.clone()
        if kwargs.get("enable_teacache"):
            colored_print("🍵 Applying TeaCache to Face Enhancement pass...", Colors.CYAN)
            model_for_face_enhance = self.teacache_patcher.apply(model_for_face_enhance, **kwargs, is_second_pass=True)
        
        face_latent = VAEEncode().encode(vae, upscaled_face)[0]
        enhanced_latent = common_ksampler(model_for_face_enhance, final_seed, kwargs['face_steps'], cfg, sampler_name, scheduler, cnet_positive, cnet_negative, face_latent, 1.0)[0]
        enhanced_face_image = VAEDecode().decode(vae, enhanced_latent)[0]

        ref_np = upscaled_face.cpu()[0].numpy()
        target_np = enhanced_face_image.cpu()[0].numpy()
        result_np = self.color_matcher.transfer(src=target_np, ref=ref_np, method='mkl')
        result_np = target_np + kwargs['face_color_match_strength'] * (result_np - target_np)
        color_matched_face = torch.from_numpy(np.clip(result_np,0,1)).unsqueeze(0).to(image.device)
        
        precise_mask = segm_detector.detect_combined(color_matched_face, kwargs['face_segm_threshold'], 0)
        feathered_mask = self._grow_mask(precise_mask.unsqueeze(0), kwargs['face_mask_expand'], kwargs['face_mask_blur'])
        
        final_pil = self.UltraDetector._tensor2pil(resized_image)
        enhanced_pil = self.UltraDetector._tensor2pil(color_matched_face)
        rmin, rmax, cmin, cmax = face_bounds
        crop_w, crop_h = cmax - cmin + 1, rmax - rmin + 1

        paste_mask_pil = F_vision.to_pil_image(feathered_mask).resize(enhanced_pil.size, Image.Resampling.LANCZOS)
        enhanced_pil_resized = enhanced_pil.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        paste_mask_pil_resized = paste_mask_pil.resize((crop_w, crop_h), Image.Resampling.LANCZOS).convert('L')
        
        final_pil.paste(enhanced_pil_resized, (cmin, rmin), paste_mask_pil_resized)
        
        final_tensor = torch.from_numpy(np.array(final_pil).astype(np.float32) / 255.0).unsqueeze(0).to(image.device)
        
        if resize_back:
            h_orig, w_orig = image.shape[1:3]
            return self._resize_to_wh(final_tensor, w_orig, h_orig)
        else:
            return final_tensor

class SageAttentionPatcher:
    _original_attention = None
    _patched = False

    @staticmethod
    def patch(sage_attention_mode: str):
        if SageAttentionPatcher._patched or sage_attention_mode == "disabled":
            return

        SageAttentionPatcher._original_attention = comfy_attention.optimized_attention
        
        print("Patching comfy attention to use sageattn")
        try:
            from sageattention import sageattn
        except ImportError:
            colored_print("Cannot import 'sageattention'. Please install it: pip install sageattention", Colors.RED)
            return

        def set_sage_func(sage_attention):
            if sage_attention == "auto":
                def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                    return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                return func
            elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                from sageattention import sageattn_qk_int8_pv_fp16_cuda
                def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                    return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                return func
            elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                from sageattention import sageattn_qk_int8_pv_fp16_triton
                def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                    return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                return func
            elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                from sageattention import sageattn_qk_int8_pv_fp8_cuda
                def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                    return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)

        sage_func = set_sage_func(sage_attention_mode)

        @torch.compiler.disable()
        def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout="HND"
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = map(
                    lambda t: t.view(b, -1, heads, dim_head),
                    (q, k, v),
                )
                tensor_layout="NHD"
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
            
            out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
            
            if tensor_layout == "HND":
                if not skip_output_reshape:
                    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    out = out.transpose(1, 2)
                else:
                    out = out.reshape(b, -1, heads * dim_head)
            return out

        comfy_attention.optimized_attention = attention_sage
        for module_path in ["comfy.ldm.flux.math", "comfy.ldm.wan.model"]:
            try:
                module = __import__(module_path, fromlist=['optimized_attention'])
                setattr(module, 'optimized_attention', attention_sage)
            except (ImportError, AttributeError):
                pass
                
        SageAttentionPatcher._patched = True
        colored_print(f"✅ Sage Attention patched with mode: {sage_attention_mode}", Colors.GREEN)

    @staticmethod
    def unpatch():
        if not SageAttentionPatcher._patched:
            return
        
        if SageAttentionPatcher._original_attention:
            comfy_attention.optimized_attention = SageAttentionPatcher._original_attention
            for module_path in ["comfy.ldm.flux.math", "comfy.ldm.wan.model"]:
                try:
                    module = __import__(module_path, fromlist=['optimized_attention'])
                    setattr(module, 'optimized_attention', SageAttentionPatcher._original_attention)
                except (ImportError, AttributeError):
                    pass
            
        SageAttentionPatcher._patched = False
        colored_print("✅ Sage Attention unpatched.", Colors.GREEN)

class PatcherOrderManager:
    _original_patch_model = None
    _original_load_lora = None
    _patched = False

    @staticmethod
    def _patched_patch_model(self, *args, **kwargs):
        with self.use_ejected():
            device_to = get_torch_device()
            lowvram = kwargs.get('lowvram_model_memory', 0)
            
            full_load_override = getattr(self.model, "full_load_override", "auto")
            full_load = full_load_override == "enabled" if full_load_override in ["enabled", "disabled"] else lowvram == 0
            
            self.load(device_to, lowvram_model_memory=lowvram, force_patch_weights=kwargs.get('force_patch_weights', False), full_load=full_load)
            
            for k in self.object_patches:
                old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
                if k not in self.object_patches_backup:
                    self.object_patches_backup[k] = old
        self.inject_model()
        return self.model

    @staticmethod
    def _patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip):
        patch_keys = list(model.object_patches_backup.keys())
        for k in patch_keys:
            comfy.utils.set_attr(model.model, k, model.object_patches_backup[k])

        key_map = {}
        if model is not None: key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None: key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
        
        new_modelpatcher = model.clone() if model else None
        new_clip = clip.clone() if clip else None
        
        if new_modelpatcher: new_modelpatcher.add_patches(loaded, strength_model)
        if new_clip: new_clip.add_patches(loaded, strength_clip)

        return (new_modelpatcher, new_clip)

    @staticmethod
    def apply(patch_order="weight_patch_first"):
        if PatcherOrderManager._patched:
            return
        
        PatcherOrderManager._original_patch_model = comfy.model_patcher.ModelPatcher.patch_model
        PatcherOrderManager._original_load_lora = comfy.sd.load_lora_for_models
        
        if patch_order == "weight_patch_first":
            comfy.model_patcher.ModelPatcher.patch_model = PatcherOrderManager._patched_patch_model
            comfy.sd.load_lora_for_models = PatcherOrderManager._patched_load_lora_for_models
            colored_print("✅ Model Patcher order set to 'weight_patch_first'.", Colors.GREEN)

        PatcherOrderManager._patched = True

    @staticmethod
    def restore():
        if not PatcherOrderManager._patched:
            return
            
        if PatcherOrderManager._original_patch_model:
            comfy.model_patcher.ModelPatcher.patch_model = PatcherOrderManager._original_patch_model
        if PatcherOrderManager._original_load_lora:
            comfy.sd.load_lora_for_models = PatcherOrderManager._original_load_lora
        
        PatcherOrderManager._patched = False
        colored_print("✅ Model Patcher order restored.", Colors.GREEN)

class TeaCachePatcher:
    SUPPORTED_MODELS_COEFFICIENTS = {
        "flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    }

    @staticmethod
    def _poly1d(coefficients, x):
        result = torch.zeros_like(x)
        for i, coeff in enumerate(coefficients):
            result += coeff * (x ** (len(coefficients) - 1 - i))
        return result
    
    @staticmethod
    def _teacache_flux_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance = None, control = None, transformer_options={}, attn_mask=None):
        from comfy.ldm.flux.layers import timestep_embedding, apply_mod

        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        enable_teacache = transformer_options.get("enable_teacache", True)
        
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None: raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})
        img_mod1, _ = self.double_blocks[0].img_mod(vec)
        modulated_inp = self.double_blocks[0].img_norm1(img)
        modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift)
        ca_idx = 0

        should_calc = True
        if enable_teacache:
            if not hasattr(self, 'accumulated_rel_l1_distance'):
                self.accumulated_rel_l1_distance = 0
            else:
                try:
                    self.accumulated_rel_l1_distance += TeaCachePatcher._poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                    if self.accumulated_rel_l1_distance < rel_l1_thresh:
                        should_calc = False
                    else:
                        self.accumulated_rel_l1_distance = 0
                except:
                    self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp

        if not should_calc:
            if img.shape[0] > self.previous_residual.shape[0] and self.previous_residual.shape[0] == 1:
                img += self.previous_residual.repeat(img.shape[0], 1, 1, 1).to(img.device)
            else:
                img += self.previous_residual.to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                        return out
                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask}, {"original_block": block_wrap})
                    txt, img = out["txt"], out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None:
                    if (control_i := control.get("input")) and i < len(control_i) and (add := control_i[i]) is not None:
                        img += add
                if getattr(self, "pulid_data", {}):
                    if i % self.pulid_double_interval == 0:
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                                img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

            img = torch.cat((txt, img), 1)
            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {"img": block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))}
                        return out
                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None:
                    if (control_o := control.get("output")) and i < len(control_o) and (add := control_o[i]) is not None:
                        img[:, txt.shape[1] :, ...] += add
                if getattr(self, "pulid_data", {}):
                    real_img, txt_part = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                    if i % self.pulid_single_interval == 0:
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                                real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                        ca_idx += 1
                    img = torch.cat((txt_part, real_img), 1)

            img = img[:, txt.shape[1] :, ...]
            self.previous_residual = (img - ori_img).to(unet_offload_device())

        return self.final_layer(img, vec)
    
    @staticmethod
    def apply(model, **kwargs):
        colored_print("Applying TeaCache for FLUX...", Colors.CYAN)
        new_model = model.clone()
        diffusion_model = new_model.get_model_object("diffusion_model")
        
        is_second_pass = kwargs.get("is_second_pass", False)
        
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        
        if is_second_pass:
            new_model.model_options["transformer_options"]["rel_l1_thresh"] = kwargs.get("teacache_2nd_rel_l1_thresh", 0.2)
            new_model.model_options["transformer_options"]["start_percent"] = kwargs.get("teacache_2nd_start_percent", 0.0)
            new_model.model_options["transformer_options"]["end_percent"] = kwargs.get("teacache_2nd_end_percent", 1.0)
        else:
            new_model.model_options["transformer_options"]["rel_l1_thresh"] = kwargs.get("teacache_rel_l1_thresh", 0.4)
            new_model.model_options["transformer_options"]["start_percent"] = kwargs.get("teacache_start_percent", 0.0)
            new_model.model_options["transformer_options"]["end_percent"] = kwargs.get("teacache_end_percent", 1.0)

        new_model.model_options["transformer_options"]["coefficients"] = TeaCachePatcher.SUPPORTED_MODELS_COEFFICIENTS["flux"]
        
        forward_patch = TeaCachePatcher._teacache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
        if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
            delattr(diffusion_model, 'accumulated_rel_l1_distance')
        if hasattr(diffusion_model, 'previous_residual'):
            delattr(diffusion_model, 'previous_residual')
            
        context = patch.multiple(diffusion_model, forward_orig=forward_patch)
        
        def unet_wrapper_function(model_function, wrapper_kwargs):
            c = wrapper_kwargs["c"]
            sigmas = c.get("transformer_options", {}).get("sample_sigmas")
            start_percent = c.get("transformer_options", {}).get("start_percent", 0.0)
            end_percent = c.get("transformer_options", {}).get("end_percent", 1.0)
            
            if sigmas is None:
                return model_function(wrapper_kwargs["input"], wrapper_kwargs["timestep"], **c)

            timestep = wrapper_kwargs["timestep"][0]
            matched_indices = (sigmas == timestep).nonzero()
            current_step_index = matched_indices[0].item() if len(matched_indices) > 0 else 0
            
            if current_step_index == 0 and hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                delattr(diffusion_model, 'accumulated_rel_l1_distance')
            
            current_percent = current_step_index / max(1, len(sigmas) - 1)
            
            c["transformer_options"]["enable_teacache"] = (start_percent <= current_percent <= end_percent)
            
            with context:
                return model_function(wrapper_kwargs["input"], wrapper_kwargs["timestep"], **c)
        
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        pass_name = "2nd Pass" if is_second_pass else "1st Pass"
        colored_print(f"✅ TeaCache applied successfully for {pass_name}.", Colors.GREEN)
        return new_model

def patch_torch_for_compile():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
        if not getattr(OptimizedModule, "_patched", False):
            def __getattribute__(self, name):
                if name == "_orig_mod":
                    return object.__getattribute__(self, "_modules")[name]
                if name in ("_modules", "state_dict", "load_state_dict", "parameters", "named_parameters", "buffers", "named_buffers", "children", "named_children", "modules", "named_modules"):
                    return getattr(object.__getattribute__(self, "_orig_mod"), name)
                return object.__getattribute__(self, name)
            
            OptimizedModule.__getattribute__ = __getattribute__
            OptimizedModule._patched = True
            colored_print("✅ Patched torch._dynamo.eval_frame.OptimizedModule for compatibility.", Colors.CYAN)
    except (ImportError, AttributeError):
        pass

class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE", ), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

class FastPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "quality" : ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "CRT/WIP"
    OUTPUT_NODE = True
    
    def tensor_to_pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def preview(self, image, format, quality):        
        pil_image = self.tensor_to_pil(image)

        with io.BytesIO() as buffered:
            pil_image.save(buffered, format=format, quality=quality)
            img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
        return { "ui": {"previews": [{"format": format.lower(), "source": img_base64}]}, "result": ()}


class LRUCache:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()

class ModelManager:
    def __init__(self, max_cache_size: int = 3):
        self.cache = LRUCache(max_cache_size)
        
    @contextmanager
    def device_context(self, model, device=None, offload_device=None):
        if device is None:
            device = get_torch_device()
        if offload_device is None:
            offload_device = unet_offload_device()
            
        try:
            model.to(device)
            yield model
        finally:
            model.to(offload_device)
            soft_empty_cache()
    
    def load_models(self, kwargs):
        cache_key = self._create_cache_key(kwargs)
        cached_models = self.cache.get(cache_key)
        
        if cached_models:
            colored_print("📦 [Cache Hit] Reusing loaded models.", Colors.GREEN)
            return cached_models
        
        colored_print("📦 [Cache Miss] Loading models...", Colors.BLUE)
        
        flux_model = comfy.sd.load_diffusion_model(
            folder_paths.get_full_path("diffusion_models", kwargs["flux_model_name"])
        )
        colored_print(f"✅ Loaded Flux model: {kwargs['flux_model_name']}", Colors.GREEN)
        
        vae = VAELoader().load_vae(kwargs["vae_name"])[0]
        clip = DualCLIPLoader().load_clip(kwargs["clip_l_name"], kwargs["t5_name"], "flux")[0]
        colored_print("✅ Loaded dual CLIP with 'flux' type", Colors.GREEN)
        
        style_model = StyleModelLoader().load_style_model(kwargs["style_model_name"])[0]
        clip_vision = CLIPVisionLoader().load_clip(kwargs["clip_vision_name"])[0]
        
        models = (flux_model, vae, clip, style_model, clip_vision)
        self.cache.put(cache_key, models)
        return models
    
    def _create_cache_key(self, kwargs):
        return json.dumps({
            k: v for k, v in kwargs.items() 
            if k in ["flux_model_name", "vae_name", "clip_l_name", "t5_name", "style_model_name", "clip_vision_name"]
        }, sort_keys=True)

class PostProcessingPipeline:
    @staticmethod
    def apply_effects(image, **kwargs):
        if not PostProcessingPipeline._has_any_effects_enabled(kwargs):
            return image
            
        colored_print("🎨 Applying Post-Processing Effects...", Colors.HEADER)
        img = image.clone()
        
        effect_order = [
            ('enable_lens_distortion', PostProcessingPipeline._apply_lens_distortion),
            ('enable_chromatic_aberration', PostProcessingPipeline._apply_chromatic_aberration),
            ('enable_temp_tint', PostProcessingPipeline._apply_temperature_tint),
            ('enable_levels', PostProcessingPipeline._apply_levels),
            ('enable_color_wheels', PostProcessingPipeline._apply_color_wheels),
            ('enable_sharpen', PostProcessingPipeline._apply_sharpening),
            ('enable_small_glow', lambda img, kw: PostProcessingPipeline._apply_glow(img, kw, 'small')),
            ('enable_large_glow', lambda img, kw: PostProcessingPipeline._apply_glow(img, kw, 'large')),
            ('enable_glare', PostProcessingPipeline._apply_glare),
            ('enable_radial_blur', PostProcessingPipeline._apply_radial_blur),
            ('enable_vignette', PostProcessingPipeline._apply_vignette),
            ('enable_film_grain', PostProcessingPipeline._apply_film_grain)
        ]
        
        for enable_key, effect_func in effect_order:
            if kwargs.get(enable_key, False):
                colored_print(f"...Applying {enable_key.replace('enable_', '').replace('_', ' ').title()}", Colors.CYAN)
                img = effect_func(img, kwargs)
                
        return img
    
    @staticmethod
    def _has_any_effects_enabled(kwargs):
        effect_keys = [
            'enable_lens_distortion', 'enable_chromatic_aberration', 'enable_temp_tint',
            'enable_levels', 'enable_color_wheels', 'enable_sharpen', 'enable_small_glow',
            'enable_large_glow', 'enable_glare', 'enable_radial_blur', 'enable_vignette',
            'enable_film_grain'
        ]
        return any(kwargs.get(key, False) for key in effect_keys)

    @staticmethod
    def _apply_temperature_tint(img, kwargs):
        temp, tint = kwargs.get('temperature', 0.0) / 100.0, kwargs.get('tint', 0.0) / 100.0
        if temp == 0.0 and tint == 0.0: return img
        img_copy = img.clone()
        if temp > 0: 
            img_copy[..., 0] *= (1.0 + temp * 0.3)
            img_copy[..., 2] *= (1.0 - temp * 0.2)
        elif temp < 0: 
            img_copy[..., 0] *= (1.0 + temp * 0.2)
            img_copy[..., 2] *= (1.0 - temp * 0.3)
        if tint > 0: 
            img_copy[..., 1] *= (1.0 + tint * 0.3)
        elif tint < 0: 
            img_copy[..., 0] *= (1.0 - tint * 0.15)
            img_copy[..., 2] *= (1.0 - tint * 0.15)
        return torch.clamp(img_copy, 0, 1)

    @staticmethod
    def _apply_levels(img, kwargs):
        img_copy = img.clone() * (2.0 ** kwargs.get('exposure', 0.0))
        gamma = kwargs.get('gamma', 1.0)
        if gamma != 1.0 and gamma > 0:
            img_copy = torch.pow(torch.clamp(img_copy, 1e-6, 1.0), 1.0 / gamma)
        img_copy += kwargs.get('brightness', 0.0)
        img_copy = (img_copy - 0.5) * kwargs.get('contrast', 1.0) + 0.5
        img_copy = torch.clamp(img_copy, 0, 1)
        
        saturation, vibrance = kwargs.get('saturation', 1.0), kwargs.get('vibrance', 0.0)
        if saturation != 1.0 or vibrance != 0.0:
            gray = torch.mean(img_copy, dim=-1, keepdim=True)
            if saturation != 1.0: 
                img_copy = gray + (img_copy - gray) * saturation
            if vibrance != 0.0:
                v_mask = 1.0 - torch.clamp(torch.abs(img_copy - gray).max(dim=-1, keepdim=True)[0] * 2.0, 0, 1)
                img_copy = gray + (img_copy - gray) * (1.0 + vibrance * v_mask)
        return torch.clamp(img_copy, 0, 1)

    @staticmethod
    def _apply_color_wheels(img, kwargs):
        lift = torch.tensor([kwargs.get(f'lift_{c}', 0.0) for c in 'rgb'], device=img.device, dtype=img.dtype)
        gamma_adj = torch.tensor([kwargs.get(f'gamma_{c}', 1.0) for c in 'rgb'], device=img.device, dtype=img.dtype)
        gain = torch.tensor([kwargs.get(f'gain_{c}', 1.0) for c in 'rgb'], device=img.device, dtype=img.dtype)
        img_copy = torch.clamp(img + lift * (1.0 - img), 0, 1)
        safe_gamma = torch.where(gamma_adj <= 1e-6, 1e-6, gamma_adj)
        img_copy = torch.pow(torch.clamp(img_copy, 1e-6, 1.0), 1.0 / safe_gamma)
        return torch.clamp(img_copy * gain, 0, 1)

    @staticmethod
    def _apply_sharpening(img, kwargs):
        strength, radius, threshold = kwargs.get('sharpen_strength', 2.5), kwargs.get('sharpen_radius', 1.85), kwargs.get('sharpen_threshold', 0.015)
        if strength == 0.0: return img
        img_conv = img.permute(0, 3, 1, 2)
        k_size = max(3, int(radius * 6) | 1)
        coords = torch.arange(k_size, dtype=img.dtype, device=img.device) - k_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * radius ** 2))
        kernel_1d /= kernel_1d.sum()
        k_h = kernel_1d.view(1, 1, 1, -1).repeat(img_conv.shape[1], 1, 1, 1)
        k_v = kernel_1d.view(1, 1, -1, 1).repeat(img_conv.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.conv2d(img_conv, k_h, padding=(0, k_size//2), groups=img_conv.shape[1]), k_v, padding=(k_size//2, 0), groups=img_conv.shape[1])
        unsharp_mask = img - blurred.permute(0, 2, 3, 1)
        if threshold > 0: 
            unsharp_mask *= (torch.abs(unsharp_mask) > threshold).float()
        return torch.clamp(img + unsharp_mask * strength, 0, 1)

    @staticmethod
    def _apply_glow(img, kwargs, glow_type):
        intensity, radius, threshold = kwargs.get(f'{glow_type}_glow_intensity'), kwargs.get(f'{glow_type}_glow_radius'), kwargs.get(f'{glow_type}_glow_threshold')
        if intensity == 0.0 or radius == 0.0: return img
        img_conv = img.permute(0, 3, 1, 2)
        k_size = max(3, int(radius * 6) | 1)
        coords = torch.arange(k_size, dtype=img.dtype, device=img.device) - k_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * radius ** 2))
        kernel_1d /= kernel_1d.sum()
        k_h = kernel_1d.view(1, 1, 1, -1).repeat(img_conv.shape[1], 1, 1, 1)
        k_v = kernel_1d.view(1, 1, -1, 1).repeat(img_conv.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.conv2d(img_conv, k_h, padding=(0, k_size//2), groups=img_conv.shape[1]), k_v, padding=(k_size//2, 0), groups=img_conv.shape[1])
        glow_mask = torch.clamp((torch.mean(img, dim=-1, keepdim=True) - threshold) / (1.0 - threshold + 1e-6), 0, 1)
        return torch.clamp(img + blurred.permute(0, 2, 3, 1) * glow_mask * intensity, 0, 1)

    @staticmethod
    def _apply_glare(img, kwargs):
        glare_type, intensity, length, angle, threshold, quality, ray_width = (
            kwargs.get('glare_type'), kwargs.get('glare_intensity'), kwargs.get('glare_length'), 
            kwargs.get('glare_angle'), kwargs.get('glare_threshold'), kwargs.get('glare_quality'), 
            kwargs.get('glare_ray_width')
        )
        if intensity == 0.0 or length == 0.0: return img
        glare_source = img * torch.clamp((torch.mean(img, dim=-1, keepdim=True) - threshold) / (1.0 - threshold + 1e-6), 0, 1)
        glare_source_bchw = glare_source.permute(0, 3, 1, 2)
        h, w = glare_source_bchw.shape[2:4]
        
        if 'star' in glare_type: 
            glare_effect_bchw = PostProcessingPipeline._create_star_glare(glare_source_bchw, int(glare_type.split('_')[1]), length, angle, h, w, quality, ray_width)
        elif glare_type == 'anamorphic_h': 
            glare_effect_bchw = PostProcessingPipeline._create_anamorphic_glare(glare_source_bchw, length, h, w)
        else: return img
        
        if torch.isnan(glare_effect_bchw).any() or torch.isinf(glare_effect_bchw).any(): return img
        return torch.clamp(img + glare_effect_bchw.permute(0, 2, 3, 1) * intensity, 0, 1)

    @staticmethod
    def _create_star_glare(img_bchw, rays, length, angle, h, w, quality, ray_width):
        device, dtype = img_bchw.device, img_bchw.dtype
        k_size = min(max(int(length * 6.0), 3) | 1, min(h, w) // 2 | 1)
        coords = torch.linspace(-k_size // 2, k_size // 2, k_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        angle_rad = math.radians(angle)
        xr, yr = xx * math.cos(angle_rad) - yy * math.sin(angle_rad), xx * math.sin(angle_rad) + yy * math.cos(angle_rad)
        kernel = torch.zeros_like(xx)
        for i in range(rays):
            ray_angle = i * 2 * math.pi / rays
            dist_sq = (xr * math.sin(ray_angle) - yr * math.cos(ray_angle))**2
            profile_across = torch.exp(-dist_sq / (2 * ray_width**2 + 1e-6))
            profile_along = torch.exp(-(xr**2 + yr**2) / (2 * length**2 + 1e-6))
            kernel += profile_across * profile_along
        if kernel.sum() < 1e-6: return torch.zeros_like(img_bchw)
        kernel = (kernel / kernel.sum()).view(1, 1, k_size, k_size)
        return F.conv2d(img_bchw, kernel.expand(img_bchw.shape[1], 1, -1, -1), padding=k_size//2, groups=img_bchw.shape[1])

    @staticmethod
    def _create_anamorphic_glare(img_bchw, length, h, w):
        sigma_h, sigma_v = length * 2.0, length * 0.1 + 0.3
        k_size_h, k_size_v = max(3, min(int(sigma_h*6)|1, w-1|1))|1, max(3, min(int(sigma_v*6)|1, h-1|1))|1
        xh = torch.exp(-(torch.linspace(-k_size_h//2,k_size_h//2,k_size_h,device=img_bchw.device)**2)/(2*sigma_h**2+1e-6))
        yv = torch.exp(-(torch.linspace(-k_size_v//2,k_size_v//2,k_size_v,device=img_bchw.device)**2)/(2*sigma_v**2+1e-6))
        k_h, k_v = (xh/xh.sum()).view(1,1,1,-1).expand(img_bchw.shape[1],1,1,-1), (yv/yv.sum()).view(1,1,-1,1).expand(img_bchw.shape[1],1,-1,1)
        return F.conv2d(F.conv2d(img_bchw, k_h, padding=(0,k_size_h//2), groups=img_bchw.shape[1]), k_v, padding=(k_size_v//2,0), groups=img_bchw.shape[1])

    @staticmethod
    def _apply_chromatic_aberration(img, kwargs):
        strength, edge_falloff, enable_hue_shift, hue_shift_degrees = (
            kwargs.get('ca_strength'), kwargs.get('ca_edge_falloff'), 
            kwargs.get('enable_ca_hue_shift'), kwargs.get('ca_hue_shift_degrees')
        )
        if strength == 0.0: return img
        b, h, w, c = img.shape
        device, dtype = img.device, img.dtype
        y, x = torch.meshgrid(torch.linspace(-1,1,h,device=device,dtype=dtype),torch.linspace(-1,1,w,device=device,dtype=dtype),indexing='ij')
        dist = torch.sqrt(x**2 + y**2)
        disp_scaler = strength * ((dist / (math.sqrt(2.0) + 1e-6)) ** edge_falloff)
        grid_r = torch.stack([x + disp_scaler * x, y + disp_scaler * y],-1).unsqueeze(0).expand(b,-1,-1,-1)
        grid_b = torch.stack([x - disp_scaler * x, y - disp_scaler * y],-1).unsqueeze(0).expand(b,-1,-1,-1)
        img_bchw = img.permute(0, 3, 1, 2)
        r_ch = F.grid_sample(img_bchw[:,0:1],grid_r,mode='bilinear',padding_mode='border',align_corners=True)
        g_ch = img_bchw[:,1:2]
        b_ch = F.grid_sample(img_bchw[:,2:3],grid_b,mode='bilinear',padding_mode='border',align_corners=True)
        
        if enable_hue_shift and c==3 and hue_shift_degrees!=0:
            sh_r_c = PostProcessingPipeline._hue_shift_rgb_color_vector(torch.tensor([1.,0.,0.],device=device,dtype=dtype),hue_shift_degrees)
            sh_b_c = PostProcessingPipeline._hue_shift_rgb_color_vector(torch.tensor([0.,0.,1.],device=device,dtype=dtype),-hue_shift_degrees)
            result_bchw = torch.cat([r_ch*sh_r_c[0]+b_ch*sh_b_c[0], r_ch*sh_r_c[1]+g_ch+b_ch*sh_b_c[1], r_ch*sh_r_c[2]+b_ch*sh_b_c[2]], 1)
        else: 
            result_bchw = torch.cat([r_ch, g_ch, b_ch], 1)
        return result_bchw.permute(0, 2, 3, 1)

    @staticmethod
    def _hue_shift_rgb_color_vector(rgb, degrees):
        r, g, b = rgb[0].item(), rgb[1].item(), rgb[2].item()
        max_v, min_v = max(r,g,b), min(r,g,b)
        delta = max_v-min_v
        v, s = max_v, delta/max_v if max_v > 1e-6 else 0
        h = 0
        if delta > 1e-6:
            if r==max_v: h=(g-b)/delta
            elif g==max_v: h=2+(b-r)/delta
            else: h=4+(r-g)/delta
        h = ((h * 60) + degrees) % 360
        h_i = h / 60
        c = v * s
        x = c * (1 - abs(h_i % 2 - 1))
        m = v - c
        
        if 0 <= h_i < 1: ro, go, bo = c, x, 0
        elif 1 <= h_i < 2: ro, go, bo = x, c, 0
        elif 2 <= h_i < 3: ro, go, bo = 0, c, x
        elif 3 <= h_i < 4: ro, go, bo = 0, x, c
        elif 4 <= h_i < 5: ro, go, bo = x, 0, c
        else: ro, go, bo = c, 0, x
        
        return torch.tensor([ro+m,go+m,bo+m],device=rgb.device,dtype=rgb.dtype)

    @staticmethod
    def _apply_vignette(img, kwargs):
        strength, radius, softness = kwargs.get('vignette_strength'), kwargs.get('vignette_radius'), kwargs.get('vignette_softness')
        if strength == 0.0: return img
        h, w = img.shape[1:3]
        min_dim = max(1, min(h,w))
        y, x = torch.meshgrid(torch.linspace(-1,1,h,device=img.device)*h/min_dim, torch.linspace(-1,1,w,device=img.device)*w/min_dim, indexing='ij')
        dist = torch.sqrt(x**2 + y**2)
        vignette = torch.clamp(1.0 - (dist-radius)/max(softness,1e-6),0,1)
        return img * (1.0 - strength*(1.0-vignette)).unsqueeze(0).unsqueeze(-1)

    @staticmethod
    def _apply_radial_blur(img, kwargs):
        blur_type = kwargs.get('radial_blur_type', 'spin')
        if blur_type == 'zoom':
            return PostProcessingPipeline._apply_zoom_blur(img, kwargs)
        elif blur_type == 'spin':
            return PostProcessingPipeline._apply_radial_spin_blur(img, kwargs)
        return img

    @staticmethod
    def _apply_zoom_blur(img, kwargs):
        strength, cx, cy, falloff, samples = (
            kwargs.get('radial_blur_strength'), kwargs.get('radial_blur_center_x'), 
            kwargs.get('radial_blur_center_y'), kwargs.get('radial_blur_falloff'), 
            kwargs.get('radial_blur_samples')
        )
        if strength == 0.0 or samples <= 0: return img
        b,h,w,c = img.shape
        device,dtype=img.device,img.dtype
        y_grid,x_grid=torch.meshgrid(torch.linspace(0,1,h,device=device,dtype=dtype),torch.linspace(0,1,w,device=device,dtype=dtype),indexing='ij')
        dx,dy=x_grid-cx,y_grid-cy
        acc=torch.zeros_like(img)
        img_bchw=img.permute(0,3,1,2)
        for i in range(samples):
            t = (i/max(1,samples-1.0))*strength - (strength/2.0) if samples>1 else 0.0
            grid = torch.stack([(x_grid+dx*t)*2-1,(y_grid+dy*t)*2-1],-1).unsqueeze(0).expand(b,-1,-1,-1)
            acc += F.grid_sample(img_bchw,grid,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1)
        blurred = acc/samples
        mask = (1.0-torch.exp(-(dx**2+dy**2)/(2*(falloff+1e-6)**2))).unsqueeze(0).unsqueeze(-1).expand(b,-1,-1,c)
        return torch.clamp(img*(1-mask)+blurred*mask, 0,1)

    @staticmethod
    def _apply_radial_spin_blur(img, kwargs):
        strength, cx, cy, falloff, samples = (
            kwargs.get('radial_blur_strength'), kwargs.get('radial_blur_center_x'), 
            kwargs.get('radial_blur_center_y'), kwargs.get('radial_blur_falloff'), 
            kwargs.get('radial_blur_samples')
        )
        if strength == 0.0 or samples <= 0: return img
        b,h,w,c = img.shape
        device,dtype=img.device,img.dtype
        y_grid,x_grid=torch.meshgrid(torch.linspace(0,1,h,device=device,dtype=dtype),torch.linspace(0,1,w,device=device,dtype=dtype),indexing='ij')
        rx,ry=x_grid-cx,y_grid-cy
        acc=torch.zeros_like(img)
        img_bchw=img.permute(0,3,1,2)
        for i in range(samples):
            angle = (i/max(1,samples-1.0))*strength - (strength/2.0) if samples>1 else 0.0
            ca,sa=math.cos(angle),math.sin(angle)
            grid = torch.stack([(rx*ca-ry*sa+cx)*2-1,(rx*sa+ry*ca+cy)*2-1],-1).unsqueeze(0).expand(b,-1,-1,-1)
            acc += F.grid_sample(img_bchw,grid,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1)
        blurred = acc/samples
        mask = (1.0-torch.exp(-(rx**2+ry**2)/(2*(falloff+1e-6)**2))).unsqueeze(0).unsqueeze(-1).expand(b,-1,-1,c)
        return torch.clamp(img*(1-mask)+blurred*mask, 0,1)

    @staticmethod
    def _apply_lens_distortion(img, kwargs):
        dist_coeff = kwargs.get('barrel_distortion', 0.0)
        if dist_coeff == 0.0: return img
        b,h,w,c = img.shape
        device,dtype=img.device,img.dtype
        y,x = torch.meshgrid(torch.linspace(-1,1,h,device=device,dtype=dtype),torch.linspace(-1,1,w,device=device,dtype=dtype),indexing='ij')
        x_r,y_r = (x*w/h,y) if w>=h else (x,y*h/w)
        r_sq=x_r**2+y_r**2
        scale = torch.where(1-dist_coeff*r_sq < 1e-4, 1e-4, 1-dist_coeff*r_sq)
        grid = torch.stack([x/scale,y/scale],-1).unsqueeze(0).expand(b,-1,-1,-1)
        return F.grid_sample(img.permute(0,3,1,2),grid,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1)

    @staticmethod
    def _apply_film_grain(img, kwargs):
        intensity, size, color = kwargs.get('grain_intensity'), kwargs.get('grain_size'), kwargs.get('grain_color_amount')
        if intensity == 0.0: return img
        b,h,w,c = img.shape
        device,dtype=img.device,dtype
        gh,gw=max(1,math.ceil(h/size)),max(1,math.ceil(w/size))
        noise_ch = 3 if c==3 and color>0 else 1
        noise = torch.randn(b,gh,gw,noise_ch,device=device,dtype=dtype)
        if noise_ch==1 and c==3: 
            noise = noise.expand(b,gh,gw,c)
        elif noise_ch==3 and c==3 and color<1.0:
            mono = torch.mean(noise,-1,True).expand_as(noise)
            noise = mono*(1-color)+noise*color
        if gh!=h or gw!=w: 
            noise=F.interpolate(noise.permute(0,3,1,2),size=(h,w),mode='bilinear',align_corners=False).permute(0,2,3,1)
        lightness = torch.mean(img,dim=-1,keepdim=True)
        grain_mask = 4*lightness*(1-lightness)
        return torch.clamp(img + noise*intensity*grain_mask, 0,1)

class Florence2Processor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.dtype = None
        self.model_manager = None

    def load_model(self, model_name, precision, attention):
        try:
            from huggingface_hub import snapshot_download
            import transformers
            model_path = os.path.join(folder_paths.get_folder_paths("LLM")[0], model_name.rsplit('/', 1)[-1])
            if not os.path.exists(model_path):
                colored_print(f"Downloading Florence-2 model to: {model_path}", Colors.YELLOW)
                snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
            
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
            device = get_torch_device()
            offload_device = unet_offload_device()
            
            if self.model_manager is None:
                self.model_manager = ModelManager(max_cache_size=1)
            
            from unittest.mock import patch
            def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
                if not str(filename).endswith("modeling_florence2.py"): return transformers.dynamic_module_utils.get_imports(filename)
                imports = transformers.dynamic_module_utils.get_imports(filename)
                try: imports.remove("flash_attn")
                except ValueError: pass
                except Exception as e: colored_print(f"An unexpected error occurred while trying to remove 'flash_attn' import: {e}", Colors.RED)
                return imports
            
            if TRANSFORMERS_AVAILABLE and transformers.__version__ < '4.41.0':
                with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): 
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, torch_dtype=dtype, trust_remote_code=True).to(offload_device)
            elif TRANSFORMERS_AVAILABLE:
                try:
                    from .modeling_florence2 import Florence2ForConditionalGeneration
                    self.model = Florence2ForConditionalGeneration.from_pretrained(model_path, attn_implementation=attention, torch_dtype=dtype).to(offload_device)
                except (ImportError, AttributeError) as e:
                    colored_print(f"Could not import custom Florence2ForConditionalGeneration. Ensure modeling_florence2.py is available. Falling back to AutoModelForCausalLM. Error: {e}", Colors.YELLOW)
                    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): 
                        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, torch_dtype=dtype, trust_remote_code=True).to(offload_device)
            
            self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) if TRANSFORMERS_AVAILABLE else None
            self.dtype = dtype
            return self.model, self.processor, self.dtype
        except ImportError as e: 
            colored_print(f"Error: Transformers library not found. Please install it with `pip install transformers`. Details: {e}", Colors.RED)
            return None, None, None
        except Exception as e: 
            colored_print(f"Error loading Florence-2 model: {e}", Colors.RED)
            return None, None, None

    def process_image(self, image, task, text_input="", max_new_tokens=1024, num_beams=3, do_sample=True, seed=1):
        if self.model is None or self.processor is None:
            colored_print("Warning: Florence-2 model or processor not loaded. Cannot process image.", Colors.YELLOW)
            return "", None, None, None
        
        device, offload_device = get_torch_device(), unet_offload_device()
        if TRANSFORMERS_AVAILABLE and seed: 
            from transformers import set_seed
            set_seed(self.hash_seed(seed))
        
        prompts = {
            'region_caption': '<OD>','dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>','caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>','more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>','ocr_with_region': '<OCR_WITH_REGION>','docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>','prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>','prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        
        task_prompt = prompts.get(task, '<CAPTION>')
        if task in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa'] and not text_input:
            colored_print(f"Warning: Text input required for task {task}, but none provided.", Colors.YELLOW)
            return "", None, None, None
        
        prompt = f"{task_prompt} {text_input}" if text_input else task_prompt
        image_pil = self._prepare_image_for_florence(image)
        
        try:
            with self.model_manager.device_context(self.model, device, offload_device):
                inputs = self.processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(self.dtype).to(device)
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens, 
                    do_sample=do_sample, num_beams=num_beams
                )
                results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                del inputs, generated_ids
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            colored_print(f"Error during Florence-2 inference: {e}", Colors.RED)
            return "", None, None, None
        finally:
            if self.model is not None:
                self.model.to(offload_device)
            soft_empty_cache()
        
        clean_results = re.sub(r'</?s>|<[^>]*>', '\n' if task == 'ocr_with_region' else '', results).strip()
        if task == 'ocr_with_region': clean_results = re.sub(r'\n+', '\n', clean_results)
        
        original_image_pil = Image.fromarray((image[0].cpu().numpy() * 255.0).astype('uint8'))
        W, H = original_image_pil.size
        try: parsed_answer = self.processor.post_process_generation(results, task=task_prompt, image_size=(W, H))
        except: parsed_answer = None
        
        return clean_results, None, None, parsed_answer

    def _prepare_image_for_florence(self, image_tensor, max_size=720):
        if image_tensor.dim() == 4: image_np = image_tensor[0].cpu().numpy()
        else: image_np = image_tensor.cpu().numpy()
        
        if image_np.max() <= 1.0: image_np = (image_np * 255.0).astype('uint8')
        else: image_np = image_np.astype('uint8')
        
        image_pil = Image.fromarray(image_np)
        original_w, original_h = image_pil.size
        
        if max(original_w, original_h) > max_size:
            if original_w > original_h: new_w, new_h = max_size, int((original_h * max_size) / original_w)
            else: new_h, new_w = max_size, int((original_w * max_size) / original_h)
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            colored_print(f"🔍 Florence-2: Resized image from {original_w}x{original_h} to {new_w}x{new_h}", Colors.CYAN)
        else:
            colored_print(f"🔍 Florence-2: Using original image size {original_w}x{original_h}", Colors.CYAN)
        
        return image_pil

    def offload_model(self):
        if self.model is not None:
            self.model.to(unet_offload_device())
            colored_print("🔄 Florence-2 model offloaded", Colors.CYAN)
        soft_empty_cache()
        
    def clear_model(self):
        if self.model is not None:
            del self.model; self.model = None
            colored_print("🗑️ Florence-2 model cleared", Colors.CYAN)
        if self.processor is not None:
            del self.processor; self.processor = None
        self.dtype = None
        soft_empty_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def hash_seed(self, seed):
        return int(hashlib.sha256(str(seed).encode('utf-8')).hexdigest(), 16) % (2**32)

class TiledSampler:
    @staticmethod
    def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, cols, rows, tile_padding, mask_blur):
        device = get_torch_device()
        latent_samples = latent["samples"].clone().to(device)
        h_latent, w_latent = latent_samples.shape[2], latent_samples.shape[3]
        padding_latent, blur_latent = tile_padding // 8, mask_blur // 8
        base_tile_w, base_tile_h = w_latent // cols, h_latent // rows
        output_latent, blend_weights = torch.zeros_like(latent_samples), torch.zeros_like(latent_samples)
        pbar = comfy.utils.ProgressBar(rows * cols)
        colored_print(f"🧩 Processing {rows}x{cols} tiles with {tile_padding}px padding", Colors.CYAN)
        
        for r in range(rows):
            for c in range(cols):
                y_start, x_start = r * base_tile_h, c * base_tile_w
                py_start, px_start = max(0, y_start - padding_latent), max(0, x_start - padding_latent)
                py_end, px_end = min(h_latent, y_start + base_tile_h + padding_latent), min(w_latent, x_start + base_tile_w + padding_latent)
                tile_latent_slice = {"samples": latent_samples[:, :, py_start:py_end, px_start:px_end]}
                
                try:
                    processed_tile = common_ksampler(model, seed + r * cols + c, steps, cfg, sampler_name, scheduler, positive, negative, tile_latent_slice, denoise)[0]["samples"].to(device)
                    mask = TiledSampler._create_tile_mask(processed_tile.shape[3], processed_tile.shape[2], padding_latent, blur_latent, device, r, c, rows, cols)
                    output_latent[:, :, py_start:py_end, px_start:px_end] += processed_tile * mask
                    blend_weights[:, :, py_start:py_end, px_start:px_end] += mask
                except Exception as e:
                    colored_print(f"Error processing tile {r},{c}: {e}", Colors.RED)
                    continue
                finally:
                    if 'processed_tile' in locals(): del processed_tile
                    if 'mask' in locals(): del mask
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                pbar.update(1)
        
        output_latent /= blend_weights.clamp(min=1e-6)
        return ({"samples": torch.nan_to_num(output_latent)},)

    @staticmethod
    def _create_tile_mask(tile_w, tile_h, padding, blur, device, r, c, rows, cols):
        mask = torch.ones((1, 1, tile_h, tile_w), device=device)
        feather = max(1, blur)
        if padding > 0 and blur > 0:
            if r > 0: mask[:, :, :feather, :] *= torch.linspace(0, 1, feather, device=device).view(1, 1, feather, 1)
            if r < rows - 1: mask[:, :, -feather:, :] *= torch.linspace(1, 0, feather, device=device).view(1, 1, feather, 1)
            if c > 0: mask[:, :, :, :feather] *= torch.linspace(0, 1, feather, device=device).view(1, 1, 1, feather)
            if c < cols - 1: mask[:, :, :, -feather:] *= torch.linspace(1, 0, feather, device=device).view(1, 1, 1, feather)
        return mask

class FluxAIO_CRT:
    FUNCTION = "execute"
    CATEGORY = "CRT/WIP"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    MAX_CLIP_SEQUENCE_LENGTH = 77
    MAX_T5XXL_SEQUENCE_LENGTH = 512

    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    precision_options = ["auto", "fp32", "fp16", "bf16"]
    radial_blur_types = ["spin", "zoom"]
    glare_types = ["star_4", "star_6", "star_8", "anamorphic_h"]
    florence2_models = [
        'microsoft/Florence-2-base', 'microsoft/Florence-2-base-ft',
        'microsoft/Florence-2-large', 'microsoft/Florence-2-large-ft',
        'HuggingFaceM4/Florence-2-DocVQA', 'thwri/CogFlorence-2.1-Large',
        'thwri/CogFlorence-2.2-Large', 'gokaygokay/Florence-2-SD3-Captioner',
        'gokaygokay/Florence-2-Flux-Large', 'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
        'MiaoshouAI/Florence-2-large-PromptGen-v1.5', 'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
        'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
    ]
    florence2_tasks = [
        'region_caption', 'dense_region_caption', 'region_proposal', 'caption',
        'detailed_caption', 'more_detailed_caption', 'caption_to_phrase_grounding',
        'referring_expression_segmentation', 'ocr', 'ocr_with_region', 'docvqa',
        'prompt_gen_tags', 'prompt_gen_mixed_caption', 'prompt_gen_analyze',
        'prompt_gen_mixed_caption_plus'
    ]
    
    _input_types_cache = None

    @classmethod
    def INPUT_TYPES(cls):
        if cls._input_types_cache is not None:
            return cls._input_types_cache
        flux_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        text_encoder_models = folder_paths.get_filename_list("text_encoders")
        upscale_models = folder_paths.get_filename_list("upscale_models")
        style_models = folder_paths.get_filename_list("style_models")
        clip_vision_models = folder_paths.get_filename_list("clip_vision")
        cnet_models = ["None"] + folder_paths.get_filename_list("controlnet")
        bbox_files = folder_paths.get_filename_list("ultralytics_bbox")
        segm_files = folder_paths.get_filename_list("ultralytics_segm")

        resolution_presets = [
            "896x1152 (3:4 Portrait)", "768x1344 (9:16 Portrait)", "832x1216 (2:3 Portrait)", 
            "1024x1024 (1:1 Square)", "1152x896 (4:3 Landscape)", "1344x768 (16:9 Widescreen)", 
            "1216x832 (3:2 Landscape)", "1536x640 (21:9 CinemaScope)"
        ]
        
        def get_default_cnet_model(models):
            preferred_list = [m for m in models if "upscaler" in m.lower() and "flux" in m.lower()]
            if preferred_list: return preferred_list[0]
            preferred_list = [m for m in models if "tile" in m.lower()]
            if preferred_list: return preferred_list[0]
            if "Flux.1-dev-Controlnet-Upscaler.safetensors" in models:
                return "Flux.1-dev-Controlnet-Upscaler.safetensors"
            return "None"
        
        ordered_inputs = {
            "flux_model_name": (flux_models, {"default": "flux1-dev-fp8.safetensors"}),
            "vae_name": (vae_models, {"default": "ae.safetensors"}),
            "clip_l_name": (text_encoder_models, {"default": "clip_l.safetensors"}),
            "t5_name": (text_encoder_models, {"default": "t5xxl_fp16.safetensors"}),
            "flux_cnet_upscaler_model": (cnet_models, {"default": get_default_cnet_model(cnet_models)}),
            "face_bbox_model": (["bbox/" + x for x in bbox_files] + ["segm/" + x for x in segm_files], {"default": "bbox/face_yolov8m.pt" if bbox_files else "No models found"}),
            "face_segm_model": (["segm/" + x for x in segm_files], {"default": "segm/face_yolov8n-seg2_60.pt" if segm_files else "No models found"}),
            "enable_fast_preview": ("BOOLEAN", {"default": True}),
            "fast_preview_format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
            "fast_preview_quality": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
            "enable_sage_attention": ("BOOLEAN", {"default": True}),
            "sage_attention_mode": (["auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": "auto"}),
            "full_load": (["auto", "enabled", "disabled"], {"default": "auto"}), 
            "enable_torch_compile": ("BOOLEAN", {"default": False}), 
            "compile_backend": (["inductor", "aot_eager", "cudagraphs"], {"default": "inductor"}),
            "compile_mode": (["default", "max-autotune", "reduce-overhead"], {"default": "default"}),
            "compile_fullgraph": ("BOOLEAN", {"default": False}),
            "compile_dynamic": ("BOOLEAN", {"default": False}),
            "enable_teacache": ("BOOLEAN", {"default": True}),
            "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "teacache_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "teacache_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "teacache_2nd_rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "teacache_2nd_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "teacache_2nd_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "deis"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
            "enable_latent_injection": ("BOOLEAN", {"default": True}),
            "injection_point": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            "injection_seed_offset": ("INT", {"default": 1, "min": -100, "max": 100, "step": 1}),
            "injection_strength": ("FLOAT", {"default": 0.3, "min": -20.0, "max": 20.0, "step": 0.01}),
            "normalize_injected_noise": (["enable", "disable"], {"default": "enable"}),
            "resolution_preset": (resolution_presets, {"default": "832x1216 (2:3 Portrait)"}),
            "enable_img2img": ("BOOLEAN", {"default": False}),
            "img2img_denoise": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_lora_stack": ("BOOLEAN", {"default": False}),
            "lora_1_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_2_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_3_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_3_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_4_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_4_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_4_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_5_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_5_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_5_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_6_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
            "lora_6_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_6_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "flux_guidance": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            "dry_wet_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_florence2": ("BOOLEAN", {"default": False}),
            "florence2_model": (cls.florence2_models, {"default": "microsoft/Florence-2-base"}),
            "florence2_task": (cls.florence2_tasks, {"default": "caption"}),
            "florence2_precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            "florence2_attention": (["flash_attention_2", "sdpa", "eager"], {"default": "sdpa"}),
            "florence2_max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            "florence2_num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
            "florence2_do_sample": ("BOOLEAN", {"default": True}),
            "florence2_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            "enable_style_model": ("BOOLEAN", {"default": False}),
            "style_model_name": (style_models, {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision_name": (clip_vision_models, {"default": "sigclip_vision_patch14_384.safetensors"}),
            "style_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "strength_type": (["multiply", "attn_bias"], {"default": "multiply"}), 
            "crop": (["center", "none"], {"default": "none"}),
            "enable_tiling": ("BOOLEAN", {"default": False}), 
            "tiles": (["1x1", "2x2 (4)", "3x3 (9)", "4x4 (16)"], {"default": "2x2 (4)"}),
            "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}), 
            "mask_blur": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1}),
            "enable_lora_block_patcher": ("BOOLEAN", {"default": False}),
            "enable_upscale_with_model": ("BOOLEAN", {"default": True}), 
            "upscale_model_name": (upscale_models, {"default": "4x_foolhardy_Remacri.pth"}),
            "downscale_by": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05}), 
            "rescale_method": (cls.rescale_methods, {"default": "bicubic"}),
            "precision": (cls.precision_options, {"default": "bf16"}), 
            "batch_size_2ND": ("INT", {"default": 1, "min": 1, "max": 100}),
            "enable_2nd_pass": ("BOOLEAN", {"default": True}), 
            "denoise_2ND": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps_2ND": ("INT", {"default": 20, "min": 1, "max": 100}), 
            "seed_shift_2ND": ("INT", {"default": 1, "min": -100, "max": 100, "step": 1}),
            "enable_cnet_upscale": ("BOOLEAN", {"default": False}),
            "cnet_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
            "cnet_latent_scale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.01}),
            "cnet_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "cnet_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "enable_image_mix": ("BOOLEAN", {"default": False}),
            "image_mix_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_face_enhancement": ("BOOLEAN", {"default": False}),
            "face_initial_resize": ("INT", {"default": 4096, "min": 256, "max": 8192, "step": 64}),
            "face_resize_back": ("BOOLEAN", {"default": False}),
            "face_upscale_res": ("INT", {"default": 1536, "min": 512, "max": 4096, "step": 64}),
            "face_bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            "face_segm_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            "face_padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            "face_mask_expand": ("INT", {"default": 16, "min": -512, "max": 512, "step": 1}),
            "face_mask_blur": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 256.0, "step": 0.5}),
            "face_seed_shift": ("INT", {"default": 1, "min": -1000, "max": 1000, "step": 1}),
            "face_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            "face_cnet_end": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "face_cnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            "face_color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "enable_levels": ("BOOLEAN", {"default": False}), 
            "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}), 
            "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}), 
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "enable_color_wheels": ("BOOLEAN", {"default": False}), 
            "lift_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "lift_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}), 
            "lift_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "gamma_r": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}), 
            "gamma_g": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
            "gamma_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}), 
            "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}), 
            "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "enable_temp_tint": ("BOOLEAN", {"default": False}), 
            "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            "enable_sharpen": ("BOOLEAN", {"default": False}), 
            "sharpen_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.01}),
            "sharpen_radius": ("FLOAT", {"default": 1.85, "min": 0.1, "max": 5.0, "step": 0.1}), 
            "sharpen_threshold": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_small_glow": ("BOOLEAN", {"default": False}), 
            "small_glow_intensity": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 2.0, "step": 0.01}),
            "small_glow_radius": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1}), 
            "small_glow_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_large_glow": ("BOOLEAN", {"default": False}), 
            "large_glow_intensity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
            "large_glow_radius": ("FLOAT", {"default": 50.0, "min": 30.0, "max": 100.0, "step": 0.5}), 
            "large_glow_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_glare": ("BOOLEAN", {"default": False}), 
            "glare_type": (cls.glare_types, {"default": "star_4"}),
            "glare_intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 100.0, "step": 0.01}), 
            "glare_length": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 100.0, "step": 0.01}),
            "glare_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}), 
            "glare_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "glare_quality": ("INT", {"default": 16, "min": 4, "max": 32, "step": 4}), 
            "glare_ray_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            "enable_chromatic_aberration": ("BOOLEAN", {"default": False}), 
            "ca_strength": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
            "ca_edge_falloff": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01}), 
            "enable_ca_hue_shift": ("BOOLEAN", {"default": False}),
            "ca_hue_shift_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            "enable_vignette": ("BOOLEAN", {"default": False}), 
            "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            "vignette_radius": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 3.0, "step": 0.01}), 
            "vignette_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            "enable_radial_blur": ("BOOLEAN", {"default": False}), 
            "radial_blur_type": (cls.radial_blur_types, {"default": "spin"}),
            "radial_blur_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}), 
            "radial_blur_center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "radial_blur_center_y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}), 
            "radial_blur_falloff": ("FLOAT", {"default": 0.25, "min": 0.001, "max": 1.0, "step": 0.01}),
            "radial_blur_samples": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
            "enable_film_grain": ("BOOLEAN", {"default": False}), 
            "grain_intensity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.15, "step": 0.01}),
            "grain_size": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}), 
            "grain_color_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_lens_distortion": ("BOOLEAN", {"default": False}), 
            "barrel_distortion": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001}),
            "enable_save_image": ("BOOLEAN", {"default": True}), 
            "filename_prefix": ("STRING", {"default": "flux_aio/image"}),
        }

        inputs = {
            "required": ordered_inputs,
            "optional": {
                "seed": ("INT", {"forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "img2img_image": ("IMAGE",), 
                "style_image": ("IMAGE",), 
                "florence2_image": ("IMAGE",),
            },
            "hidden": {
                "lora_stack": ("STRING", {"multiline": True, "default": "[]"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "florence2_text_input": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        
        MAX_SINGLE_BLOCKS_COUNT, MAX_DOUBLE_BLOCKS_COUNT = 38, 19
        for i in range(MAX_SINGLE_BLOCKS_COUNT): 
            inputs["optional"][f"lora_block_{i}_weight"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})
        for i in range(MAX_DOUBLE_BLOCKS_COUNT):
            inputs["optional"][f"lora_block_{i}_double_weight"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})
            
        cls._input_types_cache = inputs
        return inputs

    
    def _create_fast_preview(self, image, format, quality):
        pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        with io.BytesIO() as buffered:
            pil_image.save(buffered, format=format.upper(), quality=quality)
            img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return {"previews": [{"source": img_base64, "format": format.lower()}]}

    def __init__(self):
        self.model_manager = ModelManager(max_cache_size=2)
        self.conditioning_cache = LRUCache(max_size=2)
        self.pass1_cache = LRUCache(max_size=2)
        self.second_pass_cache = LRUCache(max_size=2)
        self.lora_processed_cache = LRUCache(max_size=2)
        self.final_results_cache = LRUCache(max_size=2) 
        
        self.upscaler_loader = None
        self.image_upscaler = None
        self.lora_loader = LoraLoader()
        self.style_loader = StyleModelLoader()
        self.clip_vision_loader = CLIPVisionLoader()
        self.vision_encoder = CLIPVisionEncode()
        self.style_applier = StyleModelApply()
        self.florence2_processor = Florence2Processor()
        self.post_processor = PostProcessingPipeline()
        self.teacache_patcher = TeaCachePatcher()
        self.cnet_loader = None
        self.cnet_applier = None
        self.face_enhancement_processor = None

    def _lazy_init_helpers(self):
        if self.upscaler_loader is None:
            self.upscaler_loader = UpscaleModelLoader()
            self.image_upscaler = ImageUpscaleWithModel()
        if self.cnet_loader is None:
            try:
                self.cnet_loader = ControlNetLoader()
                self.cnet_applier = ControlNetApplyAdvanced()
                colored_print("✅ Initialized ControlNet helpers.", Colors.GREEN)
            except Exception as e:
                colored_print(f"⚠️ Could not initialize ControlNet helpers: {e}", Colors.YELLOW)
        if self.face_enhancement_processor is None:
            self.face_enhancement_processor = FaceEnhancementProcessor(self.teacache_patcher)
            colored_print("✅ Initialized Face Enhancement helper.", Colors.GREEN)

    def _create_second_pass_cache_key(self, kwargs):
        keys = [
            "seed", "steps", "sampler_name", "scheduler", "resolution_preset", "img2img_denoise",
            "enable_2nd_pass", "denoise_2ND", "steps_2ND", "seed_shift_2ND",
            "enable_cnet_upscale", "cnet_model_name", "cnet_strength", "cnet_latent_scale_by", "cnet_start_percent", "cnet_end_percent",
            "enable_upscale_with_model", "upscale_model_name", "downscale_by", "rescale_method",
            "enable_tiling", "tiles", "tile_padding", "mask_blur"
        ]
        lora_config = {k: v for k, v in kwargs.items() if k.startswith('lora_')}
        key_dict = {k: kwargs.get(k) for k in keys}
        key_dict.update(lora_config)
        key_dict.update({"positive_prompt": kwargs.get("positive_prompt"), "negative_prompt": kwargs.get("negative_prompt")})
        return json.dumps(key_dict, sort_keys=True)
            
    def _create_cache_key(self, kwargs, cache_type):
        if cache_type == "models":
            keys = ["flux_model_name", "vae_name", "clip_l_name", "t5_name", "style_model_name", "clip_vision_name"]
        elif cache_type == "conditioning":
            keys = ["positive_prompt", "negative_prompt", "flux_guidance", "multiplier", "dry_wet_mix"]
        elif cache_type == "first_pass":
            keys = [
                "seed", "steps", "sampler_name", "scheduler", "resolution_preset",
                "enable_img2img", "img2img_denoise",
                "enable_latent_injection", "injection_point", "injection_seed_offset", "injection_strength", "normalize_injected_noise"
            ]
        elif cache_type == "final_result":
            post_fx_keys = {
                'enable_image_mix', 'image_mix_factor', 'enable_levels', 'exposure', 'gamma', 'brightness', 'contrast', 'saturation', 
                'vibrance', 'enable_color_wheels', 'lift_r', 'lift_g', 'lift_b', 'gamma_r', 'gamma_g', 'gamma_b', 
                'gain_r', 'gain_g', 'gain_b', 'enable_temp_tint', 'temperature', 'tint', 'enable_sharpen', 
                'sharpen_strength', 'sharpen_radius', 'sharpen_threshold', 'enable_small_glow', 'small_glow_intensity', 
                'small_glow_radius', 'small_glow_threshold', 'enable_large_glow', 'large_glow_intensity', 
                'large_glow_radius', 'large_glow_threshold', 'enable_glare', 'glare_type', 'glare_intensity', 
                'glare_length', 'glare_angle', 'glare_threshold', 'glare_quality', 'glare_ray_width', 
                'enable_chromatic_aberration', 'ca_strength', 'ca_edge_falloff', 'enable_ca_hue_shift', 
                'ca_hue_shift_degrees', 'enable_vignette', 'vignette_strength', 'vignette_radius', 'vignette_softness', 
                'enable_radial_blur', 'radial_blur_type', 'radial_blur_strength', 'radial_blur_center_x', 
                'radial_blur_center_y', 'radial_blur_falloff', 'radial_blur_samples', 'enable_film_grain', 
                'grain_intensity', 'grain_size', 'grain_color_amount', 'enable_lens_distortion', 'barrel_distortion'
            }
            return json.dumps({
                k: v for k, v in kwargs.items() 
                if k not in post_fx_keys and not isinstance(v, (torch.Tensor, list, dict))
            }, sort_keys=True)
        else:
            return None
        return json.dumps({k: kwargs.get(k) for k in keys}, sort_keys=True)
            
    def _create_lora_cache_key(self, kwargs):
        lora_config = {k: v for k, v in kwargs.items() if k.startswith('lora_')}
        lora_config.update({
            "flux_model_name": kwargs.get("flux_model_name"), "clip_l_name": kwargs.get("clip_l_name"), "t5_name": kwargs.get("t5_name"),
            "enable_lora_stack": kwargs.get("enable_lora_stack", False), "enable_lora_block_patcher": kwargs.get("enable_lora_block_patcher", False),
            "enable_sage_attention": kwargs.get("enable_sage_attention"), "sage_attention_mode": kwargs.get("sage_attention_mode"),
            "full_load": kwargs.get("full_load"), "enable_torch_compile": kwargs.get("enable_torch_compile"),
            "compile_backend": kwargs.get("compile_backend"), "compile_mode": kwargs.get("compile_mode"),
            "compile_fullgraph": kwargs.get("compile_fullgraph"), "compile_dynamic": kwargs.get("compile_dynamic"),
            "enable_teacache": kwargs.get("enable_teacache"), "teacache_rel_l1_thresh": kwargs.get("teacache_rel_l1_thresh"),
            "teacache_start_percent": kwargs.get("teacache_start_percent"), "teacache_end_percent": kwargs.get("teacache_end_percent"),
        })
        return json.dumps(lora_config, sort_keys=True)         

    def execute(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is None:
            seed = 1
            colored_print(f"⚠️ Seed not connected, using default value: {seed}", Colors.YELLOW)
        kwargs["seed"] = seed

        positive_prompt = kwargs.get("positive_prompt") or ""
        kwargs["positive_prompt"] = positive_prompt
        negative_prompt = kwargs.get("negative_prompt") or ""
        kwargs["negative_prompt"] = negative_prompt
        
        fallback_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
        ui_output = {"images": []}
        
        try:
            self._lazy_init_helpers()
            model, vae, clip, style_model, clip_vision = self.model_manager.load_models(kwargs)
            processed_model, processed_clip = self._process_model_pipeline(model, clip, kwargs)
            positive, negative = self._prepare_conditioning(processed_clip, **kwargs)
            width, height = self._calculate_dimensions_from_kwargs(kwargs)
            final_cache_key = self._create_cache_key(kwargs, "final_result")
            cached_data = self.final_results_cache.get(final_cache_key)

            if cached_data is not None and not any(kwargs.get(k) is not None for k in ["img2img_image", "style_image", "florence2_image"]):
                colored_print("🚀 [Cache Hit] Reusing pre-composited image.", Colors.GREEN)
                image_before_final_fx, image_to_process = cached_data
            else:
                colored_print("\n🔥 [Cache Miss] Starting new computation", Colors.HEADER)
                final_latent = self._execute_first_pass(processed_model, (positive, negative), vae, width, height, kwargs)
                image_to_process = VAEDecode().decode(vae, final_latent)[0]
                
                if kwargs["enable_2nd_pass"]:
                    second_pass_cache_key = self._create_second_pass_cache_key(kwargs)
                    cached_2nd_pass = self.second_pass_cache.get(second_pass_cache_key)
                    if cached_2nd_pass is not None:
                         image_to_process = cached_2nd_pass
                    else:
                        second_pass_base_image = VAEDecode().decode(vae, final_latent)[0]
                        if kwargs.get("enable_cnet_upscale") and kwargs.get("flux_cnet_upscaler_model", "None") != "None":
                            control_net = self.cnet_loader.load_controlnet(kwargs["flux_cnet_upscaler_model"])[0]
                            scale_by = kwargs["cnet_latent_scale_by"]
                            upscaled_latent = final_latent.copy()
                            w_up, h_up = round(final_latent["samples"].shape[-1] * scale_by), round(final_latent["samples"].shape[-2] * scale_by)
                            upscaled_latent["samples"] = comfy.utils.common_upscale(final_latent["samples"], w_up, h_up, "nearest-exact", "disabled")
                            positive_cnet, negative_cnet = self.cnet_applier.apply_controlnet(positive, negative, control_net, second_pass_base_image, kwargs["cnet_strength"], kwargs["cnet_start_percent"], kwargs["cnet_end_percent"], vae)
                            model_for_pass2 = processed_model.clone()
                            if kwargs.get("enable_teacache"): model_for_pass2 = self.teacache_patcher.apply(model_for_pass2, **kwargs, is_second_pass=True)
                            latent_pass2 = common_ksampler(model_for_pass2, seed + kwargs["seed_shift_2ND"], kwargs["steps_2ND"], 1.0, kwargs["sampler_name"], kwargs["scheduler"], positive_cnet, negative_cnet, upscaled_latent, 1.0)[0]
                            image_to_process = VAEDecode().decode(vae, latent_pass2)[0]
                            if 'control_net' in locals(): control_net.cleanup()
                        
                        elif kwargs.get("enable_upscale_with_model"):
                           upscale_model = self.upscaler_loader.load_model(kwargs["upscale_model_name"])[0]
                           upscaled_image = self._perform_upscale(upscale_model, second_pass_base_image, kwargs["downscale_by"], kwargs["rescale_method"], kwargs["precision"], kwargs["batch_size_2ND"])
                           upscaled_latent = VAEEncode().encode(vae, upscaled_image)[0]
                           if upscaled_latent["samples"].shape[1] == 4:
                                b, _, h_latent, w_latent = upscaled_latent["samples"].shape
                                padding = torch.zeros((b, 12, h_latent, w_latent), device=upscaled_latent["samples"].device, dtype=upscaled_latent["samples"].dtype)
                                upscaled_latent["samples"] = torch.cat((upscaled_latent["samples"], padding), dim=1)
                           model_for_pass2 = processed_model.clone()
                           if kwargs.get("enable_teacache"): model_for_pass2 = self.teacache_patcher.apply(model_for_pass2, **kwargs, is_second_pass=True)
                           
                           cols, rows = map(int, kwargs['tiles'].split(' ')[0].split('x'))
                           if kwargs.get("enable_tiling", False) and (cols > 1 or rows > 1):
                               latent_pass2 = TiledSampler.sample(model_for_pass2, seed + kwargs["seed_shift_2ND"], kwargs["steps_2ND"], 1.0, kwargs["sampler_name"], kwargs["scheduler"], positive, negative, upscaled_latent, kwargs["denoise_2ND"], cols, rows, kwargs["tile_padding"], kwargs["mask_blur"])[0]
                           else:
                               latent_pass2 = common_ksampler(model_for_pass2, seed + kwargs["seed_shift_2ND"], kwargs["steps_2ND"], 1.0, kwargs["sampler_name"], kwargs["scheduler"], positive, negative, upscaled_latent, kwargs["denoise_2ND"])[0]
                           image_to_process = VAEDecode().decode(vae, latent_pass2)[0]
                           if 'upscale_model' in locals(): upscale_model.to(unet_offload_device())
                        self.second_pass_cache.put(second_pass_cache_key, image_to_process.clone())
                image_before_final_fx = image_to_process.clone()

                if kwargs.get("enable_face_enhancement"):
                    colored_print("🎭 Applying Face Enhancement (on miss)...", Colors.GREEN)
                    image_to_process = self.face_enhancement_processor.enhance_face(
                        image=image_to_process, model=processed_model, positive=positive, negative=negative, vae=vae, **kwargs
                    )
                
                self.final_results_cache.put(final_cache_key, (image_before_final_fx, image_to_process.clone()))
            
            if kwargs.get("enable_face_enhancement") and kwargs.get("enable_image_mix"):
                colored_print("🎨 Blending Face Enhancement result...", Colors.BLUE)
                h, w = image_to_process.shape[1:3]
                resized_before_image = comfy.utils.common_upscale(image_before_final_fx.permute(0, 3, 1, 2), w, h, "bicubic", "center").permute(0, 2, 3, 1)
                mix_factor = kwargs["image_mix_factor"]
                image_to_process = torch.clamp((resized_before_image * (1.0 - mix_factor)) + (image_to_process * mix_factor), 0, 1)
            
            final_image = self.post_processor.apply_effects(image_to_process, **kwargs)
            ui_previews_for_gallery = []
            if kwargs.get("enable_fast_preview", False):
                ui_output.update(self._create_fast_preview(final_image, kwargs["fast_preview_format"], kwargs["fast_preview_quality"]))
            else:
                temp_previewer = PreviewImage()
                ui_previews_for_gallery.extend(temp_previewer.save_images(images=final_image, filename_prefix="FluxAIO_Final")["ui"]["images"])
            
            if kwargs.get("enable_save_image"):
                save_results = SaveImage().save_images(images=final_image, filename_prefix=kwargs["filename_prefix"])
                if "images" in save_results.get("ui", {}): ui_previews_for_gallery.extend(save_results["ui"]["images"])
            
            ui_output["images"] = ui_previews_for_gallery
            final_output = {"ui": ui_output, "result": (final_image,)}
            
            colored_print("✅ Process completed successfully!", Colors.GREEN)
            return final_output
            
        except InterruptProcessingException:
            colored_print("\n🛑 Execution Interrupted by User.", Colors.YELLOW)
            return {"ui": ui_output, "result": (fallback_image,)}
        
        except Exception as e:
            colored_print(f"❌ FATAL ERROR in execute: {e}", Colors.RED)
            import traceback
            traceback.print_exc()
            return {"ui": ui_output, "result": (fallback_image,)}
            
        finally:
            if kwargs.get("enable_sage_attention"):
                SageAttentionPatcher.unpatch()
            soft_empty_cache()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    def _validate_and_fix_parameters(self, kwargs):
        fixed_kwargs = kwargs.copy()
        if not hasattr(self, '_all_inputs_def'):
            input_defs = self.INPUT_TYPES()
            all_inputs = input_defs.get('required', {})
            all_inputs.update(input_defs.get('optional', {}))
            all_inputs.update(input_defs.get('hidden', {}))
            self._all_inputs_def = all_inputs
        all_inputs = self._all_inputs_def
        for key, value in fixed_kwargs.items():
            if key not in all_inputs: continue
            param_def = all_inputs[key]
            param_type, param_config = param_def[0], param_def[1] if len(param_def) > 1 else {}
            default_val = param_config.get('default')
            if param_type in ("FLOAT", "INT") and isinstance(value, str) and value.strip().lower() in ['none', '']:
                if value != default_val: colored_print(f"🔧 Fixing '{key}': Converted string '{value}' to default '{default_val}'.", Colors.CYAN)
                fixed_kwargs[key] = default_val
                continue
            if param_type == "FLOAT":
                try: new_val = float(value); fixed_kwargs[key] = new_val
                except (ValueError, TypeError):
                    if value != default_val: colored_print(f"⚠️ Warning: Could not convert '{value}' to float for '{key}'. Using default '{default_val}'.", Colors.YELLOW)
                    fixed_kwargs[key] = default_val
            elif param_type == "INT":
                try: new_val = int(float(value)); fixed_kwargs[key] = new_val
                except (ValueError, TypeError):
                    if value != default_val: colored_print(f"⚠️ Warning: Could not convert '{value}' to int for '{key}'. Using default '{default_val}'.", Colors.YELLOW)
                    fixed_kwargs[key] = default_val
            elif isinstance(param_type, list) and not isinstance(param_type[0], (list, dict)):
                if value not in param_type:
                    if value != default_val: colored_print(f"⚠️ Warning: Value '{value}' for '{key}' not in list. Using default '{default_val}'.", Colors.YELLOW)
                    fixed_kwargs[key] = default_val
        return fixed_kwargs
                
    def _prepare_conditioning(self, processed_clip, **kwargs):
        conditioning_keys = ["positive_prompt", "negative_prompt", "flux_guidance", "multiplier", "dry_wet_mix"]
        if kwargs.get("enable_style_model"):
            conditioning_keys.extend(["enable_style_model", "style_model_name", "clip_vision_name", "style_strength", "strength_type", "crop"])

        conditioning_cache_key = json.dumps({k: kwargs.get(k) for k in conditioning_keys}, sort_keys=True)
        if (cached_conditioning := self.conditioning_cache.get(conditioning_cache_key)) and kwargs.get("style_image") is None:
            colored_print("🎨 [Cache Hit] Reusing conditioning.", Colors.GREEN)
            return cached_conditioning
            
        colored_print("🎨 [Cache Miss] Creating conditioning...", Colors.BLUE)
        positive_cond = self._create_semantic_conditioning(processed_clip, kwargs.get("positive_prompt", ""), **kwargs)
        negative_cond = self._create_semantic_conditioning(processed_clip, kwargs.get("negative_prompt", ""), **kwargs)
        
        if kwargs.get("enable_style_model"):
            colored_print("🎨 Applying Redux Style Model...", Colors.CYAN)
            
            style_model = self.style_loader.load_style_model(kwargs["style_model_name"])[0]
            clip_vision = self.clip_vision_loader.load_clip(kwargs["clip_vision_name"])[0]
            
            crop_setting = kwargs.get("crop", "center")
            if kwargs.get("style_image") is not None:
                style_image = kwargs.get("style_image")
                colored_print("... using provided style image.", Colors.CYAN)
                clip_vision_output = self.vision_encoder.encode(clip_vision, style_image, crop_setting)[0]
            else:
                colored_print("... using model's default style.", Colors.CYAN)
                width, height = self._calculate_dimensions_from_kwargs(kwargs)
                neutral_image = torch.zeros([1, height, width, 3], dtype=torch.float32, device="cpu")
                clip_vision_output = self.vision_encoder.encode(clip_vision, neutral_image, crop_setting)[0]

            positive_cond = self.style_applier.apply_stylemodel(
                positive_cond,
                style_model,
                clip_vision_output,
                kwargs.get("style_strength", 1.0),
                kwargs.get("strength_type", "multiply")
            )[0]
            
            del style_model, clip_vision, clip_vision_output
            soft_empty_cache()

        result = (positive_cond, negative_cond)
        if kwargs.get("style_image") is None:
            self.conditioning_cache.put(conditioning_cache_key, result)
            
        return result

    def _execute_first_pass(self, model, conditioning, vae, width, height, kwargs):
        pass1_cache_key = self._create_cache_key(kwargs, "first_pass")
        if kwargs.get("img2img_image") is None and (cached_latent := self.pass1_cache.get(pass1_cache_key)):
            colored_print("🚀 [Cache Hit] Reusing first pass latent.", Colors.GREEN)
            return cached_latent
        colored_print("🔥 [Cache Miss] Executing first pass generation...", Colors.GREEN)
        positive, negative = conditioning
        if kwargs["enable_img2img"] and kwargs.get("img2img_image") is not None:
            initial_latent = self._prepare_img2img_latent(kwargs["img2img_image"], vae, width, height)
            denoise = kwargs["img2img_denoise"]
        else:
            initial_latent = EmptyLatentImage().generate(width, height, 1)[0]
            denoise = 1.0
        if initial_latent["samples"].shape[1] == 4:
            b, _, h, w = initial_latent["samples"].shape
            padding = torch.zeros((b, 12, h, w), device=initial_latent["samples"].device, dtype=initial_latent["samples"].dtype)
            initial_latent["samples"] = torch.cat((initial_latent["samples"], padding), dim=1)
        if kwargs.get("enable_latent_injection", False):
            seed, steps, sampler_name, scheduler = kwargs["seed"], kwargs["steps"], kwargs["sampler_name"], kwargs["scheduler"]
            injection_point, injection_seed_offset, injection_strength, normalize_injected_noise = kwargs["injection_point"], kwargs["injection_seed_offset"], kwargs["injection_strength"], kwargs["normalize_injected_noise"]
            if injection_point <= 0.0 or injection_point >= 1.0:
                final_latent = common_ksampler(model, seed, steps, 1.0, sampler_name, scheduler, positive, negative, initial_latent, denoise)[0]
            else:
                first_stage_steps = max(1, int(steps * injection_point))
                latent_after_stage1 = common_ksampler(model, seed, steps, 1.0, sampler_name, scheduler, positive, negative, initial_latent, denoise=denoise, start_step=0, last_step=first_stage_steps, force_full_denoise=False)[0]
                torch.manual_seed(seed + injection_seed_offset)
                new_noise = torch.randn_like(latent_after_stage1["samples"])
                if normalize_injected_noise == "enable" and (std := latent_after_stage1["samples"].std()) > 1e-6:
                    new_noise = new_noise * std + latent_after_stage1["samples"].mean()
                latent_after_stage1["samples"] += new_noise * injection_strength
                final_latent = common_ksampler(model, seed, steps, 1.0, sampler_name, scheduler, positive, negative, latent_after_stage1, denoise=1.0, disable_noise=True, start_step=first_stage_steps, last_step=steps, force_full_denoise=True)[0]
        else:
            final_latent = common_ksampler(model, kwargs["seed"], kwargs["steps"], 1.0, kwargs["sampler_name"], kwargs["scheduler"], positive, negative, initial_latent, denoise)[0]
        if kwargs.get("img2img_image") is None: self.pass1_cache.put(pass1_cache_key, final_latent)
        return final_latent

    def _calculate_dimensions_from_kwargs(self, kwargs):
        if kwargs["enable_img2img"] and kwargs.get("img2img_image") is not None:
            img = kwargs["img2img_image"]
            h, w = img.shape[1], img.shape[2]
            side = max(h, w)
            scale = 1216 / side if side > 1216 else 1.0
            width, height = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
            colored_print(f"📐 Scaled Img2Img dimensions to: {width}x{height}", Colors.CYAN)
            return (width, height)
        preset_str = kwargs["resolution_preset"].split(' ')[0]
        try: return tuple(map(int, preset_str.split('x')))
        except: return (1024, 1024)

    def _process_model_pipeline(self, model, clip, kwargs):
        lora_cache_key = self._create_lora_cache_key(kwargs)
        if cached_models := self.lora_processed_cache.get(lora_cache_key):
            colored_print("🎯 [Cache Hit] Reusing processed models.", Colors.GREEN)
            return cached_models
        colored_print("🎯 [Cache Miss] Starting model processing pipeline...", Colors.BLUE)
        processed_model, processed_clip = model.clone(), clip.clone()
        if kwargs.get("enable_lora_stack", False):
            colored_print("🎯 Applying LoRAs...", Colors.CYAN)
            for i in range(1, 7):
                if (lora_name := kwargs.get(f"lora_{i}_name", "None")) and lora_name != "None":
                    processed_model, processed_clip = self.lora_loader.load_lora(processed_model, processed_clip, lora_name, kwargs.get(f"lora_{i}_strength", 1.0), kwargs.get(f"lora_{i}_clip_strength", 1.0))
        if kwargs.get("enable_lora_block_patcher"): processed_model = self._apply_lora_block_patching(processed_model, kwargs)
        result = (processed_model, processed_clip)
        self.lora_processed_cache.put(lora_cache_key, result)
        return result
    
    def _create_semantic_conditioning(self, clip, text, **kwargs):
        device = get_torch_device()
        text = str(text).strip() or " "
        try:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled, "flux_guidance": kwargs.get("flux_guidance", 2.7)}]]
        except Exception as e:
            colored_print(f"⚠️ Warning: Conditioning failed: {e}. Creating fallback.", Colors.RED)
            return [[torch.zeros((1, 1, 4096), device=device), {"pooled_output": torch.zeros((1, 2048), device=device)}]]

    def _prepare_img2img_latent(self, i, v, w, h):
        i_p = comfy.utils.common_upscale(i.permute(0, 3, 1, 2), w, h, "bicubic", "center").permute(0, 2, 3, 1)
        return VAEEncode().encode(v, i_p)[0]

    def _apply_lora_block_patching(self, m, k):
        s1 = {i: k.get(f"lora_block_{i}_weight", 1.0) for i in range(38)}
        s2 = {i: k.get(f"lora_block_{i}_double_weight", 1.0) for i in range(19)}
        if not (any(abs(s - 1.0) > 1e-5 for s in s1.values()) or any(abs(s - 1.0) > 1e-5 for s in s2.values())): return m
        colored_print("🎯 Applying Per-Block LoRA patching...", Colors.CYAN)
        p = re.compile(r"(single_blocks|double_blocks)\.(\d+)")
        if hasattr(m, 'patches') and m.patches:
            for key, pl in m.patches.items():
                if match := p.search(key):
                    bt, bi = match.groups()
                    bi, sc = int(bi), 1.0
                    if bt == "single_blocks" and bi < 38: sc = s1.get(bi, 1.0)
                    elif bt == "double_blocks" and bi < 19: sc = s2.get(bi, 1.0)
                    if abs(sc - 1.0) > 1e-5:
                        for i, op in enumerate(pl):
                            if isinstance(op, tuple) and len(op) > 0: pl[i] = tuple([op[0] * sc] + list(op[1:]))
            m.add_patches({}, 1.0, 1.0)
        return m

    def _perform_upscale(self, um, i, dsb, rsm, p, bs):
        device, offload_device = get_torch_device(), unet_offload_device()
        dt = torch.bfloat16 if p == "auto" and mm.is_device_mps(device) else (torch.float16 if p == "fp16" or p == "auto" else (torch.bfloat16 if p == "bf16" else torch.float32))
        with self.model_manager.device_context(um, device, offload_device):
            um = um.to(dt)
            ul = []
            for batch in torch.split(i, bs):
                b = batch.to(dt)
                with torch.no_grad(), torch.autocast(device.type, dt): ub = self.image_upscaler.upscale(um, b)[0]
                if dsb < 1.0:
                    h, w = ub.shape[1:3]
                    ub = comfy.utils.common_upscale(ub.permute(0, 3, 1, 2), round(w * dsb), round(h * dsb), rsm, "disabled").permute(0, 2, 3, 1)
                ul.append(ub.to(i.dtype))
            return torch.cat(ul).to(i.device)

NODE_CLASS_MAPPINGS = {"FluxAIO_CRT": FluxAIO_CRT}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxAIO_CRT": "FLUX All-In-One (CRT)"}
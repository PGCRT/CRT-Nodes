import os
import logging
import urllib.error
import urllib.request

import comfy.sd
import comfy.utils
import folder_paths
import nodes
import torch
from comfy.cli_args import args
from comfy.ldm.modules.attention import attention_pytorch, wrap_attn


MODEL_DEFS = {
    "flux2_model": {
        "filename": "OB_FK.safetensors",
        "url": "https://huggingface.co/PGCRYPT/OB_FK/resolve/main/OB_FK.safetensors",
        "folder": "diffusion_models",
    },
    "zit_model": {
        "filename": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
        "url": "https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
        "folder": "diffusion_models",
    },
    "flux2_clip": {
        "filename": "qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "folder": "text_encoders",
    },
    "zit_clip": {
        "filename": "qwen_3_4b_fp8_mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "folder": "text_encoders",
    },
    "flux2_vae": {
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        "folder": "vae",
    },
    "zit_vae": {
        "filename": "ae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
        "folder": "vae",
    },
    "flux2_lora_hdr360": {
        "filename": "Klein_9B - HDRI_360_panoramic.safetensors",
        "url": "https://huggingface.co/PGCRYPT/Flux2Klein_9B-HDRI/resolve/main/Klein_9B%20-%20HDRI_360_panoramic.safetensors",
        "folder": "loras",
    },
}


def _ensure_download(key):
    spec = MODEL_DEFS[key]
    target = os.path.join(folder_paths.models_dir, spec["folder"], spec["filename"])
    if os.path.isfile(target):
        return spec["filename"]

    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp_path = f"{target}.part"
    print(f"[CRT AutoLoader] Downloading {spec['filename']}...")
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "CRT-AutoLoader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(temp_path, target)
    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Download failed for {spec['filename']}: HTTP {e.code} {e.reason}") from e
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    print(f"[CRT AutoLoader] Download complete: {spec['filename']}")
    return spec["filename"]


SAGE_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]


def _get_sage_func(sage_attention):
    from sageattention import sageattn

    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    else:
        from sageattn3 import sageattn3_blackwell

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(lambda t: t.view(b, -1, heads, dim_head), (q, k, v))
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out

    return attention_sage


def _load_diffusion_model_local(unet_path, model_options=None, extra_state_dict=None):
    model_options = {} if model_options is None else dict(model_options)
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

    if extra_state_dict:
        extra_sd = comfy.utils.load_torch_file(extra_state_dict)
        sd.update(extra_sd)
        del extra_sd
        diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd)
        sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=False)

    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
    model.cached_patcher_init = (_load_diffusion_model_local, (unet_path, model_options, extra_state_dict))
    return model


def _load_model_with_kj_features(model_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict=None):
    dtype_map = {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    model_options = {}
    dtype = dtype_map.get(weight_dtype)
    if dtype:
        model_options["dtype"] = dtype
    if weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True

    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enable_fp16_accumulation)

    if patch_cublaslinear:
        args.fast.add("cublas_ops")
    else:
        args.fast.discard("cublas_ops")

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
    model = _load_diffusion_model_local(unet_path, model_options=model_options, extra_state_dict=extra_state_dict)

    compute = dtype_map.get(compute_dtype)
    if compute:
        model.set_model_compute_dtype(compute)
        model.force_cast_weights = False

    if sage_attention != "disabled":
        new_attention = _get_sage_func(sage_attention)

        def attention_override(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        model.model_options["transformer_options"]["optimized_attention_override"] = attention_override

    logging.info("Loaded %s with CRT local KJ-compatible loader", model_name)
    return model


class Flux2KleinModelAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"], {"default": "default"}),
                "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default"}),
                "patch_cublaslinear": ("BOOLEAN", {"default": True}),
                "sage_attention": (SAGE_MODES, {"default": "auto"}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_state_dict": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "CRT/Flux2"

    def load(self, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict=None):
        name = _ensure_download("flux2_model")
        model = _load_model_with_kj_features(name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict)
        return (model,)


class ZImageTurboModelAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"], {"default": "default"}),
                "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default"}),
                "patch_cublaslinear": ("BOOLEAN", {"default": True}),
                "sage_attention": (SAGE_MODES, {"default": "auto"}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_state_dict": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "CRT/ZIT"

    def load(self, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict=None):
        name = _ensure_download("zit_model")
        model = _load_model_with_kj_features(name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict)
        return (model,)


class Flux2KleinClipAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "CRT/Flux2"

    def load(self):
        name = _ensure_download("flux2_clip")
        return nodes.CLIPLoader().load_clip(name, "flux2", "default")


class ZImageTurboClipAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "CRT/ZIT"

    def load(self):
        name = _ensure_download("zit_clip")
        return nodes.CLIPLoader().load_clip(name, "qwen_image", "default")


class Flux2KleinVAEAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "CRT/Flux2"

    def load(self):
        name = _ensure_download("flux2_vae")
        return nodes.VAELoader().load_vae(name)


class ZImageTurboVAEAutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "CRT/ZIT"

    def load(self):
        name = _ensure_download("zit_vae")
        return nodes.VAELoader().load_vae(name)


class Flux2KleinLoRAHDR360AutoDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "CRT/Flux2"

    def load_lora(self, model, strength_model=1.0):
        name = _ensure_download("flux2_lora_hdr360")
        path = folder_paths.get_full_path_or_raise("loras", name)
        if float(strength_model) == 0:
            return (model,)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, float(strength_model), 0)
        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinModelAutoDownload": Flux2KleinModelAutoDownload,
    "ZImageTurboModelAutoDownload": ZImageTurboModelAutoDownload,
    "Flux2KleinClipAutoDownload": Flux2KleinClipAutoDownload,
    "ZImageTurboClipAutoDownload": ZImageTurboClipAutoDownload,
    "Flux2KleinVAEAutoDownload": Flux2KleinVAEAutoDownload,
    "ZImageTurboVAEAutoDownload": ZImageTurboVAEAutoDownload,
    "Flux2KleinLoRAHDR360AutoDownload": Flux2KleinLoRAHDR360AutoDownload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinModelAutoDownload": "Flux2Klein Model download (CRT)",
    "ZImageTurboModelAutoDownload": "ZImageTurbo Model download (CRT)",
    "Flux2KleinClipAutoDownload": "Flux2Klein Clip download (CRT)",
    "ZImageTurboClipAutoDownload": "ZImageTurbo Clip download (CRT)",
    "Flux2KleinVAEAutoDownload": "Flux2Klein VAE download (CRT)",
    "ZImageTurboVAEAutoDownload": "ZImageTurbo VAE download (CRT)",
    "Flux2KleinLoRAHDR360AutoDownload": "Flux2Klein LoRA HDR 360 download (CRT)",
}

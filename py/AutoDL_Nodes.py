import json
import logging
import os
import tempfile
import urllib.request

import comfy.model_management as model_management
import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy.cli_args import args
from comfy.ldm.modules.attention import attention_pytorch, wrap_attn


TAG = "crt-autodl"
SAGE_ATTENTION_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]


MODELS = {
    "ltx23_model": {
        "folder": "diffusion_models",
        "filename": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    },
    "ltx23_audio_vae": {
        "folder": "vae",
        "filename": "LTX23_audio_vae_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    },
    "ltx23_video_vae": {
        "folder": "vae",
        "filename": "LTX23_video_vae_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    },
    "ltx23_projection": {
        "folder": "text_encoders",
        "filename": "ltx-2.3_text_projection_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    },
    "ltx23_gemma3": {
        "folder": "text_encoders",
        "filename": "gemma_3_12B_it_fp4_mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    },
    "ltx23_upscaler": {
        "folder": "latent_upscale_models",
        "filename": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    },
    "ltx23_ic_lora": {
        "folder": "loras",
        "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    },
    "ltx23_outpaint_lora": {
        "folder": "loras",
        "filename": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "url": "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
    },
    "zimage_model": {
        "folder": "diffusion_models",
        "filename": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
        "url": "https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
    },
    "zimage_vae": {
        "folder": "vae",
        "filename": "ae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    },
    "zimage_clip": {
        "folder": "text_encoders",
        "filename": "qwen_3_4b_fp8_mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
    },
    "fluxklein_vae": {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
    "fluxklein_model": {
        "folder": "diffusion_models",
        "filename": "flux-2-klein-9b-fp8.safetensors",
        "url": "https://huggingface.co/PGCRYPT/OB_FK/resolve/main/OB_FK.safetensors",
        "alternate_filenames": ["OB_FK.safetensors"],
    },
    "fluxklein_clip": {
        "folder": "text_encoders",
        "filename": "qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
    },
    "fluxklein_hdri_lora": {
        "folder": "loras",
        "filename": "Klein_9B - HDRI_360_panoramic.safetensors",
        "url": "https://huggingface.co/PGCRYPT/Flux2Klein_9B-HDRI/resolve/main/Klein_9B%20-%20HDRI_360_panoramic.safetensors",
    },
    "ernie_turbo_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-turbo-fp8.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-turbo-fp8.safetensors",
    },
    "ernie_turbo_nvfp4_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-turbo-nvfp4.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-turbo-nvfp4.safetensors",
    },
    "ernie_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-fp8.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-fp8.safetensors",
    },
    "ernie_turbo_clip": {
        "folder": "text_encoders",
        "filename": "ministral-3-3b.safetensors",
        "url": "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/82d237fcf02a10b75154717487d07a724a25dc5b/text_encoders/ministral-3-3b.safetensors",
    },
    "ernie_turbo_vae": {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
}


def _model_dir(folder):
    return os.path.join(folder_paths.models_dir, folder)


def _target_path(spec):
    return os.path.join(_model_dir(spec["folder"]), spec["filename"])


def _download(url, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".", suffix=".tmp", dir=os.path.dirname(target))
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CRT-AutoDL/1.0"})
        with urllib.request.urlopen(req) as response, open(tmp, "wb") as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
        os.replace(tmp, target)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def ensure_model(key):
    spec = MODELS[key]
    target = _target_path(spec)
    if os.path.exists(target):
        return target
    for alternate in spec.get("alternate_filenames", []):
        alternate_path = os.path.join(_model_dir(spec["folder"]), alternate)
        if os.path.exists(alternate_path):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(alternate_path, target)
            return target
    _download(spec["url"], target)
    return target


def _dtype_map(name):
    return {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(name)


def _get_sage_attention(sage_attention):
    if sage_attention == "auto":
        from sageattention import sageattn

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    else:
        raise RuntimeError(f"[{TAG}] Unsupported sage_attention mode: {sage_attention}")

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
        elif skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
        return out

    return attention_sage


def _load_diffusion_model(path, weight_dtype="bf16", compute_dtype="bf16"):
    model_options = {}
    dtype = _dtype_map(weight_dtype)
    if dtype is not None:
        model_options["dtype"] = dtype
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
    if model is None:
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    if model is None:
        raise RuntimeError(f"[{TAG}] Failed to load diffusion model from: {path}")
    compute = _dtype_map(compute_dtype)
    if compute is not None:
        model.set_model_compute_dtype(compute)
        model.force_cast_weights = False
    return model


class _FixedDiffusionLoader:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patch_cublaslinear": ("BOOLEAN", {"default": False}),
                "sage_attention": (SAGE_ATTENTION_MODES, {"default": "auto"}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
            }
        }

    def load_model(self, patch_cublaslinear, sage_attention, enable_fp16_accumulation):
        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = enable_fp16_accumulation
        model = _load_diffusion_model(ensure_model(self.MODEL_KEY))
        if sage_attention == "auto":
            attention_sage = _get_sage_attention(sage_attention)

            def attention_override_sage(func, *args, **kwargs):
                return attention_sage.__wrapped__(*args, **kwargs)

            model.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = attention_override_sage
        return (model,)


class _FixedVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_vae(self):
        path = ensure_model(self.MODEL_KEY)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        is_audio_vae = (
            "vocoder.conv_post.weight" in sd
            or "vocoder.vocoder.conv_post.weight" in sd
            or "vocoder.resblocks.0.convs1.0.weight" in sd
            or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd
        )
        if is_audio_vae:
            sd = comfy.utils.state_dict_prefix_replace(dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        else:
            vae = comfy.sd.VAE(sd=sd, device=model_management.get_torch_device(), dtype=torch.bfloat16, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


class _CoreVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_vae(self):
        sd, metadata = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


class _FixedCLIPLoader:
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_clip(self):
        clip_path = ensure_model(self.MODEL_KEY)
        clip_type = getattr(comfy.sd.CLIPType, self.CLIP_TYPE.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={},
        )
        return (clip,)


class LTX23CLIP:
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "CRT/AutoDL/LTX2.3"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_clip(self):
        clip = comfy.sd.load_clip(
            ckpt_paths=[ensure_model("ltx23_gemma3"), ensure_model("ltx23_projection")],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.LTXV,
            model_options={},
        )
        return (clip,)


class _FixedLatentUpscaleModelLoader:
    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_model(self):
        from comfy_extras.nodes_hunyuan import HunyuanVideo15SRModel, LatentUpsampler

        sd, metadata = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), safe_load=True, return_metadata=True)
        if "blocks.0.block.0.conv.weight" in sd:
            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                "global_residual": False,
            }
            model = HunyuanVideo15SRModel("720p", config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
            }
            model = HunyuanVideo15SRModel("1080p", config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config).to(dtype=model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            model.load_state_dict(sd)
        else:
            raise RuntimeError(f"[{TAG}] Unsupported latent upscale model format: {self.MODEL_KEY}")
        return (model,)


class _FixedLoRALoader:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    def load_lora(self, model, strength_model):
        lora = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora,)


class LTX23Model(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_model"


class LTX23AudioVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_audio_vae"


class LTX23VideoVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_video_vae"


class LTX23LatentUpscaler(_FixedLatentUpscaleModelLoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_upscaler"


class LTX23ICLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_ic_lora"
    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("MODEL", "latent_downscale_factor")

    def load_lora(self, model, strength_model):
        lora_path = ensure_model(self.MODEL_KEY)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            logging.warning("[%s] Failed to extract reference_downscale_factor for %s", TAG, lora_path)
        if strength_model == 0:
            return (model, latent_downscale_factor)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora, latent_downscale_factor)


class LTX23ICOutpaintLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.3"
    MODEL_KEY = "ltx23_outpaint_lora"


class ZImageTurboModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_model"


class ZImageTurboVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_vae"


class ZImageTurboCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_clip"
    CLIP_TYPE = "lumina2"


class Flux2KleinModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_model"


class Flux2KleinVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_vae"


class Flux2KleinCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_clip"
    CLIP_TYPE = "flux2"


class Flux2KleinHDRILoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_hdri_lora"


class ErnieTurboModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_model"


class ErnieTurboNVFP4Model(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_nvfp4_model"


class ErnieModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_model"


class ErnieVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_vae"


class ErnieCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_clip"
    CLIP_TYPE = "flux2"


NODE_CLASS_MAPPINGS = {
    "CRTAutoDLLTX23Model": LTX23Model,
    "CRTAutoDLLTX23AudioVAE": LTX23AudioVAE,
    "CRTAutoDLLTX23VideoVAE": LTX23VideoVAE,
    "CRTAutoDLLTX23CLIP": LTX23CLIP,
    "CRTAutoDLLTX23LatentUpscaler": LTX23LatentUpscaler,
    "CRTAutoDLLTX23ICLoRA": LTX23ICLoRA,
    "CRTAutoDLLTX23ICOutpaintLoRA": LTX23ICOutpaintLoRA,
    "CRTAutoDLZImageTurboModel": ZImageTurboModel,
    "CRTAutoDLZImageTurboVAE": ZImageTurboVAE,
    "CRTAutoDLZImageTurboCLIP": ZImageTurboCLIP,
    "CRTAutoDLFlux2KleinModel": Flux2KleinModel,
    "CRTAutoDLFlux2KleinVAE": Flux2KleinVAE,
    "CRTAutoDLFlux2KleinCLIP": Flux2KleinCLIP,
    "CRTAutoDLFlux2KleinHDRILoRA": Flux2KleinHDRILoRA,
    "CRTAutoDLErnieTurboModel": ErnieTurboModel,
    "CRTAutoDLErnieTurboNVFP4Model": ErnieTurboNVFP4Model,
    "CRTAutoDLErnieModel": ErnieModel,
    "CRTAutoDLErnieVAE": ErnieVAE,
    "CRTAutoDLErnieCLIP": ErnieCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTAutoDLLTX23Model": "LTX2.3 Model (CRT AutoDL)",
    "CRTAutoDLLTX23AudioVAE": "LTX2.3 AUDIO VAE (CRT AutoDL)",
    "CRTAutoDLLTX23VideoVAE": "LTX2.3 VIDEO VAE (CRT AutoDL)",
    "CRTAutoDLLTX23CLIP": "LTX2.3 CLIP (CRT AutoDL)",
    "CRTAutoDLLTX23LatentUpscaler": "LTX2.3 Latent Upscaler (CRT AutoDL)",
    "CRTAutoDLLTX23ICLoRA": "LTX2.3 IC Cnet LoRA (CRT AutoDL)",
    "CRTAutoDLLTX23ICOutpaintLoRA": "LTX2.3 IC Outpaint LoRA (CRT AutoDL)",
    "CRTAutoDLZImageTurboModel": "ZImageTurbo Model (CRT AutoDL)",
    "CRTAutoDLZImageTurboVAE": "ZImageTurbo VAE (CRT AutoDL)",
    "CRTAutoDLZImageTurboCLIP": "ZImageTurbo CLIP (CRT AutoDL)",
    "CRTAutoDLFlux2KleinModel": "Flux2Klein Model (CRT AutoDL)",
    "CRTAutoDLFlux2KleinVAE": "Flux2Klein VAE (CRT AutoDL)",
    "CRTAutoDLFlux2KleinCLIP": "Flux2Klein CLIP (CRT AutoDL)",
    "CRTAutoDLFlux2KleinHDRILoRA": "Flux2Klein HDRI LoRA (CRT AutoDL)",
    "CRTAutoDLErnieTurboModel": "ERNIE_Turbo Model (CRT AutoDL)",
    "CRTAutoDLErnieTurboNVFP4Model": "ERNIE_Turbo NVFP4 Model (CRT AutoDL)",
    "CRTAutoDLErnieModel": "ERNIE Model (CRT AutoDL)",
    "CRTAutoDLErnieVAE": "ERNIE VAE (CRT AutoDL)",
    "CRTAutoDLErnieCLIP": "ERNIE CLIP (CRT AutoDL)",
}

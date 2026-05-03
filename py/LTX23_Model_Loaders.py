import logging
import math
import os
import urllib.error
import urllib.request

import comfy.model_management as model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes
import torch
from comfy.cli_args import args
from comfy.sd import VAE
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader

from .LTX23_AutoDownload import MODEL_DEFINITIONS


LOGGER = logging.getLogger(__name__)

SAGEATTN_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]

MODEL_TARGETS = {
    "diffusion_model": ("diffusion_models", MODEL_DEFINITIONS["diffusion_model"]["filename"]),
    "video_vae": ("vae", MODEL_DEFINITIONS["video_vae"]["filename"]),
    "audio_vae": ("vae", MODEL_DEFINITIONS["audio_vae"]["filename"]),
    "clip_gemma": ("text_encoders", MODEL_DEFINITIONS["clip_gemma"]["filename"]),
    "clip_projection": ("text_encoders", MODEL_DEFINITIONS["clip_projection"]["filename"]),
    "spatial_upscaler": ("latent_upscale_models", MODEL_DEFINITIONS["spatial_upscaler"]["filename"]),
    "ic_lora": ("loras", MODEL_DEFINITIONS["ic_lora"]["filename"]),
    "ic_lora_outpaint": ("loras", MODEL_DEFINITIONS["ic_lora_outpaint"]["filename"]),
}


def _target_path(model_type):
    subdir, filename = MODEL_TARGETS[model_type]
    return os.path.join(folder_paths.models_dir, subdir, filename)


def _download_if_missing(model_type):
    if model_type not in MODEL_TARGETS:
        raise ValueError(f"Unknown LTX 2.3 model type: {model_type}")

    path = _target_path(model_type)
    if os.path.isfile(path):
        return MODEL_TARGETS[model_type][1]

    info = MODEL_DEFINITIONS[model_type]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.part"
    print(f"[CRT LTX23 Loader] Downloading {info['filename']}...")

    try:
        req = urllib.request.Request(info["url"], headers={"User-Agent": "CRT-LTX23-ModelLoader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(temp_path, path)
        print(f"[CRT LTX23 Loader] Download complete: {info['filename']}")
    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download {info['filename']}: HTTP {e.code} {e.reason}") from e
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return MODEL_TARGETS[model_type][1]


def _load_vae_kj_style(vae_name, device="main_device", weight_dtype="bf16"):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[weight_dtype]
    if device == "main_device":
        device = model_management.get_torch_device()
    else:
        device = torch.device("cpu")

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)

    is_audio_vae = (
        "vocoder.conv_post.weight" in sd
        or "vocoder.vocoder.conv_post.weight" in sd
        or "vocoder.resblocks.0.convs1.0.weight" in sd
        or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd
    )
    if is_audio_vae:
        try:
            sd_audio = comfy.utils.state_dict_prefix_replace(
                dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
            )
            vae = VAE(sd=sd_audio, metadata=metadata)
            vae.throw_exception_if_invalid()
        except Exception:
            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

            vae = AudioVAE(sd, metadata)
    else:
        vae = VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
        vae.throw_exception_if_invalid()
    return (vae,)


def _load_sage_func(sage_attention):
    try:
        # Reuse CRT's wrapped attention implementation that matches
        # ComfyUI optimized_attention override call semantics.
        from .Models_Auto_DL import _get_sage_func as _crt_get_sage_func

        return _crt_get_sage_func(sage_attention)
    except Exception as e:
        LOGGER.warning("sage_attention=%s unavailable, falling back to default attention: %s", sage_attention, e)
        return None


def _load_lora_model_only(model, model_type, strength_model):
    lora_name = _download_if_missing(model_type)
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)

    try:
        latent_downscale_factor = float(metadata["reference_downscale_factor"])
    except (KeyError, ValueError, TypeError):
        latent_downscale_factor = 1.0
        LOGGER.warning("Missing reference_downscale_factor in %s, using 1.0", lora_path)

    if float(strength_model) == 0:
        return model, latent_downscale_factor
    model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, float(strength_model), 0)
    return model_lora, latent_downscale_factor


class CRT_LTX23BaseModelAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
                "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "bf16"}),
                "patch_cublaslinear": ("BOOLEAN", {"default": True}),
                "sage_attention": (SAGEATTN_MODES, {"default": "auto"}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "CRT/LTX2.3/loaders"

    def load_model(self, weight_dtype="default", compute_dtype="bf16", patch_cublaslinear=True, sage_attention="auto", enable_fp16_accumulation=True):
        model_name = _download_if_missing("diffusion_model")
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        model_options = {}
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype in dtype_map:
            model_options["dtype"] = dtype_map[weight_dtype]

        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enable_fp16_accumulation)

        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
        if compute_dtype in dtype_map:
            model.set_model_compute_dtype(dtype_map[compute_dtype])
            model.force_cast_weights = False
        if sage_attention != "disabled":
            new_attention = _load_sage_func(sage_attention)
            if new_attention is not None:
                def attention_override_sage(func, *args_, **kwargs_):
                    return new_attention(*args_, **kwargs_)

                model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
        return (model,)


class CRT_LTX23VideoVAEAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"device": (["main_device", "cpu"],), "weight_dtype": (["bf16", "fp16", "fp32"],)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "CRT/LTX2.3/loaders"

    def load_vae(self, device="main_device", weight_dtype="bf16"):
        return _load_vae_kj_style(_download_if_missing("video_vae"), device, weight_dtype)


class CRT_LTX23AudioVAEAutoLoader(CRT_LTX23VideoVAEAutoLoader):
    def load_vae(self, device="main_device", weight_dtype="bf16"):
        return _load_vae_kj_style(_download_if_missing("audio_vae"), device, weight_dtype)


class CRT_LTX23DualCLIPAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"device": (["default", "cpu"], {"default": "default"})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "CRT/LTX2.3/loaders"

    def load_clip(self, device="default"):
        return nodes.DualCLIPLoader().load_clip(
            _download_if_missing("clip_gemma"),
            _download_if_missing("clip_projection"),
            "ltxv",
            device=device,
        )


class CRT_LTX23LatentUpscaleModelAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "CRT/LTX2.3/loaders"

    def load_model(self):
        result = LatentUpscaleModelLoader.execute(model_name=_download_if_missing("spatial_upscaler"))
        return tuple(result.result if hasattr(result, "result") else result)


class CRT_LTX23ICLoRAUnionAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "strength_model": ("FLOAT", {"default": 0.8, "min": -100.0, "max": 100.0, "step": 0.01})}}

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("model", "latent_downscale_factor")
    FUNCTION = "load_lora"
    CATEGORY = "CRT/LTX2.3/loaders"

    def load_lora(self, model, strength_model=0.8):
        return _load_lora_model_only(model, "ic_lora", strength_model)


class CRT_LTX23ICLoRAOutpaintAutoLoader(CRT_LTX23ICLoRAUnionAutoLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})}}

    def load_lora(self, model, strength_model=1.0):
        return _load_lora_model_only(model, "ic_lora_outpaint", strength_model)


NODE_CLASS_MAPPINGS = {
    "CRT_LTX23BaseModelAutoLoader": CRT_LTX23BaseModelAutoLoader,
    "CRT_LTX23VideoVAEAutoLoader": CRT_LTX23VideoVAEAutoLoader,
    "CRT_LTX23AudioVAEAutoLoader": CRT_LTX23AudioVAEAutoLoader,
    "CRT_LTX23DualCLIPAutoLoader": CRT_LTX23DualCLIPAutoLoader,
    "CRT_LTX23LatentUpscaleModelAutoLoader": CRT_LTX23LatentUpscaleModelAutoLoader,
    "CRT_LTX23ICLoRAUnionAutoLoader": CRT_LTX23ICLoRAUnionAutoLoader,
    "CRT_LTX23ICLoRAOutpaintAutoLoader": CRT_LTX23ICLoRAOutpaintAutoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_LTX23BaseModelAutoLoader": "LTX 2.3 Base Model Auto Loader (CRT)",
    "CRT_LTX23VideoVAEAutoLoader": "LTX 2.3 VAE VIDEO Auto Loader (CRT)",
    "CRT_LTX23AudioVAEAutoLoader": "LTX 2.3 VAE AUDIO Auto Loader (CRT)",
    "CRT_LTX23DualCLIPAutoLoader": "LTX 2.3 DualCLIP Auto Loader (CRT)",
    "CRT_LTX23LatentUpscaleModelAutoLoader": "LTX 2.3 Latent Upscale Auto Loader (CRT)",
    "CRT_LTX23ICLoRAUnionAutoLoader": "LTX 2.3 IC-LoRA Union Auto Loader (CRT)",
    "CRT_LTX23ICLoRAOutpaintAutoLoader": "LTX 2.3 IC-LoRA Outpaint Auto Loader (CRT)",
}

import math
import os

import torch
import comfy.sd
import comfy.model_management as mm
import comfy.utils
import folder_paths
from tqdm import tqdm

from .download_progress import download_url_with_progress
from comfy_extras.nodes_depth_anything_3 import DA3Inference, DA3Render


TAG = "crt-da3"

MODEL_OPTIONS = [
    "depth_anything_3_small.safetensors",
    "depth_anything_3_base.safetensors",
    "depth_anything_3_mono_large.safetensors",
    "depth_anything_3_metric_large.safetensors",
]


def _geometry_estimation_dir():
    return os.path.join(folder_paths.models_dir, "geometry_estimation")


def _ensure_model(filename):
    target_dir = _geometry_estimation_dir()
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, filename)
    if os.path.exists(target):
        return target
    url = f"https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/{filename}"
    download_url_with_progress(
        url,
        target,
        label=filename,
        user_agent="CRT-DepthAnything3/1.0",
        console_prefix="CRT DepthAnything3",
    )
    return target


def _mp_to_resolution(megapixels, H, W):
    target_pixels = float(megapixels) * 1_000_000.0
    scale = math.sqrt(target_pixels / max(1.0, H * W))
    long_side = max(H, W) * scale
    res = int(round(long_side / 14.0)) * 14
    return min(max(res, 140), 2520)


def _unwrap_node_output(value):
    if hasattr(value, "args") and isinstance(value.args, tuple):
        if len(value.args) == 1:
            return value.args[0]
        return value.args
    return value


def _stack_geometries(geometries):
    if not geometries:
        return {}
    stacked = {}
    for key in geometries[0]:
        tensors = [g[key] for g in geometries]
        if key == "mode":
            stacked[key] = tensors[0]
        elif key in ("extrinsics", "intrinsics"):
            stacked[key] = torch.cat(tensors, dim=1)
        else:
            stacked[key] = torch.cat(tensors, dim=0)
    return stacked


def _run_da3_with_progress(model, image, resolution, mode, mode_dict):
    B = image.shape[0]
    if mode != "mono" or B <= 1:
        pbar = comfy.utils.ProgressBar(1)
        pbar.update(0)
        geometry = DA3Inference.execute.__func__(
            DA3Inference, model, image, resolution, "upper_bound_resize", mode_dict
        )
        geometry = _unwrap_node_output(geometry)
        pbar.update(1)
        return geometry

    geometries = []
    pbar = comfy.utils.ProgressBar(B)
    for i in tqdm(range(B), desc="[CRT DepthAnything3]", ncols=80, file=None, leave=False):
        geo = DA3Inference.execute.__func__(
            DA3Inference, model, image[i:i + 1], resolution, "upper_bound_resize", {"mode": "mono"}
        )
        geo = _unwrap_node_output(geo)
        geometries.append(geo)
        pbar.update(1)
    return _stack_geometries(geometries)


class CRT_DepthAnything3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "weight_dtype": (
                    ["default", "fp16", "bf16", "fp32"],
                    {"default": "default"},
                ),
                "image": ("IMAGE",),
                "megapixels": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.05, "max": 8.0, "step": 0.05},
                ),
                "mode": (["mono", "multiview"], {"default": "mono"}),
                "output": (
                    ["depth", "depth_colored", "sky_mask", "confidence"],
                    {"default": "depth"},
                ),
                "normalization": (
                    ["v2_style", "min_max", "raw"],
                    {"default": "v2_style"},
                ),
            },
            "optional": {
                "apply_sky_clip": ("BOOLEAN", {"default": False}),
                "colored": ("BOOLEAN", {"default": False}),
                "ref_view_strategy": (
                    ["saddle_balanced", "saddle_sim_range", "first", "middle"],
                    {"default": "saddle_balanced"},
                ),
                "pose_method": (["cam_dec", "ray_pose"], {"default": "cam_dec"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "CRT/DepthAnything3"

    def execute(
        self,
        model_name,
        weight_dtype,
        image,
        megapixels,
        mode,
        output,
        normalization,
        keep_model_loaded,
        apply_sky_clip=False,
        colored=False,
        ref_view_strategy="saddle_balanced",
        pose_method="cam_dec",
    ):
        H, W = image.shape[1], image.shape[2]
        resolution = _mp_to_resolution(megapixels, H, W)

        model_path = _ensure_model(model_name)
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        da3_model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
        if da3_model is None:
            raise RuntimeError(f"[{TAG}] Failed to load DA3 model from: {model_path}")

        mode_dict = {"mode": mode}
        if mode == "multiview":
            mode_dict["ref_view_strategy"] = ref_view_strategy
            mode_dict["pose_method"] = pose_method

        geometry = _run_da3_with_progress(da3_model, image, resolution, mode, mode_dict)

        output_dict = {"output": output}
        if output in ("depth", "depth_colored"):
            output_dict["normalization"] = normalization
            output_dict["apply_sky_clip"] = apply_sky_clip
        elif output in ("sky_mask", "confidence"):
            output_dict["colored"] = colored

        result = DA3Render.execute.__func__(DA3Render, geometry, output_dict)
        result = _unwrap_node_output(result)

        if not keep_model_loaded:
            mm.unload_model_and_clones(da3_model, unload_additional_models=True)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "CRT_DepthAnything3": CRT_DepthAnything3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_DepthAnything3": "DepthAnything3 (CRT)",
}

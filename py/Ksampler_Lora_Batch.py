import os

import torch

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths
import latent_preview


LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt")


def _resolve_lora_folder(folder_path):
    if not folder_path or not folder_path.strip():
        return None
    path = os.path.expanduser(folder_path.strip().strip('"').strip("'"))
    if os.path.isdir(path):
        return os.path.abspath(path)
    for base in [os.getcwd()] + folder_paths.get_folder_paths("loras"):
        candidate = os.path.join(base, path)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


def _list_lora_files(folder, recursive):
    if folder is None or not os.path.isdir(folder):
        return []
    files = []
    if recursive:
        for root, _dirs, names in os.walk(folder):
            for name in names:
                if name.lower().endswith(LORA_EXTENSIONS):
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and name.lower().endswith(LORA_EXTENSIONS):
                files.append(full)
    files.sort(key=lambda p: os.path.relpath(p, folder).lower())
    return files


class CRT_KSamplerLoraBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_folder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder containing the LoRAs to compare. Every .safetensors/.ckpt/.pt file is sampled once, in alphabetical order.",
                    },
                ),
                "model": ("MODEL",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Also scan subfolders of the LoRA folder."},
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latents", "lora_list")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    @classmethod
    def IS_CHANGED(cls, lora_folder="", recursive=False, **kwargs):
        try:
            folder = _resolve_lora_folder(lora_folder)
            files = _list_lora_files(folder, recursive)
            if not files:
                return float("nan")
            return "|".join(
                f"{os.path.relpath(p, folder)}:{os.path.getmtime(p)}" for p in files
            )
        except Exception:
            return float("nan")

    def sample(
        self,
        lora_folder,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        latent_image,
        denoise,
        lora_strength,
        recursive,
        negative=None,
    ):
        if model is None:
            raise RuntimeError(
                "[CRT KSampler LoRA Batch] MODEL input is None. Connect a valid diffusion MODEL."
            )

        folder = _resolve_lora_folder(lora_folder)
        lora_files = _list_lora_files(folder, recursive)
        if not lora_files:
            print(
                f"[CRT KSampler LoRA Batch] WARNING: no LoRA files found in "
                f"'{lora_folder}'. Sampling once with the base model."
            )

        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask", None)

        if negative is None:
            ref = positive[0][0]
            negative = [
                [
                    torch.zeros_like(ref),
                    {
                        "pooled_output": torch.zeros(
                            ref.shape[0],
                            ref.shape[-1],
                            device=ref.device,
                            dtype=ref.dtype,
                        )
                    },
                ]
            ]

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        runs = lora_files if lora_files else [None]
        latents_out = []
        loaded_names = []

        for i, lora_path in enumerate(runs):
            run_model = model
            if lora_path is not None:
                name = os.path.relpath(lora_path, folder)
                run_model = model.clone()
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                run_model, _ = comfy.sd.load_lora_for_models(
                    run_model, None, lora_data, lora_strength, lora_strength
                )
                del lora_data
                loaded_names.append(name)
                print(
                    f"[CRT KSampler LoRA Batch] {i + 1}/{len(runs)}: {name} "
                    f"@ {lora_strength}"
                )

            callback = latent_preview.prepare_callback(run_model, steps)
            out = comfy.sample.sample(
                run_model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_samples,
                denoise=denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
            latents_out.append(out)

            if lora_path is not None:
                del run_model
                comfy.model_management.soft_empty_cache()

        samples = torch.cat(latents_out, dim=0) if len(latents_out) > 1 else latents_out[0]

        result = latent_image.copy()
        result.pop("downscale_ratio_spacial", None)
        result.pop("downscale_ratio_temporal", None)
        result["samples"] = samples

        lora_list = (
            "\n".join(loaded_names)
            if loaded_names
            else f"No LoRA files found in: {lora_folder}"
        )
        return (result, lora_list)


NODE_CLASS_MAPPINGS = {"CRT_KSamplerLoraBatch": CRT_KSamplerLoraBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"CRT_KSamplerLoraBatch": "KSampler LoRA Batch (CRT)"}

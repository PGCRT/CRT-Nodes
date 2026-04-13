import numpy as np
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
import node_helpers


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"


def colored_print(message, color=Colors.ENDC):
    safe_message = str(message).encode("ascii", "ignore").decode("ascii")
    print(f"{color}{safe_message}{Colors.ENDC}")


def build_empty_negative_from_positive(positive):
    out = []
    for t in positive:
        out.append([torch.zeros_like(t[0]), t[1].copy()])
    return out


def attach_reference_latent(conditioning, ref_samples):
    if conditioning is None or isinstance(conditioning, str):
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [ref_samples]},
        append=False,
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
    total = len(out)
    start_idx = int(start * total)
    end_idx = int(end * total)
    for i in range(start_idx, end_idx):
        out[i] *= factor
    return out


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


def run_custom_sample(
    model, latent, sampler_obj, sigmas, positive, negative, cfg, seed, disable_noise
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
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, max(0, sigmas.shape[-1] - 1))
    sampled = comfy.sample.sample_custom(
        model,
        noise,
        cfg,
        sampler_obj,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = sampled
    return out


class LatentNoiseInjectionSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "euler"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "edit_model_flux2klein": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable ReferenceLatent-style conditioning for edit models",
                    },
                ),
                "stage1_sigma_factor": (
                    "FLOAT",
                    {"default": 1.005, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage2_sigma_factor": (
                    "FLOAT",
                    {"default": 0.995, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "stage1_sigma_start": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage1_sigma_end": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage2_sigma_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "stage2_sigma_end": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "details_amount_stage1": (
                    "FLOAT",
                    {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "details_amount_stage2": (
                    "FLOAT",
                    {"default": 0.15, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "enable_noise_injection": (
                    ["disable", "enable"],
                    {"default": "enable"},
                ),
                "injection_point": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "injection_seed_offset": (
                    "INT",
                    {"default": 1, "min": -100, "max": 100, "step": 1},
                ),
                "injection_strength": (
                    "FLOAT",
                    {"default": 0.10, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "normalize_injected_noise": (
                    ["enable", "disable"],
                    {"default": "enable"},
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(
        self,
        model,
        positive,
        sampler_name,
        scheduler,
        cfg,
        steps,
        denoise,
        seed,
        edit_model_flux2klein,
        stage1_sigma_factor,
        stage2_sigma_factor,
        stage1_sigma_start,
        stage1_sigma_end,
        stage2_sigma_start,
        stage2_sigma_end,
        details_amount_stage1,
        details_amount_stage2,
        enable_noise_injection,
        injection_point,
        injection_seed_offset,
        injection_strength,
        normalize_injected_noise,
        negative=None,
        latent_image=None,
        image=None,
        vae=None,
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
        colored_print("Starting Latent Noise Injection Sampler...", Colors.HEADER)

        if negative is None:
            negative = build_empty_negative_from_positive(positive)

        if vae is None:
            raise ValueError("LatentNoiseInjectionSampler requires a VAE input.")

        if image is not None:
            comfy.model_management.load_model_gpu(vae.patcher)
            latent_image = {"samples": vae.encode(image[:, :, :, :3])}
        elif latent_image is None:
            raise ValueError(
                "LatentNoiseInjectionSampler requires either an image or latent_image input."
            )

        base_sampler = comfy.samplers.sampler_object(sampler_name)
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

        if edit_model_flux2klein:
            pos_cond = attach_reference_latent(positive, latent_image["samples"])
            neg_cond = attach_reference_latent(
                negative,
                torch.zeros_like(latent_image["samples"]),
            )
        else:
            pos_cond = positive
            neg_cond = negative

        if enable_noise_injection == "enable":
            stage1_latent = run_custom_sample(
                model,
                latent_image,
                sampler_stage1,
                stage1_sigmas,
                pos_cond,
                neg_cond,
                cfg,
                seed,
                disable_noise=False,
            )

            actual_injection_seed = seed + injection_seed_offset
            injected_samples = stage1_latent["samples"].clone()
            torch.manual_seed(actual_injection_seed)
            new_noise = torch.randn_like(injected_samples)

            if normalize_injected_noise == "enable":
                original_mean = injected_samples.mean().item()
                original_std = injected_samples.std().item()
                if original_std > 1e-6:
                    new_noise = new_noise * original_std + original_mean

            injected_samples += new_noise * injection_strength
            injected_latent = stage1_latent.copy()
            injected_latent["samples"] = injected_samples

            final_latent = run_custom_sample(
                model,
                injected_latent,
                sampler_stage2,
                stage2_sigmas,
                pos_cond,
                neg_cond,
                cfg,
                seed,
                disable_noise=True,
            )
        else:
            full_sigmas = multiply_sigmas_stateless(
                sigmas_full,
                stage1_sigma_factor,
                stage1_sigma_start,
                stage1_sigma_end,
            )
            final_latent = run_custom_sample(
                model,
                latent_image,
                sampler_stage1,
                full_sigmas,
                pos_cond,
                neg_cond,
                cfg,
                seed,
                disable_noise=False,
            )

        decoded_image = vae.decode(final_latent["samples"])
        return (final_latent, decoded_image)


NODE_CLASS_MAPPINGS = {"LatentNoiseInjectionSampler": LatentNoiseInjectionSampler}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)"
}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .runtime import RuntimeRollbackSnapshot, SpectrumH3Runtime


@dataclass(slots=True)
class _PendingEulerStep:
    index: int
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    latent: torch.Tensor
    denoised: torch.Tensor
    runtime: RuntimeRollbackSnapshot


def _advance_euler(sampling_module, x, sigma, sigma_next, denoised):
    derivative = sampling_module.to_d(x, sigma, denoised)
    return x + derivative * (sigma_next - sigma)


def _callback(callback, pending: _PendingEulerStep, total_steps: int) -> None:
    if callback is not None:
        callback(pending.index, pending.denoised, pending.latent, total_steps)


@torch.no_grad()
def run_selective_rollback_euler(
    sampler: Any,
    runtime: SpectrumH3Runtime,
    model_wrap: Any,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any],
    callback,
    noise: torch.Tensor,
    latent_image: torch.Tensor | None,
    denoise_mask,
    disable_pbar: bool,
):
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_management
    import comfy.samplers

    extra_args["denoise_mask"] = denoise_mask
    model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
    model_k.latent_image = latent_image
    if sampler.inpaint_options.get("random", False):
        generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
        model_k.noise = torch.randn(
            noise.shape,
            generator=generator,
            device="cpu",
        ).to(noise.dtype).to(noise.device)
    else:
        model_k.noise = noise

    x = model_wrap.inner_model.model_sampling.noise_scaling(
        sigmas[0],
        noise,
        latent_image,
        sampler.max_denoise(model_wrap, sigmas),
    )
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    pending: _PendingEulerStep | None = None

    for index in sampling.trange(total_steps, disable=disable_pbar):
        comfy.model_management.throw_exception_if_processing_interrupted()
        runtime_snapshot = runtime.create_rollback_snapshot()
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        denoised = model_k(x, sigma * s_in, **extra_args)
        completed_mode = runtime.last_completed_mode

        if pending is None and completed_mode == "forecast":
            pending = _PendingEulerStep(
                index=index,
                sigma=sigma,
                sigma_next=sigma_next,
                latent=x.detach().clone(),
                denoised=denoised,
                runtime=runtime_snapshot,
            )
            x = _advance_euler(sampling, x, sigma, sigma_next, denoised)
            continue

        if pending is not None:
            if completed_mode != "actual":
                runtime.disable_experiment(
                    "selective rollback requires an actual anchor immediately after a forecast"
                )
                _callback(callback, pending, total_steps)
                pending = None
            elif runtime.consume_rollback_request():
                runtime.restore_rollback_snapshot(pending.runtime)
                runtime.begin_rollback_replay()
                try:
                    x = pending.latent
                    runtime.force_next_actual(
                        "selective rollback replay",
                        rollback_replay=True,
                    )
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    replayed = model_k(x, pending.sigma * s_in, **extra_args)
                    replayed_step = _PendingEulerStep(
                        pending.index,
                        pending.sigma,
                        pending.sigma_next,
                        x,
                        replayed,
                        pending.runtime,
                    )
                    _callback(callback, replayed_step, total_steps)
                    x = _advance_euler(
                        sampling,
                        x,
                        pending.sigma,
                        pending.sigma_next,
                        replayed,
                    )

                    runtime.force_next_actual(
                        "selective rollback corrected anchor",
                        rollback_replay=True,
                    )
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    corrected = model_k(x, sigma * s_in, **extra_args)
                    corrected_step = _PendingEulerStep(
                        index,
                        sigma,
                        sigma_next,
                        x,
                        corrected,
                        pending.runtime,
                    )
                    _callback(callback, corrected_step, total_steps)
                    x = _advance_euler(sampling, x, sigma, sigma_next, corrected)
                finally:
                    runtime.end_rollback_replay()
                pending = None
                continue
            else:
                _callback(callback, pending, total_steps)
                pending = None

        current = _PendingEulerStep(
            index,
            sigma,
            sigma_next,
            x,
            denoised,
            runtime_snapshot,
        )
        _callback(callback, current, total_steps)
        x = _advance_euler(sampling, x, sigma, sigma_next, denoised)

    if pending is not None:
        _callback(callback, pending, total_steps)

    return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)

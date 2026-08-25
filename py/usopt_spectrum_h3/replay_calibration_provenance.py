from __future__ import annotations

from typing import Any, Callable

from . import replay_calibration as _calibration
from . import sampling as _sampling
from .runtime import SpectrumH3Runtime

_RUNTIME_SEED_ATTR = "_spectrum_h3_observed_seed"
_ORIGINAL_OUTER_SAMPLE_WRAPPER: Callable[..., Any] | None = None
_ORIGINAL_BUILD_BLOCK: Callable[..., dict[str, Any]] | None = None


def _observed_seed(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return converted if converted == value else None
    except (TypeError, ValueError):
        return None


def _outer_sample_with_provenance(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    if _ORIGINAL_OUTER_SAMPLE_WRAPPER is None:
        raise RuntimeError("replay calibration provenance was not installed correctly")
    guider = executor.class_obj
    binding = _sampling._binding_from_model_options(
        getattr(guider, "model_options", None)
    )
    runtime = None if binding is None else binding.runtime
    if runtime is not None:
        setattr(runtime, _RUNTIME_SEED_ATTR, _observed_seed(seed))
    try:
        return _ORIGINAL_OUTER_SAMPLE_WRAPPER(
            executor,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes,
        )
    finally:
        if runtime is not None and hasattr(runtime, _RUNTIME_SEED_ATTR):
            delattr(runtime, _RUNTIME_SEED_ATTR)


def _deployable_target_signature(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "target_step_id": int(row["target_step_id"]),
            "coordinate": float(row["coordinate"]),
            "left_anchor_step_id": int(row["left_anchor_step_id"]),
            "right_anchor_step_id": int(row["right_anchor_step_id"]),
            "current_weight": float(row["current_weight"]),
            "causal_disagreement": float(row["causal_disagreement"]),
            "validation_penalty": float(row["validation_penalty"]),
            "spectral_gap": float(row["spectral_gap"]),
        }
        for row in block["target_rows"]
    ]


def _calibration_content_signature(block: dict[str, Any]) -> list[dict[str, Any]]:
    # Prompt/reference strings are intentionally not plumbed through the runtime.
    # Hash the exact scalar calibration content instead, excluding only run-local
    # and self-referential identifiers. This prevents two genuinely different
    # calibration trajectories with identical sampler/config/topology/seed from
    # collapsing to one trace identity while exact copied logs still deduplicate.
    return [
        {
            key: value
            for key, value in sorted(row.items())
            if key not in {"run_id", "trace_fingerprint"}
        }
        for row in block["target_rows"]
    ]


def _build_block_with_provenance(
    runtime: SpectrumH3Runtime,
    state: _calibration._CalibrationState,
) -> dict[str, Any]:
    if _ORIGINAL_BUILD_BLOCK is None:
        raise RuntimeError("replay calibration provenance was not installed correctly")
    block = _ORIGINAL_BUILD_BLOCK(runtime, state)
    provenance = block["provenance"]
    metadata = block["metadata"]
    seed = _observed_seed(getattr(runtime, _RUNTIME_SEED_ATTR, None))

    run_config = {
        "spectrum_config": block["config"],
        "sampler": metadata.get("sampler"),
        "steps": metadata.get("steps"),
        "scheduler": metadata.get("scheduler"),
    }
    config_hash = _calibration._sha256_json(run_config)
    provenance["seed"] = seed
    provenance["config_hash"] = config_hash
    metadata["config_hash_definition"] = (
        "sha256(canonical_json({spectrum_config,sampler,steps,scheduler}))"
    )
    metadata["seed_source"] = (
        "ComfyUI OUTER_SAMPLE seed" if seed is not None else "unavailable"
    )
    metadata["trace_fingerprint_definition"] = (
        "sha256(canonical_json(schema/source/package revision, seed, run config hash, "
        "sampler/steps, schedule/topology fingerprints, deployable target signature, "
        "exact scalar calibration-content signature excluding run_id/self fingerprint))"
    )

    trace_fingerprint = _calibration._sha256_json(
        {
            "schema_version": int(block["schema_version"]),
            "source_schema_revision": provenance.get("source_schema_revision"),
            "package_version": provenance.get("package_version"),
            "source_revision": provenance.get("source_revision"),
            "seed": seed,
            "config_hash": config_hash,
            "sampler": metadata.get("sampler"),
            "steps": metadata.get("steps"),
            "schedule_fingerprint": provenance.get("schedule_fingerprint"),
            "topology_fingerprint": provenance.get("topology_fingerprint"),
            "deployable_target_signature": _deployable_target_signature(block),
            "calibration_content_signature": _calibration_content_signature(block),
        }
    )
    provenance["trace_fingerprint"] = trace_fingerprint
    for row in block["target_rows"]:
        row["trace_fingerprint"] = trace_fingerprint
    return block


def install_replay_calibration_provenance() -> None:
    """Capture clean outer-sampler provenance without Git or workflow coupling."""
    global _ORIGINAL_OUTER_SAMPLE_WRAPPER
    global _ORIGINAL_BUILD_BLOCK
    if getattr(SpectrumH3Runtime, "_replay_calibration_provenance_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_calibration_installed", False):
        raise RuntimeError("install replay calibration before provenance capture")

    _ORIGINAL_OUTER_SAMPLE_WRAPPER = _sampling.outer_sample_wrapper
    _ORIGINAL_BUILD_BLOCK = _calibration._build_block
    _sampling.outer_sample_wrapper = _outer_sample_with_provenance
    _calibration._build_block = _build_block_with_provenance
    SpectrumH3Runtime._replay_calibration_provenance_installed = True


__all__ = ["install_replay_calibration_provenance"]

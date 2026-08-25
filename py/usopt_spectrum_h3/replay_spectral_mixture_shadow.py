from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import replay_component_shadow as _component
from . import replay_trust_shadow as _replay
from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .runtime import SpectrumH3Runtime

_ARCHIVE_ATTR = "_model_aware_trust_replay_spectral_mixture_shadow"
_FIXED_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

_ORIGINAL_COMPONENT_VALIDATOR: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(slots=True)
class _SpectralMixtureStream:
    count: int = 0
    local_ratio_sum: float = 0.0
    current_blend_ratio_sum: float = 0.0
    spectral_ratio_sum: float = 0.0
    oracle_ratio_sum: float = 0.0
    oracle_advantage_vs_local_sum: float = 0.0
    oracle_advantage_vs_current_sum: float = 0.0
    oracle_weight_sum: float = 0.0
    oracle_weight_min: float = 1.0
    oracle_weight_max: float = 0.0
    current_weight_projection_sum: float = 0.0
    current_weight_projection_min: float = math.inf
    current_weight_projection_max: float = -math.inf
    effective_blend_mean_sum: float = 0.0
    validation_penalty_mean_sum: float = 0.0
    spectral_gap_sum: float = 0.0
    fixed_ratio_sums: dict[float, float] = field(
        default_factory=lambda: {weight: 0.0 for weight in _FIXED_WEIGHTS}
    )
    fixed_advantage_vs_local_sums: dict[float, float] = field(
        default_factory=lambda: {weight: 0.0 for weight in _FIXED_WEIGHTS}
    )
    fixed_advantage_vs_current_sums: dict[float, float] = field(
        default_factory=lambda: {weight: 0.0 for weight in _FIXED_WEIGHTS}
    )
    causal_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    causal_adjustment_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    spectral_gap_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    spectral_gap_adjustment_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    validation_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    validation_adjustment_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    current_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    coordinate_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    coordinate_adjustment_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)

    def record(
        self,
        case: dict[str, Any],
        *,
        causal_disagreement: float,
        coordinate: float,
    ) -> None:
        required = (
            float(case["oracle_weight"])
            - float(case["current_weight_projection"])
        )
        values = (
            case["local_ratio"],
            case["current_blend_ratio"],
            case["spectral_ratio"],
            case["oracle_ratio"],
            case["oracle_weight"],
            case["current_weight_projection"],
            case["effective_blend_mean"],
            case["validation_penalty_mean"],
            case["spectral_gap"],
            causal_disagreement,
            coordinate,
            required,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return

        local = max(float(case["local_ratio"]), 1e-12)
        current = max(float(case["current_blend_ratio"]), 1e-12)
        oracle = float(case["oracle_ratio"])
        weight = float(case["oracle_weight"])
        projection = float(case["current_weight_projection"])
        gap = float(case["spectral_gap"])
        validation = float(case["validation_penalty_mean"])

        self.count += 1
        self.local_ratio_sum += float(case["local_ratio"])
        self.current_blend_ratio_sum += float(case["current_blend_ratio"])
        self.spectral_ratio_sum += float(case["spectral_ratio"])
        self.oracle_ratio_sum += oracle
        self.oracle_advantage_vs_local_sum += (local - oracle) / local
        self.oracle_advantage_vs_current_sum += (current - oracle) / current
        self.oracle_weight_sum += weight
        self.oracle_weight_min = min(self.oracle_weight_min, weight)
        self.oracle_weight_max = max(self.oracle_weight_max, weight)
        self.current_weight_projection_sum += projection
        self.current_weight_projection_min = min(
            self.current_weight_projection_min,
            projection,
        )
        self.current_weight_projection_max = max(
            self.current_weight_projection_max,
            projection,
        )
        self.effective_blend_mean_sum += float(case["effective_blend_mean"])
        self.validation_penalty_mean_sum += validation
        self.spectral_gap_sum += gap

        for fixed_weight in _FIXED_WEIGHTS:
            ratio = float(case["fixed_ratios"][fixed_weight])
            self.fixed_ratio_sums[fixed_weight] += ratio
            self.fixed_advantage_vs_local_sums[fixed_weight] += (local - ratio) / local
            self.fixed_advantage_vs_current_sums[fixed_weight] += (
                current - ratio
            ) / current

        self.causal_weight_corr.add(float(causal_disagreement), weight)
        self.causal_adjustment_corr.add(float(causal_disagreement), required)
        self.spectral_gap_weight_corr.add(gap, weight)
        self.spectral_gap_adjustment_corr.add(gap, required)
        self.validation_weight_corr.add(validation, weight)
        self.validation_adjustment_corr.add(validation, required)
        self.current_weight_corr.add(projection, weight)
        self.coordinate_weight_corr.add(float(coordinate), weight)
        self.coordinate_adjustment_corr.add(float(coordinate), required)

    def mean(self, total: float) -> float:
        return float(total) / self.count if self.count else 0.0

    def resolved_oracle_min(self) -> float:
        return self.oracle_weight_min if self.count else 0.0

    def resolved_oracle_max(self) -> float:
        return self.oracle_weight_max if self.count else 0.0

    def resolved_projection_min(self) -> float:
        return self.current_weight_projection_min if self.count else 0.0

    def resolved_projection_max(self) -> float:
        return self.current_weight_projection_max if self.count else 0.0


@dataclass(slots=True)
class _SpectralMixtureAggregate:
    compute_seconds: float = 0.0
    video: _SpectralMixtureStream = field(default_factory=_SpectralMixtureStream)


def _aggregate(archive: Any) -> _SpectralMixtureAggregate:
    aggregate = getattr(archive, _ARCHIVE_ATTR, None)
    if not isinstance(aggregate, _SpectralMixtureAggregate):
        aggregate = _SpectralMixtureAggregate()
        setattr(archive, _ARCHIVE_ATTR, aggregate)
    return aggregate


def _weight_projection(
    start: torch.Tensor,
    current: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    direction = (end - start).reshape(-1).to(torch.float32)
    current_delta = (current - start).reshape(-1).to(torch.float32)
    epsilon = _trust._tensor_rms(start).mul(1e-6).clamp_min(_trust._EPS)
    elements = max(1, int(direction.numel()))
    denominator = torch.dot(direction, direction).clamp_min(
        epsilon.square() * elements
    )
    return torch.dot(current_delta, direction) / denominator


def _spectral_case(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
) -> dict[str, Any] | None:
    if record.stream_name != "video" or record.blend_weight <= 1e-12:
        return None

    candidates = _component._construct_candidates(
        smoother,
        record,
        samples,
        anchor_ids,
    )
    if candidates is None:
        return None

    target_index = anchor_ids.index(record.step_id)
    retained = [index for index in range(len(anchor_ids)) if index != target_index]
    retained_ids = [anchor_ids[index] for index in retained]
    left_id = anchor_ids[target_index - 1]
    right_id = anchor_ids[target_index + 1]
    left_position = retained_ids.index(left_id)
    right_position = retained_ids.index(right_id)

    effective_blends = _replay._effective_blends_for_withheld_target(
        smoother,
        record,
        samples,
        anchor_ids,
        retained,
        left_position,
        right_position,
    )
    if torch.any(effective_blends <= 0):
        raise RuntimeError("spectral mixture shadow received nonpositive video blend")

    current_weight_projection = _weight_projection(
        candidates.local,
        candidates.blend_uncorrected,
        candidates.spectral,
    )
    validation_penalties = float(record.blend_weight) / effective_blends
    spectral_gap = _trust._tensor_rms(
        candidates.spectral - candidates.local
    ) / _trust._tensor_rms(candidates.local).clamp_min(_trust._EPS)
    fixed_predictions = {
        weight: candidates.local
        + float(weight) * (candidates.spectral - candidates.local)
        for weight in _FIXED_WEIGHTS
    }

    # Ground truth is read only after the full spectral/local candidate family and
    # every predictor observable have been built from the retained LOO cache.
    actual = samples[target_index]
    epsilon = _trust._tensor_rms(actual).mul(1e-6).clamp_min(_trust._EPS)
    hold_rms = _trust._tensor_rms(actual - candidates.hold).clamp_min(epsilon)
    local_ratio = _component._ratio(actual, candidates.local, hold_rms)
    current_blend_ratio = _component._ratio(
        actual,
        candidates.blend_uncorrected,
        hold_rms,
    )
    spectral_ratio = _component._ratio(actual, candidates.spectral, hold_rms)
    oracle_ratio, oracle_weight = _component._axis_score(
        actual,
        candidates.local,
        candidates.spectral,
        hold_rms,
    )
    fixed_ratios = {
        weight: _component._ratio(actual, prediction, hold_rms)
        for weight, prediction in fixed_predictions.items()
    }

    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("local_ratio", local_ratio),
        ("current_blend_ratio", current_blend_ratio),
        ("spectral_ratio", spectral_ratio),
        ("oracle_ratio", oracle_ratio),
        ("oracle_weight", oracle_weight),
        ("current_weight_projection", current_weight_projection),
        ("effective_blend_mean", effective_blends.mean()),
        ("validation_penalty_mean", validation_penalties.mean()),
        ("spectral_gap", spectral_gap),
        *[
            (f"fixed:{weight:.2f}", fixed_ratios[weight])
            for weight in _FIXED_WEIGHTS
        ],
    ]
    values = (
        torch.stack([tensor for _, tensor in named_tensors])
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .tolist()
    )
    resolved = {
        name: float(value)
        for (name, _), value in zip(named_tensors, values, strict=True)
    }
    return {
        "local_ratio": resolved["local_ratio"],
        "current_blend_ratio": resolved["current_blend_ratio"],
        "spectral_ratio": resolved["spectral_ratio"],
        "oracle_ratio": resolved["oracle_ratio"],
        "oracle_weight": resolved["oracle_weight"],
        "current_weight_projection": resolved["current_weight_projection"],
        "effective_blend_mean": resolved["effective_blend_mean"],
        "validation_penalty_mean": resolved["validation_penalty_mean"],
        "spectral_gap": resolved["spectral_gap"],
        "fixed_ratios": {
            weight: resolved[f"fixed:{weight:.2f}"]
            for weight in _FIXED_WEIGHTS
        },
    }


def _validate_spectral_mixture_shadow(
    smoother: OfflineSmoother,
    trust_aggregate: _trust._TrustAggregate,
) -> None:
    records = getattr(
        smoother.archive,
        "_model_aware_trust_replay_shadow_records",
        None,
    )
    if not isinstance(records, list) or not records:
        return

    aggregate = _aggregate(smoother.archive)
    started = time.perf_counter()
    try:
        try:
            ranges = {
                name: (start, end) for name, start, end in smoother._stream_ranges
            }
            video_range = ranges.get("video")
            if video_range is None:
                return
            samples = _trust._sample_archive_stream(
                smoother,
                video_range[0],
                video_range[1],
            )
            anchor_ids = list(smoother._anchor_ids)
        except torch.cuda.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError, KeyError, IndexError):
            trust_aggregate.replay_shadow_failures += 1
            return

        for record in records:
            if (
                not isinstance(record, _trust._ReplayShadowRecord)
                or record.stream_name != "video"
            ):
                continue
            try:
                case = _spectral_case(smoother, record, samples, anchor_ids)
                if case is None:
                    continue
                aggregate.video.record(
                    case,
                    causal_disagreement=record.disagreement,
                    coordinate=record.coordinate,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                KeyError,
                IndexError,
            ):
                trust_aggregate.replay_shadow_failures += 1
    finally:
        aggregate.compute_seconds += time.perf_counter() - started


def _validate_component_with_spectral_mixture(
    smoother: OfflineSmoother,
    aggregate: _trust._TrustAggregate,
) -> None:
    if _ORIGINAL_COMPONENT_VALIDATOR is None:
        raise RuntimeError("spectral mixture shadow was not installed correctly")
    failures_before = aggregate.replay_shadow_failures
    _ORIGINAL_COMPONENT_VALIDATOR(smoother, aggregate)
    if aggregate.replay_shadow_failures != failures_before:
        return
    _validate_spectral_mixture_shadow(smoother, aggregate)


def _weight_suffix(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def _video_summary(stream: _SpectralMixtureStream) -> str:
    prefix = "model_aware_trust_replay_spectral_video"
    fields = [
        f"{prefix}_samples={stream.count}",
        f"{prefix}_local_ratio_mean={stream.mean(stream.local_ratio_sum):.6f}",
        f"{prefix}_current_blend_ratio_mean={stream.mean(stream.current_blend_ratio_sum):.6f}",
        f"{prefix}_full_spectral_ratio_mean={stream.mean(stream.spectral_ratio_sum):.6f}",
        f"{prefix}_oracle_ratio_mean={stream.mean(stream.oracle_ratio_sum):.6f}",
        f"{prefix}_oracle_advantage_vs_local_mean={stream.mean(stream.oracle_advantage_vs_local_sum):.6f}",
        f"{prefix}_oracle_advantage_vs_current_mean={stream.mean(stream.oracle_advantage_vs_current_sum):.6f}",
        f"{prefix}_oracle_weight_mean={stream.mean(stream.oracle_weight_sum):.6f}",
        f"{prefix}_oracle_weight_min={stream.resolved_oracle_min():.6f}",
        f"{prefix}_oracle_weight_max={stream.resolved_oracle_max():.6f}",
        f"{prefix}_current_weight_projection_mean={stream.mean(stream.current_weight_projection_sum):.6f}",
        f"{prefix}_current_weight_projection_min={stream.resolved_projection_min():.6f}",
        f"{prefix}_current_weight_projection_max={stream.resolved_projection_max():.6f}",
        f"{prefix}_effective_blend_mean={stream.mean(stream.effective_blend_mean_sum):.6f}",
        f"{prefix}_validation_penalty_mean={stream.mean(stream.validation_penalty_mean_sum):.6f}",
        f"{prefix}_spectral_gap_mean={stream.mean(stream.spectral_gap_sum):.6f}",
        f"{prefix}_causal_disagreement_oracle_weight_corr={stream.causal_weight_corr.correlation():.6f}",
        f"{prefix}_causal_disagreement_required_adjustment_corr={stream.causal_adjustment_corr.correlation():.6f}",
        f"{prefix}_spectral_gap_oracle_weight_corr={stream.spectral_gap_weight_corr.correlation():.6f}",
        f"{prefix}_spectral_gap_required_adjustment_corr={stream.spectral_gap_adjustment_corr.correlation():.6f}",
        f"{prefix}_validation_penalty_oracle_weight_corr={stream.validation_weight_corr.correlation():.6f}",
        f"{prefix}_validation_penalty_required_adjustment_corr={stream.validation_adjustment_corr.correlation():.6f}",
        f"{prefix}_current_weight_oracle_weight_corr={stream.current_weight_corr.correlation():.6f}",
        f"{prefix}_coordinate_oracle_weight_corr={stream.coordinate_weight_corr.correlation():.6f}",
        f"{prefix}_coordinate_required_adjustment_corr={stream.coordinate_adjustment_corr.correlation():.6f}",
    ]
    for weight in _FIXED_WEIGHTS:
        suffix = _weight_suffix(weight)
        fields.extend(
            (
                f"{prefix}_fixed_{suffix}_ratio_mean={stream.mean(stream.fixed_ratio_sums[weight]):.6f}",
                f"{prefix}_fixed_{suffix}_advantage_vs_local_mean={stream.mean(stream.fixed_advantage_vs_local_sums[weight]):.6f}",
                f"{prefix}_fixed_{suffix}_advantage_vs_current_mean={stream.mean(stream.fixed_advantage_vs_current_sums[weight]):.6f}",
            )
        )
    return " ".join(fields)


def _debug_summary_with_spectral_mixture(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("spectral mixture shadow was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    if archive is None or not _replay._archive_shadow_only(archive):
        return summary
    aggregate = getattr(archive, _ARCHIVE_ATTR, None)
    if not isinstance(aggregate, _SpectralMixtureAggregate):
        aggregate = _SpectralMixtureAggregate()
    return (
        f"{summary} "
        "model_aware_trust_replay_spectral_mixture=video_local_to_full_spectral_shadow "
        "model_aware_trust_replay_spectral_mixture_applied=0 "
        "model_aware_trust_replay_spectral_mixture_baseline=uncorrected_validation_attenuated_blend "
        f"model_aware_trust_replay_spectral_mixture_compute_s={aggregate.compute_seconds:.6f} "
        f"{_video_summary(aggregate.video)}"
    )


def install_replay_spectral_mixture_shadow() -> None:
    """Install video-only shadow calibration for replay local/spectral mixture."""
    global _ORIGINAL_COMPONENT_VALIDATOR
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_spectral_mixture_shadow_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_shadow_composition_installed", False):
        raise RuntimeError("install replay shadow composition before spectral mixture shadow")

    _ORIGINAL_COMPONENT_VALIDATOR = _component._validate_replay_decomposition
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    _component._validate_replay_decomposition = _validate_component_with_spectral_mixture
    SpectrumH3Runtime.debug_summary = _debug_summary_with_spectral_mixture
    SpectrumH3Runtime._replay_spectral_mixture_shadow_installed = True


__all__ = ["install_replay_spectral_mixture_shadow"]

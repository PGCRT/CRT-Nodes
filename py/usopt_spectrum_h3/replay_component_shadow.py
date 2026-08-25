from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import replay_trust_shadow as _replay
from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .runtime import SpectrumH3Runtime

_ARCHIVE_COMPONENT_ATTR = "_model_aware_trust_replay_component_shadow"
_CANDIDATES = (
    "local",
    "blend_uncorrected",
    "local_corrected",
    "blend_corrected",
)
_AXES = (
    "local_to_blend_uncorrected",
    "local_to_local_corrected",
    "blend_uncorrected_to_blend_corrected",
    "local_to_current",
)
_SCALARS = (
    "correction_gain",
    "local_correction_advantage",
    "blend_correction_advantage",
    "correction_blend_interaction_ratio_delta",
    "local_residual_replay_delta_projection",
    "blend_residual_replay_delta_projection",
    "local_residual_replay_delta_cosine",
    "blend_residual_replay_delta_cosine",
    "causal_replay_delta_cosine",
    "local_projection_minus_causal_gain",
    "blend_projection_minus_causal_gain",
)

_ORIGINAL_REPLAY_VALIDATOR: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(slots=True)
class _ScalarMean:
    count: int = 0
    total: float = 0.0

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.count += 1
        self.total += float(value)

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(slots=True)
class _AxisAggregate:
    count: int = 0
    ratio_sum: float = 0.0
    advantage_sum: float = 0.0
    kappa_sum: float = 0.0
    kappa_min: float = 1.0
    kappa_max: float = 0.0

    def record(
        self,
        *,
        baseline_ratio: float,
        oracle_ratio: float,
        oracle_kappa: float,
    ) -> None:
        values = (baseline_ratio, oracle_ratio, oracle_kappa)
        if not all(math.isfinite(float(value)) for value in values):
            return
        baseline = max(float(baseline_ratio), 1e-12)
        kappa = float(oracle_kappa)
        self.count += 1
        self.ratio_sum += float(oracle_ratio)
        self.advantage_sum += (baseline - float(oracle_ratio)) / baseline
        self.kappa_sum += kappa
        self.kappa_min = min(self.kappa_min, kappa)
        self.kappa_max = max(self.kappa_max, kappa)

    def mean_ratio(self) -> float:
        return self.ratio_sum / self.count if self.count else 0.0

    def mean_advantage(self) -> float:
        return self.advantage_sum / self.count if self.count else 0.0

    def mean_kappa(self) -> float:
        return self.kappa_sum / self.count if self.count else 0.0

    def resolved_min(self) -> float:
        return self.kappa_min if self.count else 0.0

    def resolved_max(self) -> float:
        return self.kappa_max if self.count else 0.0


@dataclass(slots=True)
class _ReplayComponentStream:
    count: int = 0
    candidate_ratio_sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATES}
    )
    candidate_advantage_sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATES}
    )
    axes: dict[str, _AxisAggregate] = field(
        default_factory=lambda: {name: _AxisAggregate() for name in _AXES}
    )
    scalars: dict[str, _ScalarMean] = field(
        default_factory=lambda: {name: _ScalarMean() for name in _SCALARS}
    )
    causal_local_kappa_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    causal_required_local_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    replay_observer_count: int = 0
    replay_local_kappa_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    replay_required_local_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    replay_blend_kappa_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)

    def record(
        self,
        case: dict[str, Any],
        *,
        causal_disagreement: float,
    ) -> None:
        ratios = case["candidate_ratios"]
        baseline_ratio = float(ratios["blend_corrected"])
        if not math.isfinite(baseline_ratio):
            return
        baseline = max(baseline_ratio, 1e-12)
        if not all(math.isfinite(float(ratios[name])) for name in _CANDIDATES):
            return

        self.count += 1
        for name in _CANDIDATES:
            ratio = float(ratios[name])
            self.candidate_ratio_sums[name] += ratio
            self.candidate_advantage_sums[name] += (baseline_ratio - ratio) / baseline

        for name in _AXES:
            axis = case["axes"][name]
            self.axes[name].record(
                baseline_ratio=baseline_ratio,
                oracle_ratio=axis["oracle_ratio"],
                oracle_kappa=axis["oracle_kappa"],
            )

        for name in _SCALARS:
            self.scalars[name].add(case.get(name))

        local_kappa = float(case["axes"]["local_to_current"]["oracle_kappa"])
        required_local = 1.0 - local_kappa
        self.causal_local_kappa_corr.add(causal_disagreement, local_kappa)
        self.causal_required_local_corr.add(causal_disagreement, required_local)

        replay_disagreement = case.get("replay_disagreement")
        if replay_disagreement is not None and math.isfinite(float(replay_disagreement)):
            value = float(replay_disagreement)
            self.replay_observer_count += 1
            self.replay_local_kappa_corr.add(value, local_kappa)
            self.replay_required_local_corr.add(value, required_local)
            blend_kappa = float(
                case["axes"]["local_to_blend_uncorrected"]["oracle_kappa"]
            )
            self.replay_blend_kappa_corr.add(value, blend_kappa)

    def mean_candidate_ratio(self, name: str) -> float:
        return self.candidate_ratio_sums[name] / self.count if self.count else 0.0

    def mean_candidate_advantage(self, name: str) -> float:
        return self.candidate_advantage_sums[name] / self.count if self.count else 0.0


@dataclass(slots=True)
class _ReplayComponentAggregate:
    compute_seconds: float = 0.0
    audio: _ReplayComponentStream = field(default_factory=_ReplayComponentStream)
    video: _ReplayComponentStream = field(default_factory=_ReplayComponentStream)


@dataclass(frozen=True, slots=True)
class _ReplayCandidates:
    retained_anchor_ids: tuple[int, ...]
    local: torch.Tensor
    blend_uncorrected: torch.Tensor
    local_corrected: torch.Tensor
    blend_corrected: torch.Tensor
    spectral: torch.Tensor
    hold: torch.Tensor
    correction_delta: torch.Tensor
    causal_delta: torch.Tensor | None


def _component_aggregate(archive: Any) -> _ReplayComponentAggregate:
    aggregate = getattr(archive, _ARCHIVE_COMPONENT_ATTR, None)
    if not isinstance(aggregate, _ReplayComponentAggregate):
        aggregate = _ReplayComponentAggregate()
        setattr(archive, _ARCHIVE_COMPONENT_ATTR, aggregate)
    return aggregate


def _construct_candidates(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
) -> _ReplayCandidates | None:
    target_index = anchor_ids.index(record.step_id)
    latest_index = anchor_ids.index(record.latest_anchor_id)
    if (
        target_index <= 0
        or target_index >= len(anchor_ids) - 1
        or latest_index >= target_index
    ):
        return None

    retained = [index for index in range(len(anchor_ids)) if index != target_index]
    if len(retained) < max(2, record.degree + 1):
        return None
    retained_ids = [anchor_ids[index] for index in retained]
    if record.step_id in retained_ids:
        raise RuntimeError("replay decomposition LOO target leaked into retained anchors")

    spectral = _replay._spectral_prediction(
        smoother,
        samples,
        anchor_ids,
        retained,
        record.coordinate,
        degree=record.degree,
        ridge_lambda=record.ridge_lambda,
        model_aware=True,
    )

    left_id = anchor_ids[target_index - 1]
    right_id = anchor_ids[target_index + 1]
    left_position = retained_ids.index(left_id)
    right_position = retained_ids.index(right_id)
    left = smoother.archive.anchors[target_index - 1]
    right = smoother.archive.anchors[target_index + 1]
    spacing = right.coordinate - left.coordinate
    if abs(spacing) <= 1e-12:
        raise RuntimeError("replay decomposition bracket has duplicate coordinates")
    ratio = (record.coordinate - left.coordinate) / spacing

    retained_samples = samples[retained]
    local_weights = torch.zeros(len(retained), dtype=torch.float32)
    local_weights[left_position] = 1.0 - ratio
    local_weights[right_position] = ratio
    local = torch.einsum("k,kbs->bs", local_weights, retained_samples)

    effective_blends = _replay._effective_blends_for_withheld_target(
        smoother,
        record,
        samples,
        anchor_ids,
        retained,
        left_position,
        right_position,
    )
    blend_uncorrected = (
        effective_blends[:, None] * spectral
        + (1.0 - effective_blends[:, None]) * local
    )

    correction_delta = retained_samples[right_position] - retained_samples[left_position]
    correction = float(record.correction_gain) * correction_delta
    local_corrected = local + correction
    blend_corrected = blend_uncorrected + correction

    latest_position = retained_ids.index(record.latest_anchor_id)
    hold = retained_samples[latest_position]
    causal_delta = None
    if latest_index > 0:
        previous_id = anchor_ids[latest_index - 1]
        previous_position = retained_ids.index(previous_id)
        causal_delta = hold - retained_samples[previous_position]

    return _ReplayCandidates(
        retained_anchor_ids=tuple(retained_ids),
        local=local,
        blend_uncorrected=blend_uncorrected,
        local_corrected=local_corrected,
        blend_corrected=blend_corrected,
        spectral=spectral,
        hold=hold,
        correction_delta=correction_delta,
        causal_delta=causal_delta,
    )


def _ratio(
    actual: torch.Tensor,
    prediction: torch.Tensor,
    hold_rms: torch.Tensor,
) -> torch.Tensor:
    return _trust._tensor_rms(actual - prediction) / hold_rms


def _axis_score(
    actual: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    hold_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kappa = _trust.oracle_segment_kappa(actual, start, end)
    oracle = start + kappa.to(dtype=end.dtype) * (end - start)
    return _ratio(actual, oracle, hold_rms), kappa


def _projection_and_cosine(
    residual: torch.Tensor,
    direction: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_flat = residual.reshape(-1).to(torch.float32)
    direction_flat = direction.reshape(-1).to(torch.float32)
    elements = max(1, int(direction_flat.numel()))
    direction_norm_sq = torch.dot(direction_flat, direction_flat)
    denominator = direction_norm_sq.clamp_min(epsilon.square() * elements)
    projection = torch.dot(residual_flat, direction_flat) / denominator
    residual_norm = torch.linalg.vector_norm(residual_flat)
    direction_norm = torch.sqrt(denominator)
    cosine = torch.dot(residual_flat, direction_flat) / (
        residual_norm.clamp_min(epsilon) * direction_norm.clamp_min(epsilon)
    )
    return projection, cosine.clamp(-1.0, 1.0)


def _cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    left_flat = left.reshape(-1).to(torch.float32)
    right_flat = right.reshape(-1).to(torch.float32)
    denominator = (
        torch.linalg.vector_norm(left_flat).clamp_min(epsilon)
        * torch.linalg.vector_norm(right_flat).clamp_min(epsilon)
    )
    return (torch.dot(left_flat, right_flat) / denominator).clamp(-1.0, 1.0)


def _decomposition_case(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
) -> dict[str, Any] | None:
    candidates = _construct_candidates(smoother, record, samples, anchor_ids)
    if candidates is None:
        return None

    target_index = anchor_ids.index(record.step_id)
    # Ground truth is intentionally read only after every candidate and validation
    # quantity has been constructed from the retained LOO cache.
    actual = samples[target_index]
    epsilon = _trust._tensor_rms(actual).mul(1e-6).clamp_min(_trust._EPS)
    hold_rms = _trust._tensor_rms(actual - candidates.hold).clamp_min(epsilon)

    predictions = {
        "local": candidates.local,
        "blend_uncorrected": candidates.blend_uncorrected,
        "local_corrected": candidates.local_corrected,
        "blend_corrected": candidates.blend_corrected,
    }
    ratio_tensors = {
        name: _ratio(actual, prediction, hold_rms)
        for name, prediction in predictions.items()
    }

    axis_predictions = {
        "local_to_blend_uncorrected": (
            candidates.local,
            candidates.blend_uncorrected,
        ),
        "local_to_local_corrected": (
            candidates.local,
            candidates.local_corrected,
        ),
        "blend_uncorrected_to_blend_corrected": (
            candidates.blend_uncorrected,
            candidates.blend_corrected,
        ),
        "local_to_current": (
            candidates.local,
            candidates.blend_corrected,
        ),
    }
    axis_tensors = {
        name: _axis_score(actual, start, end, hold_rms)
        for name, (start, end) in axis_predictions.items()
    }

    local_projection, local_cosine = _projection_and_cosine(
        actual - candidates.local,
        candidates.correction_delta,
        epsilon,
    )
    blend_projection, blend_cosine = _projection_and_cosine(
        actual - candidates.blend_uncorrected,
        candidates.correction_delta,
        epsilon,
    )
    causal_replay_cosine = (
        None
        if candidates.causal_delta is None
        else _cosine(candidates.causal_delta, candidates.correction_delta, epsilon)
    )

    replay_disagreement = None
    if record.blend_weight > 1e-12:
        replay_disagreement = _trust._tensor_rms(
            candidates.spectral - candidates.local
        ) / _trust._tensor_rms(candidates.blend_corrected).clamp_min(_trust._EPS)

    local_ratio = ratio_tensors["local"]
    blend_uncorrected_ratio = ratio_tensors["blend_uncorrected"]
    local_corrected_ratio = ratio_tensors["local_corrected"]
    blend_corrected_ratio = ratio_tensors["blend_corrected"]
    local_correction_advantage = (
        local_ratio - local_corrected_ratio
    ) / local_ratio.clamp_min(epsilon)
    blend_correction_advantage = (
        blend_uncorrected_ratio - blend_corrected_ratio
    ) / blend_uncorrected_ratio.clamp_min(epsilon)
    interaction = (
        blend_corrected_ratio
        - blend_uncorrected_ratio
        - (local_corrected_ratio - local_ratio)
    )

    named_tensors: list[tuple[str, torch.Tensor]] = [
        *[(f"candidate:{name}", ratio_tensors[name]) for name in _CANDIDATES],
        ("local_correction_advantage", local_correction_advantage),
        ("blend_correction_advantage", blend_correction_advantage),
        ("correction_blend_interaction_ratio_delta", interaction),
        ("local_residual_replay_delta_projection", local_projection),
        ("blend_residual_replay_delta_projection", blend_projection),
        ("local_residual_replay_delta_cosine", local_cosine),
        ("blend_residual_replay_delta_cosine", blend_cosine),
    ]
    for name in _AXES:
        oracle_ratio, oracle_kappa = axis_tensors[name]
        named_tensors.extend(
            (
                (f"axis:{name}:oracle_ratio", oracle_ratio),
                (f"axis:{name}:oracle_kappa", oracle_kappa),
            )
        )
    if causal_replay_cosine is not None:
        named_tensors.append(("causal_replay_delta_cosine", causal_replay_cosine))
    if replay_disagreement is not None:
        named_tensors.append(("replay_disagreement", replay_disagreement))

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
    ratios = {
        name: resolved[f"candidate:{name}"]
        for name in _CANDIDATES
    }
    result: dict[str, Any] = {
        "retained_anchor_ids": candidates.retained_anchor_ids,
        "candidate_ratios": ratios,
        "correction_gain": float(record.correction_gain),
        "local_correction_advantage": resolved["local_correction_advantage"],
        "blend_correction_advantage": resolved["blend_correction_advantage"],
        "correction_blend_interaction_ratio_delta": resolved[
            "correction_blend_interaction_ratio_delta"
        ],
        "local_residual_replay_delta_projection": resolved[
            "local_residual_replay_delta_projection"
        ],
        "blend_residual_replay_delta_projection": resolved[
            "blend_residual_replay_delta_projection"
        ],
        "local_residual_replay_delta_cosine": resolved[
            "local_residual_replay_delta_cosine"
        ],
        "blend_residual_replay_delta_cosine": resolved[
            "blend_residual_replay_delta_cosine"
        ],
    }
    result["local_projection_minus_causal_gain"] = (
        result["local_residual_replay_delta_projection"] - float(record.correction_gain)
    )
    result["blend_projection_minus_causal_gain"] = (
        result["blend_residual_replay_delta_projection"] - float(record.correction_gain)
    )
    result["axes"] = {
        name: {
            "oracle_ratio": resolved[f"axis:{name}:oracle_ratio"],
            "oracle_kappa": resolved[f"axis:{name}:oracle_kappa"],
        }
        for name in _AXES
    }
    result["causal_replay_delta_cosine"] = (
        None
        if causal_replay_cosine is None
        else resolved["causal_replay_delta_cosine"]
    )
    result["replay_disagreement"] = (
        None if replay_disagreement is None else resolved["replay_disagreement"]
    )
    return result


def _validate_replay_decomposition(
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

    aggregate = _component_aggregate(smoother.archive)
    started = time.perf_counter()
    try:
        try:
            ranges = {
                name: (start, end) for name, start, end in smoother._stream_ranges
            }
            samples_by_stream = {
                name: _trust._sample_archive_stream(smoother, start, end)
                for name, (start, end) in ranges.items()
                if name in {"audio", "video"}
            }
            anchor_ids = list(smoother._anchor_ids)
        except torch.cuda.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError, KeyError, IndexError):
            trust_aggregate.replay_shadow_failures += 1
            return

        for record in records:
            if not isinstance(record, _trust._ReplayShadowRecord):
                continue
            samples = samples_by_stream.get(record.stream_name)
            if samples is None:
                continue
            try:
                case = _decomposition_case(smoother, record, samples, anchor_ids)
                if case is None:
                    continue
                stream = (
                    aggregate.audio
                    if record.stream_name == "audio"
                    else aggregate.video
                )
                stream.record(case, causal_disagreement=record.disagreement)
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


def _stream_summary(name: str, stream: _ReplayComponentStream) -> str:
    prefix = f"model_aware_trust_replay_decomp_{name}"
    fields = [f"{prefix}_samples={stream.count}"]
    for candidate in _CANDIDATES:
        fields.extend(
            (
                f"{prefix}_{candidate}_ratio_mean={stream.mean_candidate_ratio(candidate):.6f}",
                f"{prefix}_{candidate}_advantage_vs_baseline_mean={stream.mean_candidate_advantage(candidate):.6f}",
            )
        )
    for axis_name in _AXES:
        axis = stream.axes[axis_name]
        fields.extend(
            (
                f"{prefix}_{axis_name}_oracle_ratio_mean={axis.mean_ratio():.6f}",
                f"{prefix}_{axis_name}_oracle_advantage_mean={axis.mean_advantage():.6f}",
                f"{prefix}_{axis_name}_oracle_kappa_mean={axis.mean_kappa():.6f}",
                f"{prefix}_{axis_name}_oracle_kappa_min={axis.resolved_min():.6f}",
                f"{prefix}_{axis_name}_oracle_kappa_max={axis.resolved_max():.6f}",
            )
        )
    for scalar_name in _SCALARS:
        fields.append(
            f"{prefix}_{scalar_name}_mean={stream.scalars[scalar_name].mean():.6f}"
        )
    fields.extend(
        (
            f"{prefix}_causal_disagreement_local_oracle_kappa_corr={stream.causal_local_kappa_corr.correlation():.6f}",
            f"{prefix}_causal_disagreement_required_local_weight_corr={stream.causal_required_local_corr.correlation():.6f}",
            f"{prefix}_replay_observer_samples={stream.replay_observer_count}",
            f"{prefix}_replay_disagreement_local_oracle_kappa_corr={stream.replay_local_kappa_corr.correlation():.6f}",
            f"{prefix}_replay_disagreement_required_local_weight_corr={stream.replay_required_local_corr.correlation():.6f}",
            f"{prefix}_replay_disagreement_local_to_blend_oracle_kappa_corr={stream.replay_blend_kappa_corr.correlation():.6f}",
        )
    )
    return " ".join(fields)


def _debug_summary_with_replay_decomposition(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("replay component shadow was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    if archive is None or not _replay._archive_shadow_only(archive):
        return summary
    aggregate = getattr(archive, _ARCHIVE_COMPONENT_ATTR, None)
    if not isinstance(aggregate, _ReplayComponentAggregate):
        aggregate = _ReplayComponentAggregate()
    return (
        f"{summary} "
        "model_aware_trust_replay_decomposition=loo_component_geometry_shadow "
        "model_aware_trust_replay_decomposition_baseline=blend_corrected_current_replay "
        f"model_aware_trust_replay_decomposition_compute_s={aggregate.compute_seconds:.6f} "
        f"{_stream_summary('audio', aggregate.audio)} "
        f"{_stream_summary('video', aggregate.video)}"
    )


def install_replay_component_decomposition() -> None:
    """Register decomposition state for the composed replay-shadow installer."""
    global _ORIGINAL_REPLAY_VALIDATOR
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_component_shadow_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_native_trust_shadow_installed", False):
        raise RuntimeError("install replay-native trust shadow before component shadow")

    _ORIGINAL_REPLAY_VALIDATOR = _replay._validate_replay_native_shadow
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    SpectrumH3Runtime.debug_summary = _debug_summary_with_replay_decomposition
    SpectrumH3Runtime._replay_component_shadow_installed = True


__all__ = ["install_replay_component_decomposition"]

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import replay_component_shadow as _component
from . import replay_spectral_mixture_shadow as _spectral
from . import replay_trust_shadow as _replay
from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .runtime import SpectrumH3Runtime

_ARCHIVE_ATTR = "_model_aware_trust_replay_spectral_alpha_shadow"
_ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
_NEAR_ZERO_ORACLE_TOLERANCE = 1e-3
_HEADROOM_EPS = 1e-12

_ORIGINAL_SPECTRAL_VALIDATOR: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(slots=True)
class _AlphaStats:
    count: int = 0
    ratio_sum: float = 0.0
    advantage_vs_local_sum: float = 0.0
    advantage_vs_current_sum: float = 0.0
    advantage_vs_oracle_sum: float = 0.0
    weight_sum: float = 0.0
    weight_min: float = math.inf
    weight_max: float = -math.inf
    weight_abs_error_sum: float = 0.0
    weight_squared_error_sum: float = 0.0
    weight_signed_error_sum: float = 0.0
    near_zero_count: int = 0
    near_zero_weight_sum: float = 0.0
    near_zero_abs_error_sum: float = 0.0
    near_zero_squared_error_sum: float = 0.0
    residual_causal_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    residual_validation_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    residual_spectral_gap_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    residual_coordinate_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    residual_current_weight_corr: _trust._RunningPair = field(default_factory=_trust._RunningPair)

    def record(
        self,
        *,
        ratio: float,
        local_ratio: float,
        current_ratio: float,
        oracle_ratio: float,
        predicted_weight: float,
        oracle_weight: float,
        causal_disagreement: float,
        validation_penalty: float,
        spectral_gap: float,
        coordinate: float,
        current_weight: float,
    ) -> None:
        values = (
            ratio,
            local_ratio,
            current_ratio,
            oracle_ratio,
            predicted_weight,
            oracle_weight,
            causal_disagreement,
            validation_penalty,
            spectral_gap,
            coordinate,
            current_weight,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return

        local = max(float(local_ratio), _HEADROOM_EPS)
        current = max(float(current_ratio), _HEADROOM_EPS)
        oracle = max(float(oracle_ratio), _HEADROOM_EPS)
        prediction = float(predicted_weight)
        target = float(oracle_weight)
        signed_error = prediction - target
        abs_error = abs(signed_error)
        squared_error = signed_error * signed_error
        residual = target - prediction

        self.count += 1
        self.ratio_sum += float(ratio)
        self.advantage_vs_local_sum += (float(local_ratio) - float(ratio)) / local
        self.advantage_vs_current_sum += (float(current_ratio) - float(ratio)) / current
        self.advantage_vs_oracle_sum += (float(oracle_ratio) - float(ratio)) / oracle
        self.weight_sum += prediction
        self.weight_min = min(self.weight_min, prediction)
        self.weight_max = max(self.weight_max, prediction)
        self.weight_abs_error_sum += abs_error
        self.weight_squared_error_sum += squared_error
        self.weight_signed_error_sum += signed_error

        self.residual_causal_corr.add(float(causal_disagreement), residual)
        self.residual_validation_corr.add(float(validation_penalty), residual)
        self.residual_spectral_gap_corr.add(float(spectral_gap), residual)
        self.residual_coordinate_corr.add(float(coordinate), residual)
        self.residual_current_weight_corr.add(float(current_weight), residual)

        if target <= _NEAR_ZERO_ORACLE_TOLERANCE:
            self.near_zero_count += 1
            self.near_zero_weight_sum += prediction
            self.near_zero_abs_error_sum += abs_error
            self.near_zero_squared_error_sum += squared_error

    def mean(self, value: float) -> float:
        return float(value) / self.count if self.count else 0.0

    def rmse(self) -> float:
        return math.sqrt(max(0.0, self.mean(self.weight_squared_error_sum)))

    def resolved_weight_min(self) -> float:
        return self.weight_min if self.count else 0.0

    def resolved_weight_max(self) -> float:
        return self.weight_max if self.count else 0.0

    def near_zero_mean(self, value: float) -> float:
        return float(value) / self.near_zero_count if self.near_zero_count else 0.0

    def near_zero_mae_contribution(self) -> float:
        if self.weight_abs_error_sum <= _HEADROOM_EPS:
            return 0.0
        return self.near_zero_abs_error_sum / self.weight_abs_error_sum

    def near_zero_squared_error_contribution(self) -> float:
        if self.weight_squared_error_sum <= _HEADROOM_EPS:
            return 0.0
        return self.near_zero_squared_error_sum / self.weight_squared_error_sum


@dataclass(slots=True)
class _AlphaAggregate:
    compute_seconds: float = 0.0
    count: int = 0
    local_ratio_sum: float = 0.0
    current_ratio_sum: float = 0.0
    oracle_ratio_sum: float = 0.0
    current_weight_sum: float = 0.0
    near_zero_count: int = 0
    near_zero_current_weight_sum: float = 0.0
    alpha: dict[float, _AlphaStats] = field(
        default_factory=lambda: {value: _AlphaStats() for value in _ALPHAS}
    )

    def record(
        self,
        case: dict[str, Any],
        *,
        causal_disagreement: float,
        coordinate: float,
    ) -> None:
        required = (
            float(case["local_ratio"]),
            float(case["current_ratio"]),
            float(case["oracle_ratio"]),
            float(case["oracle_weight"]),
            float(case["current_weight"]),
            float(case["validation_penalty"]),
            float(case["spectral_gap"]),
            float(causal_disagreement),
            float(coordinate),
        )
        if not all(math.isfinite(value) for value in required):
            return

        self.count += 1
        self.local_ratio_sum += float(case["local_ratio"])
        self.current_ratio_sum += float(case["current_ratio"])
        self.oracle_ratio_sum += float(case["oracle_ratio"])
        self.current_weight_sum += float(case["current_weight"])

        if float(case["oracle_weight"]) <= _NEAR_ZERO_ORACLE_TOLERANCE:
            self.near_zero_count += 1
            self.near_zero_current_weight_sum += float(case["current_weight"])

        for alpha in _ALPHAS:
            self.alpha[alpha].record(
                ratio=float(case["alpha_ratios"][alpha]),
                local_ratio=float(case["local_ratio"]),
                current_ratio=float(case["current_ratio"]),
                oracle_ratio=float(case["oracle_ratio"]),
                predicted_weight=float(case["alpha_weights"][alpha]),
                oracle_weight=float(case["oracle_weight"]),
                causal_disagreement=float(causal_disagreement),
                validation_penalty=float(case["validation_penalty"]),
                spectral_gap=float(case["spectral_gap"]),
                coordinate=float(coordinate),
                current_weight=float(case["current_weight"]),
            )

    def mean(self, value: float) -> float:
        return float(value) / self.count if self.count else 0.0

    def headroom_capture(self, alpha: float) -> tuple[bool, float]:
        if self.count <= 0:
            return False, 0.0
        local = self.mean(self.local_ratio_sum)
        oracle = self.mean(self.oracle_ratio_sum)
        candidate = self.alpha[alpha].mean(self.alpha[alpha].ratio_sum)
        available = local - oracle
        if available <= _HEADROOM_EPS:
            return False, 0.0
        return True, (local - candidate) / available

    def best_alpha(self) -> float:
        if self.count <= 0:
            return _ALPHAS[0]
        return min(
            _ALPHAS,
            key=lambda value: (
                self.alpha[value].mean(self.alpha[value].ratio_sum),
                value,
            ),
        )

    def near_zero_fraction(self) -> float:
        return self.near_zero_count / self.count if self.count else 0.0

    def near_zero_current_weight_mean(self) -> float:
        return (
            self.near_zero_current_weight_sum / self.near_zero_count
            if self.near_zero_count
            else 0.0
        )


def _aggregate(archive: Any) -> _AlphaAggregate:
    aggregate = getattr(archive, _ARCHIVE_ATTR, None)
    if not isinstance(aggregate, _AlphaAggregate):
        aggregate = _AlphaAggregate()
        setattr(archive, _ARCHIVE_ATTR, aggregate)
    return aggregate


def _alpha_prediction(
    local: torch.Tensor,
    current_blend: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    resolved = max(0.0, min(1.0, float(alpha)))
    return local + resolved * (current_blend - local)


def _scaled_projected_weight(current_weight: torch.Tensor, alpha: float) -> torch.Tensor:
    return (current_weight * float(alpha)).clamp(0.0, 1.0)


def _alpha_case(
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
        raise RuntimeError("spectral alpha shadow received nonpositive video blend")

    current_weight = _spectral._weight_projection(
        candidates.local,
        candidates.blend_uncorrected,
        candidates.spectral,
    )
    validation_penalty = (float(record.blend_weight) / effective_blends).mean()
    spectral_gap = _trust._tensor_rms(
        candidates.spectral - candidates.local
    ) / _trust._tensor_rms(candidates.local).clamp_min(_trust._EPS)

    alpha_predictions = {
        alpha: _alpha_prediction(
            candidates.local,
            candidates.blend_uncorrected,
            alpha,
        )
        for alpha in _ALPHAS
    }
    alpha_weights = {
        alpha: _scaled_projected_weight(current_weight, alpha)
        for alpha in _ALPHAS
    }

    # Ground truth is withheld until every alpha candidate and predictor observable
    # has been constructed from the retained LOO cache.
    actual = samples[target_index]
    epsilon = _trust._tensor_rms(actual).mul(1e-6).clamp_min(_trust._EPS)
    hold_rms = _trust._tensor_rms(actual - candidates.hold).clamp_min(epsilon)
    local_ratio = _component._ratio(actual, candidates.local, hold_rms)
    current_ratio = _component._ratio(
        actual,
        candidates.blend_uncorrected,
        hold_rms,
    )
    oracle_ratio, oracle_weight = _component._axis_score(
        actual,
        candidates.local,
        candidates.spectral,
        hold_rms,
    )
    alpha_ratios = {
        alpha: _component._ratio(actual, prediction, hold_rms)
        for alpha, prediction in alpha_predictions.items()
    }

    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("local_ratio", local_ratio),
        ("current_ratio", current_ratio),
        ("oracle_ratio", oracle_ratio),
        ("oracle_weight", oracle_weight),
        ("current_weight", current_weight),
        ("validation_penalty", validation_penalty),
        ("spectral_gap", spectral_gap),
        *[
            (f"alpha_ratio:{alpha:.2f}", alpha_ratios[alpha])
            for alpha in _ALPHAS
        ],
        *[
            (f"alpha_weight:{alpha:.2f}", alpha_weights[alpha])
            for alpha in _ALPHAS
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
        "current_ratio": resolved["current_ratio"],
        "oracle_ratio": resolved["oracle_ratio"],
        "oracle_weight": resolved["oracle_weight"],
        "current_weight": resolved["current_weight"],
        "validation_penalty": resolved["validation_penalty"],
        "spectral_gap": resolved["spectral_gap"],
        "alpha_ratios": {
            alpha: resolved[f"alpha_ratio:{alpha:.2f}"]
            for alpha in _ALPHAS
        },
        "alpha_weights": {
            alpha: resolved[f"alpha_weight:{alpha:.2f}"]
            for alpha in _ALPHAS
        },
    }


def _validate_alpha_shadow(
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
                case = _alpha_case(smoother, record, samples, anchor_ids)
                if case is None:
                    continue
                aggregate.record(
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


def _validate_spectral_with_alpha(
    smoother: OfflineSmoother,
    aggregate: _trust._TrustAggregate,
) -> None:
    if _ORIGINAL_SPECTRAL_VALIDATOR is None:
        raise RuntimeError("spectral alpha shadow was not installed correctly")
    failures_before = aggregate.replay_shadow_failures
    _ORIGINAL_SPECTRAL_VALIDATOR(smoother, aggregate)
    if aggregate.replay_shadow_failures != failures_before:
        return
    _validate_alpha_shadow(smoother, aggregate)


def _alpha_suffix(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def _alpha_summary(aggregate: _AlphaAggregate, alpha: float) -> str:
    stats = aggregate.alpha[alpha]
    prefix = f"model_aware_trust_replay_spectral_alpha_video_{_alpha_suffix(alpha)}"
    evaluable, capture = aggregate.headroom_capture(alpha)
    return " ".join(
        (
            f"{prefix}_ratio_mean={stats.mean(stats.ratio_sum):.6f}",
            f"{prefix}_advantage_vs_local_mean={stats.mean(stats.advantage_vs_local_sum):.6f}",
            f"{prefix}_advantage_vs_current_mean={stats.mean(stats.advantage_vs_current_sum):.6f}",
            f"{prefix}_advantage_vs_oracle_mean={stats.mean(stats.advantage_vs_oracle_sum):.6f}",
            f"{prefix}_headroom_capture_evaluable={int(evaluable)}",
            f"{prefix}_headroom_capture_fraction={capture:.6f}",
            f"{prefix}_weight_mean={stats.mean(stats.weight_sum):.6f}",
            f"{prefix}_weight_min={stats.resolved_weight_min():.6f}",
            f"{prefix}_weight_max={stats.resolved_weight_max():.6f}",
            f"{prefix}_weight_mae={stats.mean(stats.weight_abs_error_sum):.6f}",
            f"{prefix}_weight_rmse={stats.rmse():.6f}",
            f"{prefix}_weight_bias={stats.mean(stats.weight_signed_error_sum):.6f}",
        )
    )


def _video_summary(aggregate: _AlphaAggregate) -> str:
    prefix = "model_aware_trust_replay_spectral_alpha_video"
    best = aggregate.best_alpha()
    best_stats = aggregate.alpha[best]
    evaluable, capture = aggregate.headroom_capture(best)
    fields = [
        f"{prefix}_samples={aggregate.count}",
        f"{prefix}_local_ratio_mean={aggregate.mean(aggregate.local_ratio_sum):.6f}",
        f"{prefix}_current_ratio_mean={aggregate.mean(aggregate.current_ratio_sum):.6f}",
        f"{prefix}_oracle_ratio_mean={aggregate.mean(aggregate.oracle_ratio_sum):.6f}",
        f"{prefix}_current_weight_mean={aggregate.mean(aggregate.current_weight_sum):.6f}",
        f"{prefix}_best_alpha={best:.2f}",
        f"{prefix}_best_ratio_mean={best_stats.mean(best_stats.ratio_sum):.6f}",
        f"{prefix}_best_headroom_capture_evaluable={int(evaluable)}",
        f"{prefix}_best_headroom_capture_fraction={capture:.6f}",
        f"{prefix}_best_weight_mae={best_stats.mean(best_stats.weight_abs_error_sum):.6f}",
        f"{prefix}_best_weight_rmse={best_stats.rmse():.6f}",
        f"{prefix}_best_weight_bias={best_stats.mean(best_stats.weight_signed_error_sum):.6f}",
        f"{prefix}_best_residual_causal_disagreement_corr={best_stats.residual_causal_corr.correlation():.6f}",
        f"{prefix}_best_residual_validation_penalty_corr={best_stats.residual_validation_corr.correlation():.6f}",
        f"{prefix}_best_residual_spectral_gap_corr={best_stats.residual_spectral_gap_corr.correlation():.6f}",
        f"{prefix}_best_residual_coordinate_corr={best_stats.residual_coordinate_corr.correlation():.6f}",
        f"{prefix}_best_residual_current_weight_corr={best_stats.residual_current_weight_corr.correlation():.6f}",
        f"{prefix}_oracle_weight_near_zero_tolerance={_NEAR_ZERO_ORACLE_TOLERANCE:.6f}",
        f"{prefix}_oracle_weight_near_zero_count={aggregate.near_zero_count}",
        f"{prefix}_oracle_weight_near_zero_fraction={aggregate.near_zero_fraction():.6f}",
        f"{prefix}_oracle_weight_near_zero_current_weight_mean={aggregate.near_zero_current_weight_mean():.6f}",
        f"{prefix}_oracle_weight_near_zero_best_scaled_weight_mean={best_stats.near_zero_mean(best_stats.near_zero_weight_sum):.6f}",
        f"{prefix}_oracle_weight_near_zero_best_abs_weight_error_mean={best_stats.near_zero_mean(best_stats.near_zero_abs_error_sum):.6f}",
        f"{prefix}_oracle_weight_near_zero_best_mae_contribution={best_stats.near_zero_mae_contribution():.6f}",
        f"{prefix}_oracle_weight_near_zero_best_squared_error_contribution={best_stats.near_zero_squared_error_contribution():.6f}",
    ]
    fields.extend(_alpha_summary(aggregate, alpha) for alpha in _ALPHAS)
    return " ".join(fields)


def _debug_summary_with_alpha(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("spectral alpha shadow was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    if archive is None or not _replay._archive_shadow_only(archive):
        return summary
    aggregate = getattr(archive, _ARCHIVE_ATTR, None)
    if not isinstance(aggregate, _AlphaAggregate):
        aggregate = _AlphaAggregate()
    return (
        f"{summary} "
        "model_aware_trust_replay_spectral_alpha="
        "video_current_blend_multiplicative_shadow "
        "model_aware_trust_replay_spectral_alpha_applied=0 "
        "model_aware_trust_replay_spectral_alpha_sweep=0p00_0p25_0p50_0p75_1p00 "
        "model_aware_trust_replay_spectral_alpha_selection="
        "lowest_mean_hidden_feature_ratio_then_lower_alpha "
        "model_aware_trust_replay_spectral_alpha_headroom_capture="
        "ratio_of_means_local_minus_candidate_over_local_minus_oracle "
        f"model_aware_trust_replay_spectral_alpha_compute_s={aggregate.compute_seconds:.6f} "
        f"{_video_summary(aggregate)}"
    )


def install_replay_spectral_alpha_shadow() -> None:
    """Install video-only shadow calibration of current replay spectral placement."""
    global _ORIGINAL_SPECTRAL_VALIDATOR
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_spectral_alpha_shadow_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_spectral_mixture_shadow_installed", False):
        raise RuntimeError("install replay spectral mixture shadow before alpha shadow")

    _ORIGINAL_SPECTRAL_VALIDATOR = _spectral._validate_spectral_mixture_shadow
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    _spectral._validate_spectral_mixture_shadow = _validate_spectral_with_alpha
    SpectrumH3Runtime.debug_summary = _debug_summary_with_alpha
    SpectrumH3Runtime._replay_spectral_alpha_shadow_installed = True


__all__ = ["install_replay_spectral_alpha_shadow"]

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .forecast import HistoryWeightForecaster
from .runtime import SpectrumH3Runtime

_REPLAY_KAPPAS = (0.50, 0.70, 0.85, 0.95, 1.00)
_ARCHIVE_SHADOW_ONLY_ATTR = "_model_aware_trust_replay_shadow_only"
_ARCHIVE_NATIVE_SHADOW_ATTR = "_model_aware_trust_replay_native_shadow"

_ORIGINAL_BEGIN_OFFLINE_CAPTURE: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(slots=True)
class _ReplayNativeStream:
    count: int = 0
    baseline_ratio_sum: float = 0.0
    local_ratio_sum: float = 0.0
    causal_transfer_ratio_sum: float = 0.0
    causal_transfer_advantage_sum: float = 0.0
    oracle_ratio_sum: float = 0.0
    oracle_advantage_sum: float = 0.0
    oracle_kappa_sum: float = 0.0
    oracle_kappa_min: float = 1.0
    oracle_kappa_max: float = 0.0
    local_oracle_ratio_sum: float = 0.0
    local_oracle_advantage_sum: float = 0.0
    local_oracle_kappa_sum: float = 0.0
    local_oracle_kappa_min: float = 1.0
    local_oracle_kappa_max: float = 0.0
    effective_blend_sum: float = 0.0
    effective_blend_min: float = 1.0
    effective_blend_max: float = 0.0
    causal_error_correlation: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    causal_shrink_correlation: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    replay_observer_count: int = 0
    replay_disagreement_sum: float = 0.0
    replay_disagreement_max: float = 0.0
    replay_error_correlation: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    replay_shrink_correlation: _trust._RunningPair = field(default_factory=_trust._RunningPair)
    candidate_ratio_sums: dict[float, float] = field(
        default_factory=lambda: {kappa: 0.0 for kappa in _REPLAY_KAPPAS}
    )
    candidate_advantage_sums: dict[float, float] = field(
        default_factory=lambda: {kappa: 0.0 for kappa in _REPLAY_KAPPAS}
    )

    def record(
        self,
        *,
        baseline_ratio: float,
        local_ratio: float,
        causal_transfer_ratio: float,
        oracle_ratio: float,
        oracle_kappa: float,
        local_oracle_ratio: float,
        local_oracle_kappa: float,
        effective_blend_mean: float,
        effective_blend_min: float,
        effective_blend_max: float,
        causal_disagreement: float,
        candidate_ratios: dict[float, float],
        replay_disagreement: float | None,
    ) -> None:
        values = (
            baseline_ratio,
            local_ratio,
            causal_transfer_ratio,
            oracle_ratio,
            oracle_kappa,
            local_oracle_ratio,
            local_oracle_kappa,
            effective_blend_mean,
            effective_blend_min,
            effective_blend_max,
            causal_disagreement,
            *[candidate_ratios[kappa] for kappa in _REPLAY_KAPPAS],
        )
        if not all(math.isfinite(float(value)) for value in values):
            return

        baseline = max(float(baseline_ratio), 1e-12)
        oracle_shrink = 1.0 - float(oracle_kappa)
        self.count += 1
        self.baseline_ratio_sum += float(baseline_ratio)
        self.local_ratio_sum += float(local_ratio)
        self.causal_transfer_ratio_sum += float(causal_transfer_ratio)
        self.causal_transfer_advantage_sum += (
            float(baseline_ratio) - float(causal_transfer_ratio)
        ) / baseline
        self.oracle_ratio_sum += float(oracle_ratio)
        self.oracle_advantage_sum += (
            float(baseline_ratio) - float(oracle_ratio)
        ) / baseline
        self.oracle_kappa_sum += float(oracle_kappa)
        self.oracle_kappa_min = min(self.oracle_kappa_min, float(oracle_kappa))
        self.oracle_kappa_max = max(self.oracle_kappa_max, float(oracle_kappa))
        self.local_oracle_ratio_sum += float(local_oracle_ratio)
        self.local_oracle_advantage_sum += (
            float(baseline_ratio) - float(local_oracle_ratio)
        ) / baseline
        self.local_oracle_kappa_sum += float(local_oracle_kappa)
        self.local_oracle_kappa_min = min(
            self.local_oracle_kappa_min, float(local_oracle_kappa)
        )
        self.local_oracle_kappa_max = max(
            self.local_oracle_kappa_max, float(local_oracle_kappa)
        )
        self.effective_blend_sum += float(effective_blend_mean)
        self.effective_blend_min = min(
            self.effective_blend_min, float(effective_blend_min)
        )
        self.effective_blend_max = max(
            self.effective_blend_max, float(effective_blend_max)
        )
        self.causal_error_correlation.add(causal_disagreement, baseline_ratio)
        self.causal_shrink_correlation.add(causal_disagreement, oracle_shrink)

        for kappa in _REPLAY_KAPPAS:
            ratio = float(candidate_ratios[kappa])
            self.candidate_ratio_sums[kappa] += ratio
            self.candidate_advantage_sums[kappa] += (
                float(baseline_ratio) - ratio
            ) / baseline

        if replay_disagreement is not None and math.isfinite(float(replay_disagreement)):
            value = float(replay_disagreement)
            self.replay_observer_count += 1
            self.replay_disagreement_sum += value
            self.replay_disagreement_max = max(self.replay_disagreement_max, value)
            self.replay_error_correlation.add(value, baseline_ratio)
            self.replay_shrink_correlation.add(value, oracle_shrink)

    def mean(self, value: float) -> float:
        return float(value) / self.count if self.count else 0.0

    def observer_mean(self, value: float) -> float:
        return float(value) / self.replay_observer_count if self.replay_observer_count else 0.0

    def resolved_oracle_kappa_min(self) -> float:
        return self.oracle_kappa_min if self.count else 0.0

    def resolved_oracle_kappa_max(self) -> float:
        return self.oracle_kappa_max if self.count else 0.0

    def resolved_local_oracle_kappa_min(self) -> float:
        return self.local_oracle_kappa_min if self.count else 0.0

    def resolved_local_oracle_kappa_max(self) -> float:
        return self.local_oracle_kappa_max if self.count else 0.0

    def resolved_effective_blend_min(self) -> float:
        return self.effective_blend_min if self.count else 0.0

    def resolved_effective_blend_max(self) -> float:
        return self.effective_blend_max if self.count else 0.0


@dataclass(slots=True)
class _ReplayNativeAggregate:
    compute_seconds: float = 0.0
    audio: _ReplayNativeStream = field(default_factory=_ReplayNativeStream)
    video: _ReplayNativeStream = field(default_factory=_ReplayNativeStream)


def _archive_shadow_only(archive: Any) -> bool:
    return bool(getattr(archive, _ARCHIVE_SHADOW_ONLY_ATTR, False))


def _native_aggregate(archive: Any) -> _ReplayNativeAggregate:
    aggregate = getattr(archive, _ARCHIVE_NATIVE_SHADOW_ATTR, None)
    if not isinstance(aggregate, _ReplayNativeAggregate):
        aggregate = _ReplayNativeAggregate()
        setattr(archive, _ARCHIVE_NATIVE_SHADOW_ATTR, aggregate)
    return aggregate


def _begin_offline_capture_with_replay_shadow(
    self: SpectrumH3Runtime,
    *,
    total_steps: int,
    sampler_name: str,
) -> None:
    if _ORIGINAL_BEGIN_OFFLINE_CAPTURE is None:
        raise RuntimeError("replay-native trust shadow was not installed correctly")
    _ORIGINAL_BEGIN_OFFLINE_CAPTURE(
        self,
        total_steps=total_steps,
        sampler_name=sampler_name,
    )
    archive = getattr(self, "_offline_archive", None)
    if archive is not None:
        setattr(archive, _ARCHIVE_SHADOW_ONLY_ATTR, bool(_trust._trust_requested(self)))


def _spectral_prediction(
    smoother: OfflineSmoother,
    samples: torch.Tensor,
    anchor_ids: list[int],
    retained: list[int],
    coordinate: float,
    *,
    degree: int,
    ridge_lambda: float,
    model_aware: bool,
) -> torch.Tensor:
    forecaster = HistoryWeightForecaster(
        degree=degree,
        ridge_lambda=ridge_lambda,
        max_history=max(len(retained), degree + 1, 2),
        history_storage="system_ram",
    )
    for index in retained:
        forecaster.update(
            smoother.archive.anchors[index].coordinate,
            samples[index],
            anchor_id=anchor_ids[index],
            take_ownership=True,
        )
    if model_aware:
        weights = forecaster.model_aware_weights(
            coordinate,
            1.0,
            degree=degree,
            ridge_lambda=ridge_lambda,
            correction_gain=0.0,
        )
    else:
        # OfflineSmoother._build_validation_scores uses its configured spectral
        # forecaster here, independent of any per-step model-aware degree/ridge.
        weights = forecaster.spectral_weights(coordinate)
    spectral = smoother._affine_spectral_weights(weights)
    return torch.einsum("k,kbs->bs", spectral, samples[retained])


def _retained_validation_scores(
    smoother: OfflineSmoother,
    samples: torch.Tensor,
    anchor_ids: list[int],
    retained: list[int],
    retained_position: int,
) -> list[float] | None:
    if retained_position <= 0 or retained_position >= len(retained) - 1:
        return None

    target_original_index = retained[retained_position]
    validation_retained = [
        index for position, index in enumerate(retained) if position != retained_position
    ]
    if len(validation_retained) < max(2, smoother.degree + 1):
        return None

    target_anchor = smoother.archive.anchors[target_original_index]
    spectral_prediction = _spectral_prediction(
        smoother,
        samples,
        anchor_ids,
        validation_retained,
        target_anchor.coordinate,
        degree=smoother.degree,
        ridge_lambda=smoother.ridge_lambda,
        model_aware=False,
    )

    left_original_index = retained[retained_position - 1]
    right_original_index = retained[retained_position + 1]
    left = smoother.archive.anchors[left_original_index]
    right = smoother.archive.anchors[right_original_index]
    spacing = right.coordinate - left.coordinate
    if abs(spacing) <= 1e-12:
        raise RuntimeError("replay-native validation anchors have duplicate coordinates")
    ratio = (target_anchor.coordinate - left.coordinate) / spacing
    local_prediction = torch.lerp(
        samples[left_original_index], samples[right_original_index], ratio
    )
    actual = samples[target_original_index]

    scores: list[float] = []
    for branch in range(int(actual.shape[0])):
        spectral_rms = _trust._tensor_rms(spectral_prediction[branch] - actual[branch])
        local_rms = _trust._tensor_rms(local_prediction[branch] - actual[branch])
        epsilon = _trust._tensor_rms(actual[branch]).mul(1e-6).clamp_min(_trust._EPS)
        if float(spectral_rms) <= float(epsilon) and float(local_rms) <= float(epsilon):
            score = 0.0
        else:
            score = float((spectral_rms / local_rms.clamp_min(epsilon)).item())
        if not math.isfinite(score):
            raise RuntimeError("replay-native validation score is nonfinite")
        scores.append(score)
    return scores


def _effective_blends_for_withheld_target(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
    retained: list[int],
    left_position: int,
    right_position: int,
) -> torch.Tensor:
    left_scores = _retained_validation_scores(
        smoother,
        samples,
        anchor_ids,
        retained,
        left_position,
    )
    right_scores = _retained_validation_scores(
        smoother,
        samples,
        anchor_ids,
        retained,
        right_position,
    )

    effective = []
    for branch in range(int(samples.shape[1])):
        nearby = []
        if left_scores is not None:
            nearby.append(left_scores[branch])
        if right_scores is not None:
            nearby.append(right_scores[branch])
        validation_score = max(nearby, default=1.0)
        effective.append(record.blend_weight / max(1.0, validation_score))
    return torch.tensor(effective, dtype=torch.float32)


def _replay_shadow_case(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
) -> dict[str, Any] | None:
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
        raise RuntimeError("replay-native LOO target leaked into retained anchors")

    spectral_prediction = _spectral_prediction(
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
        raise RuntimeError("replay-native shadow bracket has duplicate coordinates")
    ratio = (record.coordinate - left.coordinate) / spacing

    retained_samples = samples[retained]
    local_weights = torch.zeros(len(retained), dtype=torch.float32)
    local_weights[left_position] = 1.0 - ratio
    local_weights[right_position] = ratio
    local_prediction = torch.einsum("k,kbs->bs", local_weights, retained_samples)

    effective_blends = _effective_blends_for_withheld_target(
        smoother,
        record,
        samples,
        anchor_ids,
        retained,
        left_position,
        right_position,
    )
    baseline = (
        effective_blends[:, None] * spectral_prediction
        + (1.0 - effective_blends[:, None]) * local_prediction
    )
    if record.correction_gain != 0.0:
        correction_delta = (
            retained_samples[right_position] - retained_samples[left_position]
        )
        baseline = baseline + float(record.correction_gain) * correction_delta

    latest_position = retained_ids.index(record.latest_anchor_id)
    hold = retained_samples[latest_position]
    causal_transfer = hold + float(record.kappa) * (baseline - hold)
    fixed_predictions = {
        kappa: hold + float(kappa) * (baseline - hold)
        for kappa in _REPLAY_KAPPAS
    }

    # The target is read only after the complete candidate and validation state has
    # been constructed from the retained cache.
    actual = samples[target_index]
    epsilon = _trust._tensor_rms(actual).mul(1e-6).clamp_min(_trust._EPS)
    hold_rms = _trust._tensor_rms(actual - hold).clamp_min(epsilon)
    baseline_ratio = _trust._tensor_rms(actual - baseline) / hold_rms
    local_ratio = _trust._tensor_rms(actual - local_prediction) / hold_rms
    causal_transfer_ratio = _trust._tensor_rms(actual - causal_transfer) / hold_rms

    oracle_kappa = _trust.oracle_segment_kappa(actual, hold, baseline)
    oracle = hold + oracle_kappa.to(dtype=baseline.dtype) * (baseline - hold)
    oracle_ratio = _trust._tensor_rms(actual - oracle) / hold_rms

    local_oracle_kappa = _trust.oracle_segment_kappa(actual, local_prediction, baseline)
    local_oracle = local_prediction + local_oracle_kappa.to(dtype=baseline.dtype) * (
        baseline - local_prediction
    )
    local_oracle_ratio = _trust._tensor_rms(actual - local_oracle) / hold_rms

    candidate_ratios = {
        kappa: _trust._tensor_rms(actual - prediction) / hold_rms
        for kappa, prediction in fixed_predictions.items()
    }
    replay_disagreement = None
    if record.blend_weight > 1e-12:
        replay_disagreement = _trust._tensor_rms(
            spectral_prediction - local_prediction
        ) / _trust._tensor_rms(baseline).clamp_min(_trust._EPS)

    values = torch.stack(
        (
            baseline_ratio,
            local_ratio,
            causal_transfer_ratio,
            oracle_ratio,
            oracle_kappa,
            local_oracle_ratio,
            local_oracle_kappa,
            effective_blends.mean(),
            effective_blends.min(),
            effective_blends.max(),
            *[candidate_ratios[kappa] for kappa in _REPLAY_KAPPAS],
            *(() if replay_disagreement is None else (replay_disagreement,)),
        )
    ).detach().to(device="cpu", dtype=torch.float32).tolist()

    cursor = 0
    result: dict[str, Any] = {
        "retained_anchor_ids": tuple(retained_ids),
        "baseline_ratio": float(values[cursor]),
        "local_ratio": float(values[cursor + 1]),
        "causal_transfer_ratio": float(values[cursor + 2]),
        "oracle_ratio": float(values[cursor + 3]),
        "oracle_kappa": float(values[cursor + 4]),
        "local_oracle_ratio": float(values[cursor + 5]),
        "local_oracle_kappa": float(values[cursor + 6]),
        "effective_blend_mean": float(values[cursor + 7]),
        "effective_blend_min": float(values[cursor + 8]),
        "effective_blend_max": float(values[cursor + 9]),
    }
    cursor += 10
    result["candidate_ratios"] = {
        kappa: float(values[cursor + index])
        for index, kappa in enumerate(_REPLAY_KAPPAS)
    }
    cursor += len(_REPLAY_KAPPAS)
    result["replay_disagreement"] = (
        None if replay_disagreement is None else float(values[cursor])
    )
    return result


def _validate_replay_native_shadow(
    smoother: OfflineSmoother,
    aggregate: _trust._TrustAggregate,
) -> None:
    records = getattr(
        smoother.archive,
        "_model_aware_trust_replay_shadow_records",
        None,
    )
    if not isinstance(records, list) or not records:
        return

    native = _native_aggregate(smoother.archive)
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
            aggregate.replay_shadow_failures += 1
            return

        for record in records:
            if not isinstance(record, _trust._ReplayShadowRecord):
                continue
            samples = samples_by_stream.get(record.stream_name)
            if samples is None:
                continue
            try:
                case = _replay_shadow_case(smoother, record, samples, anchor_ids)
                if case is None:
                    continue
                stream = native.audio if record.stream_name == "audio" else native.video
                stream.record(
                    baseline_ratio=case["baseline_ratio"],
                    local_ratio=case["local_ratio"],
                    causal_transfer_ratio=case["causal_transfer_ratio"],
                    oracle_ratio=case["oracle_ratio"],
                    oracle_kappa=case["oracle_kappa"],
                    local_oracle_ratio=case["local_oracle_ratio"],
                    local_oracle_kappa=case["local_oracle_kappa"],
                    effective_blend_mean=case["effective_blend_mean"],
                    effective_blend_min=case["effective_blend_min"],
                    effective_blend_max=case["effective_blend_max"],
                    causal_disagreement=record.disagreement,
                    candidate_ratios=case["candidate_ratios"],
                    replay_disagreement=case["replay_disagreement"],
                )

                # Keep the original replay-shadow fields as a counterfactual record
                # of the now-rejected causal-kappa transfer, without applying it.
                legacy = (
                    aggregate.replay_shadow_audio
                    if record.stream_name == "audio"
                    else aggregate.replay_shadow_video
                )
                legacy.record(case["baseline_ratio"], case["causal_transfer_ratio"])
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
                aggregate.replay_shadow_failures += 1
    finally:
        native.compute_seconds += time.perf_counter() - started


def _offline_build_forecast_weights_shadow_only(
    self: OfflineSmoother,
):
    base_builder = _trust._ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS
    if base_builder is None:
        raise RuntimeError("causal trust base replay builder is unavailable")

    # Bypass the rejected causal-kappa replay wrapper for every OfflineSmoother.
    # With trust disabled this is baseline-identical because base_builder is the
    # existing replay builder containing the PR #39 generic correction.
    weights_by_step = base_builder(self)
    aggregate = getattr(self.archive, "_model_aware_trust_aggregate", None)
    if isinstance(aggregate, _trust._TrustAggregate):
        _validate_replay_native_shadow(self, aggregate)
        self._model_aware_trust_aggregate = aggregate
    return weights_by_step


def _kappa_suffix(kappa: float) -> str:
    return f"{kappa:.2f}".replace(".", "p")


def _native_stream_summary(name: str, stream: _ReplayNativeStream) -> str:
    observer_mode = (
        "spectral_vs_local" if stream.replay_observer_count else "inactive_no_spectral_blend"
    )
    # baseline_ratio_mean is emitted once by the wrapped trust summary. The native
    # validator populates that legacy accumulator with this same replay baseline.
    fields = [
        f"model_aware_trust_replay_shadow_{name}_oracle_ratio_mean={stream.mean(stream.oracle_ratio_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_oracle_advantage_mean={stream.mean(stream.oracle_advantage_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_oracle_kappa_mean={stream.mean(stream.oracle_kappa_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_oracle_kappa_min={stream.resolved_oracle_kappa_min():.6f}",
        f"model_aware_trust_replay_shadow_{name}_oracle_kappa_max={stream.resolved_oracle_kappa_max():.6f}",
        f"model_aware_trust_replay_shadow_{name}_effective_blend_mean={stream.mean(stream.effective_blend_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_effective_blend_min={stream.resolved_effective_blend_min():.6f}",
        f"model_aware_trust_replay_shadow_{name}_effective_blend_max={stream.resolved_effective_blend_max():.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_ratio_mean={stream.mean(stream.local_ratio_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_oracle_ratio_mean={stream.mean(stream.local_oracle_ratio_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_oracle_advantage_mean={stream.mean(stream.local_oracle_advantage_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_oracle_kappa_mean={stream.mean(stream.local_oracle_kappa_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_oracle_kappa_min={stream.resolved_local_oracle_kappa_min():.6f}",
        f"model_aware_trust_replay_shadow_{name}_local_oracle_kappa_max={stream.resolved_local_oracle_kappa_max():.6f}",
        f"model_aware_trust_replay_shadow_{name}_causal_transfer_ratio_mean={stream.mean(stream.causal_transfer_ratio_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_causal_transfer_advantage_mean={stream.mean(stream.causal_transfer_advantage_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_causal_disagreement_error_corr={stream.causal_error_correlation.correlation():.6f}",
        f"model_aware_trust_replay_shadow_{name}_causal_disagreement_shrink_corr={stream.causal_shrink_correlation.correlation():.6f}",
        f"model_aware_trust_replay_shadow_{name}_observer={observer_mode}",
        f"model_aware_trust_replay_shadow_{name}_observer_samples={stream.replay_observer_count}",
        f"model_aware_trust_replay_shadow_{name}_replay_disagreement_mean={stream.observer_mean(stream.replay_disagreement_sum):.6f}",
        f"model_aware_trust_replay_shadow_{name}_replay_disagreement_max={stream.replay_disagreement_max:.6f}",
        f"model_aware_trust_replay_shadow_{name}_replay_disagreement_error_corr={stream.replay_error_correlation.correlation():.6f}",
        f"model_aware_trust_replay_shadow_{name}_replay_disagreement_shrink_corr={stream.replay_shrink_correlation.correlation():.6f}",
    ]
    for kappa in _REPLAY_KAPPAS:
        suffix = _kappa_suffix(kappa)
        fields.extend(
            (
                f"model_aware_trust_replay_shadow_{name}_kappa_{suffix}_ratio_mean={stream.mean(stream.candidate_ratio_sums[kappa]):.6f}",
                f"model_aware_trust_replay_shadow_{name}_kappa_{suffix}_advantage_mean={stream.mean(stream.candidate_advantage_sums[kappa]):.6f}",
            )
        )
    return " ".join(fields)


def _debug_summary_with_replay_shadow(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("replay-native trust shadow was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    if archive is None or not _archive_shadow_only(archive):
        return summary

    summary = summary.replace(
        "model_aware_trust_path=offline_replay_causal_kappa_transfer",
        "model_aware_trust_path=offline_replay_shadow_only",
    ).replace(
        "model_aware_trust_replay_shadow=loo_unattenuated_future_bracket",
        "model_aware_trust_replay_shadow=loo_validation_attenuated_replay_native_calibration",
    )
    native = getattr(archive, _ARCHIVE_NATIVE_SHADOW_ATTR, None)
    if not isinstance(native, _ReplayNativeAggregate):
        native = _ReplayNativeAggregate()
    return (
        f"{summary} "
        "model_aware_trust_replay_application=disabled_rejected_causal_transfer "
        "model_aware_trust_replay_shadow_reference=validation_attenuated_corrected_future_bracket "
        f"model_aware_trust_replay_shadow_compute_s={native.compute_seconds:.6f} "
        f"{_native_stream_summary('audio', native.audio)} "
        f"{_native_stream_summary('video', native.video)}"
    )


def install_replay_native_trust_shadow() -> None:
    """Disable causal-kappa replay transfer and install replay-native shadow telemetry."""
    global _ORIGINAL_BEGIN_OFFLINE_CAPTURE
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_native_trust_shadow_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_forecast_trust_probe_installed", False):
        raise RuntimeError("install forecast trust before replay-native trust shadow")

    _ORIGINAL_BEGIN_OFFLINE_CAPTURE = SpectrumH3Runtime.begin_offline_capture
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary

    SpectrumH3Runtime.begin_offline_capture = _begin_offline_capture_with_replay_shadow
    OfflineSmoother._build_forecast_weights = _offline_build_forecast_weights_shadow_only
    SpectrumH3Runtime.debug_summary = _debug_summary_with_replay_shadow
    SpectrumH3Runtime._replay_native_trust_shadow_installed = True


__all__ = ["install_replay_native_trust_shadow"]

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import generic_correction as _generic
from .experiments import OfflineSmoother
from .forecast import HistoryWeightForecaster
from .model_aware import AnchorEvidence, ModelAwareForecastDecision
from .runtime import SpectrumH3Runtime

_EPS = torch.finfo(torch.float32).eps
_CORRECTION_GAIN_LIMIT = 0.25
_RACER_A = 0.3
_RACER_B = 4.0
_APPLIED_THETA = 0.15
_TRUST_THETAS = (0.15, 0.25, 0.40)
_REPLAY_SHADOW_SAMPLE_ELEMENTS = 4096

_ORIGINAL_GENERIC_ANCHOR_EVIDENCE: Callable[..., AnchorEvidence | None] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None
_ORIGINAL_RUNTIME_PREDICTION_SEGMENTS: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_MODEL_AWARE_WEIGHT_SEGMENTS: Callable[..., Any] | None = None
_ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS: Callable[..., Any] | None = None


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


def _stable_sigmoid(value: float) -> float:
    resolved = float(value)
    if resolved >= 0.0:
        exp_term = math.exp(-min(resolved, 80.0))
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(max(resolved, -80.0))
    return exp_term / (1.0 + exp_term)


def trust_kappa(
    disagreement: float,
    horizon: float,
    *,
    theta: float,
    a: float = _RACER_A,
    b: float = _RACER_B,
) -> float:
    """Return the bounded scalar trust coefficient for one stream."""
    risk = max(0.0, float(disagreement))
    horizon_decay = math.exp(-float(a) * max(float(horizon) - 1.0, 0.0))
    confidence = _stable_sigmoid(float(b) * (float(theta) - risk))
    return max(0.0, min(1.0, horizon_decay * confidence))


def apply_trust_to_history_weights(
    weights: torch.Tensor,
    kappa: float,
    latest_anchor_index: int,
) -> torch.Tensor:
    """Shrink affine history weights toward one retained causal anchor."""
    if (
        not torch.is_tensor(weights)
        or weights.ndim != 1
        or weights.numel() == 0
        or not weights.dtype.is_floating_point
    ):
        raise ValueError("trust shrinkage requires a non-empty floating-point weight vector")
    resolved_kappa = float(kappa)
    if not math.isfinite(resolved_kappa) or not 0.0 <= resolved_kappa <= 1.0:
        raise ValueError("trust kappa must be finite and in [0, 1]")
    index = int(latest_anchor_index)
    if index < 0:
        index += int(weights.numel())
    if index < 0 or index >= int(weights.numel()):
        raise ValueError("latest causal anchor is outside the retained history weights")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("trust shrinkage received nonfinite history weights")

    if resolved_kappa >= 1.0:
        return weights.clone()
    result = weights.clone()
    if resolved_kappa <= 0.0:
        result.zero_()
        result[index] = 1.0
        return result
    result.mul_(resolved_kappa)
    result[index] += 1.0 - resolved_kappa
    return result


def oracle_segment_kappa(
    actual: torch.Tensor,
    latest: torch.Tensor,
    proposal: torch.Tensor,
) -> torch.Tensor:
    """Best [latest, proposal] interpolation coefficient for diagnostic use."""
    target = (actual - latest).reshape(-1).to(torch.float32)
    direction = (proposal - latest).reshape(-1).to(torch.float32)
    actual_rms = _tensor_rms(actual)
    epsilon = actual_rms.mul(1e-6).clamp_min(_EPS)
    denominator = torch.dot(direction, direction).clamp_min(
        epsilon.square() * max(1, int(direction.numel()))
    )
    return (torch.dot(target, direction) / denominator).clamp(0.0, 1.0)


@dataclass(slots=True)
class _RunningPair:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            return
        self.count += 1
        self.sum_x += x_value
        self.sum_y += y_value
        self.sum_x2 += x_value * x_value
        self.sum_y2 += y_value * y_value
        self.sum_xy += x_value * y_value

    def correlation(self) -> float:
        if self.count < 2:
            return 0.0
        n = float(self.count)
        covariance = n * self.sum_xy - self.sum_x * self.sum_y
        variance_x = n * self.sum_x2 - self.sum_x * self.sum_x
        variance_y = n * self.sum_y2 - self.sum_y * self.sum_y
        denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
        if denominator <= 1e-12:
            return 0.0
        result = covariance / denominator
        return max(-1.0, min(1.0, result)) if math.isfinite(result) else 0.0


@dataclass(slots=True)
class _StreamProbe:
    count: int = 0
    horizon_sum: float = 0.0
    horizon_max: float = 0.0
    disagreement_sum: float = 0.0
    disagreement_max: float = 0.0
    corrected_ratio_sum: float = 0.0
    delta_bound_gain_sum: float = 0.0
    delta_bound_ratio_sum: float = 0.0
    delta_bound_advantage_sum: float = 0.0
    oracle_ratio_sum: float = 0.0
    oracle_kappa_sum: float = 0.0
    oracle_advantage_sum: float = 0.0
    error_correlation: _RunningPair = field(default_factory=_RunningPair)
    shrink_correlation: _RunningPair = field(default_factory=_RunningPair)
    candidate_ratio_sums: dict[float, float] = field(
        default_factory=lambda: {theta: 0.0 for theta in _TRUST_THETAS}
    )
    candidate_advantage_sums: dict[float, float] = field(
        default_factory=lambda: {theta: 0.0 for theta in _TRUST_THETAS}
    )

    def record(
        self,
        *,
        horizon: float,
        disagreement: float,
        corrected_ratio: float,
        delta_bound_gain: float,
        delta_bound_ratio: float,
        oracle_ratio: float,
        oracle_kappa: float,
        candidate_ratios: dict[float, float],
    ) -> None:
        values = (
            float(horizon),
            float(disagreement),
            float(corrected_ratio),
            float(delta_bound_gain),
            float(delta_bound_ratio),
            float(oracle_ratio),
            float(oracle_kappa),
            *[float(candidate_ratios[theta]) for theta in _TRUST_THETAS],
        )
        if not all(math.isfinite(value) for value in values):
            return
        baseline = max(float(corrected_ratio), 1e-12)
        delta_bound_advantage = (
            float(corrected_ratio) - float(delta_bound_ratio)
        ) / baseline
        oracle_advantage = (float(corrected_ratio) - float(oracle_ratio)) / baseline
        self.count += 1
        self.horizon_sum += float(horizon)
        self.horizon_max = max(self.horizon_max, float(horizon))
        self.disagreement_sum += float(disagreement)
        self.disagreement_max = max(self.disagreement_max, float(disagreement))
        self.corrected_ratio_sum += float(corrected_ratio)
        self.delta_bound_gain_sum += float(delta_bound_gain)
        self.delta_bound_ratio_sum += float(delta_bound_ratio)
        self.delta_bound_advantage_sum += delta_bound_advantage
        self.oracle_ratio_sum += float(oracle_ratio)
        self.oracle_kappa_sum += float(oracle_kappa)
        self.oracle_advantage_sum += oracle_advantage
        self.error_correlation.add(float(disagreement), float(corrected_ratio))
        self.shrink_correlation.add(float(disagreement), 1.0 - float(oracle_kappa))
        for theta in _TRUST_THETAS:
            ratio = float(candidate_ratios[theta])
            self.candidate_ratio_sums[theta] += ratio
            self.candidate_advantage_sums[theta] += (
                float(corrected_ratio) - ratio
            ) / baseline

    def mean(self, value: float) -> float:
        return float(value) / self.count if self.count else 0.0


@dataclass(slots=True)
class _AppliedStream:
    count: int = 0
    disagreement_sum: float = 0.0
    disagreement_max: float = 0.0
    kappa_sum: float = 0.0
    kappa_min: float = 1.0
    kappa_max: float = 0.0

    def record(self, disagreement: float, kappa: float) -> None:
        risk = float(disagreement)
        trust = float(kappa)
        if not math.isfinite(risk) or not math.isfinite(trust):
            return
        self.count += 1
        self.disagreement_sum += risk
        self.disagreement_max = max(self.disagreement_max, risk)
        self.kappa_sum += trust
        self.kappa_min = min(self.kappa_min, trust)
        self.kappa_max = max(self.kappa_max, trust)

    def disagreement_mean(self) -> float:
        return self.disagreement_sum / self.count if self.count else 0.0

    def kappa_mean(self) -> float:
        return self.kappa_sum / self.count if self.count else 0.0

    def resolved_kappa_min(self) -> float:
        return self.kappa_min if self.count else 0.0

    def resolved_kappa_max(self) -> float:
        return self.kappa_max if self.count else 0.0


@dataclass(slots=True)
class _ReplayShadowStream:
    count: int = 0
    baseline_ratio_sum: float = 0.0
    trust_ratio_sum: float = 0.0
    advantage_sum: float = 0.0

    def record(self, baseline_ratio: float, trust_ratio: float) -> None:
        baseline = float(baseline_ratio)
        trusted = float(trust_ratio)
        if not math.isfinite(baseline) or not math.isfinite(trusted):
            return
        self.count += 1
        self.baseline_ratio_sum += baseline
        self.trust_ratio_sum += trusted
        self.advantage_sum += (baseline - trusted) / max(baseline, 1e-12)

    def mean(self, value: float) -> float:
        return value / self.count if self.count else 0.0


@dataclass(slots=True)
class _TrustAggregate:
    failures: int = 0
    applications: int = 0
    compute_seconds: float = 0.0
    scalar_transfer_seconds: float = 0.0
    weight_apply_seconds: float = 0.0
    replay_shadow_failures: int = 0
    audio: _AppliedStream = field(default_factory=_AppliedStream)
    video: _AppliedStream = field(default_factory=_AppliedStream)
    replay_shadow_audio: _ReplayShadowStream = field(default_factory=_ReplayShadowStream)
    replay_shadow_video: _ReplayShadowStream = field(default_factory=_ReplayShadowStream)


@dataclass(frozen=True, slots=True)
class _StreamTrustDecision:
    disagreement: float
    kappa: float


@dataclass(frozen=True, slots=True)
class _ForecastTrustDecision:
    step_id: int
    horizon: float
    latest_anchor_id: int
    audio: _StreamTrustDecision
    video: _StreamTrustDecision
    compute_seconds: float
    scalar_transfer_seconds: float


@dataclass(frozen=True, slots=True)
class _ReplayShadowRecord:
    step_id: int
    coordinate: float
    latest_anchor_id: int
    stream_name: str
    degree: int
    ridge_lambda: float
    blend_weight: float
    correction_gain: float
    disagreement: float
    # Shadow-only anchor-step coefficient; it is not the applied forecast-step kappa.
    # model_aware_trust_replay_shadow_* must not be read as applied-path evidence.
    kappa: float


@dataclass(slots=True)
class _ProbeState:
    run_id: int | None = None
    failures: int = 0
    audio: _StreamProbe = field(default_factory=_StreamProbe)
    video: _StreamProbe = field(default_factory=_StreamProbe)
    trust: _TrustAggregate = field(default_factory=_TrustAggregate)
    trust_by_step: dict[int, _ForecastTrustDecision | None] = field(default_factory=dict)
    applied_keys: set[tuple[int, str]] = field(default_factory=set)


def _active_run_id(runtime: SpectrumH3Runtime) -> int | None:
    run = getattr(runtime, "_run", None)
    return None if run is None else int(run.run_id)


def _state(runtime: SpectrumH3Runtime) -> _ProbeState:
    run_id = _active_run_id(runtime)
    state = getattr(runtime, "_forecast_trust_probe_state", None)
    if not isinstance(state, _ProbeState) or state.run_id != run_id:
        state = _ProbeState(run_id=run_id)
        runtime._forecast_trust_probe_state = state
    return state


def _trust_requested(runtime: SpectrumH3Runtime) -> bool:
    return bool(
        getattr(runtime.config, "model_aware_trust_shrinkage", False)
        and runtime.config.model_aware_mode == "full"
    )


def _trust_live(runtime: SpectrumH3Runtime) -> bool:
    return bool(_trust_requested(runtime) and runtime._model_aware_enabled())


def _archive_trust_aggregate(runtime: SpectrumH3Runtime) -> _TrustAggregate | None:
    archive = getattr(runtime, "_offline_archive", None)
    if archive is None:
        return None
    aggregate = getattr(archive, "_model_aware_trust_aggregate", None)
    if not isinstance(aggregate, _TrustAggregate):
        aggregate = _TrustAggregate()
        archive._model_aware_trust_aggregate = aggregate
    return aggregate


def _active_trust_aggregate(runtime: SpectrumH3Runtime) -> _TrustAggregate:
    if getattr(runtime, "_offline_phase", None) in {"first_pass", "replay"}:
        archive = getattr(runtime, "_offline_archive", None)
        aggregate = (
            getattr(archive, "_model_aware_trust_aggregate", None)
            if archive is not None
            else None
        )
        if isinstance(aggregate, _TrustAggregate):
            return aggregate
    return _state(runtime).trust


def _combine_samples(
    weights: torch.Tensor,
    samples: list[torch.Tensor],
) -> torch.Tensor:
    if int(weights.numel()) != len(samples):
        raise ValueError("trust probe weights are not aligned with evidence history")
    output = torch.zeros_like(samples[0])
    for weight, sample in zip(weights.tolist(), samples, strict=True):
        if weight != 0.0:
            output.add_(sample, alpha=float(weight))
    return output


def _logical_horizon(step: Any, forecaster: Any) -> float:
    if not forecaster._history:
        return 1.0
    anchor_id = forecaster._history[-1].anchor_id
    if anchor_id is None:
        return 1.0
    return float(max(1, int(step.step_id) - int(anchor_id)))


def _compute_forecast_trust(
    runtime: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
) -> _ForecastTrustDecision | None:
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or not forecaster._evidence_history:
        return None
    if len(forecaster._evidence_history) != forecaster.history_length:
        return None
    if not forecaster._history or forecaster._history[-1].anchor_id is None:
        return None
    ranges = runtime._stream_ranges(call)
    if len(ranges) != 2 or {name for name, _, _ in ranges} != {"audio", "video"}:
        # Packed topology does not prove a modality boundary. Stay baseline-identical.
        return None

    started = time.perf_counter()
    spectral_weights = forecaster._spectral_weights_configured(
        coordinate,
        degree=decision.degree,
        ridge_lambda=decision.ridge_lambda,
    )
    linear_weights = forecaster._linear_weights(coordinate)
    disagreement_tensors: dict[str, torch.Tensor] = {}
    for name, _start, _end in ranges:
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        if len(history_samples) != forecaster.history_length:
            return None
        spectral = _combine_samples(spectral_weights, history_samples)
        linear = _combine_samples(linear_weights, history_samples)
        spectral_rms = _tensor_rms(spectral)
        disagreement_tensors[name] = _tensor_rms(spectral - linear) / spectral_rms.clamp_min(
            _EPS
        )
    compute_seconds = time.perf_counter() - started

    transfer_started = time.perf_counter()
    ordered = torch.stack(
        (disagreement_tensors["audio"], disagreement_tensors["video"])
    )
    audio_disagreement, video_disagreement = ordered.detach().to(
        device="cpu", dtype=torch.float32
    ).tolist()
    scalar_transfer_seconds = time.perf_counter() - transfer_started

    step = getattr(runtime, "_step", None)
    horizon = (
        _logical_horizon(step, forecaster)
        if step is not None
        else max(1.0, float(decision.forecast_horizon))
    )
    audio_kappa = trust_kappa(
        audio_disagreement,
        horizon,
        theta=_APPLIED_THETA,
    )
    video_kappa = trust_kappa(
        video_disagreement,
        horizon,
        theta=_APPLIED_THETA,
    )
    return _ForecastTrustDecision(
        step_id=int(step.step_id) if step is not None else -1,
        horizon=float(horizon),
        latest_anchor_id=int(forecaster._history[-1].anchor_id),
        audio=_StreamTrustDecision(float(audio_disagreement), audio_kappa),
        video=_StreamTrustDecision(float(video_disagreement), video_kappa),
        compute_seconds=compute_seconds,
        scalar_transfer_seconds=scalar_transfer_seconds,
    )


def _record_applied_decision(
    aggregate: _TrustAggregate,
    decision: _ForecastTrustDecision,
) -> None:
    aggregate.compute_seconds += max(0.0, decision.compute_seconds)
    aggregate.scalar_transfer_seconds += max(0.0, decision.scalar_transfer_seconds)
    aggregate.audio.record(decision.audio.disagreement, decision.audio.kappa)
    aggregate.video.record(decision.video.disagreement, decision.video.kappa)


def _ensure_step_trust(
    runtime: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
) -> _ForecastTrustDecision | None:
    state = _state(runtime)
    step = getattr(runtime, "_step", None)
    if step is None:
        return None
    step_id = int(step.step_id)
    if step_id in state.trust_by_step:
        return state.trust_by_step[step_id]

    target_aggregate = (
        _archive_trust_aggregate(runtime)
        if getattr(runtime, "_offline_phase", None) == "first_pass"
        else state.trust
    )
    if target_aggregate is None:
        target_aggregate = state.trust
    try:
        resolved = _compute_forecast_trust(
            runtime,
            call,
            decision,
            coordinate=coordinate,
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError, KeyError, IndexError):
        target_aggregate.failures += 1
        state.trust_by_step[step_id] = None
        return None

    state.trust_by_step[step_id] = resolved
    if resolved is None:
        return None
    _record_applied_decision(target_aggregate, resolved)
    if getattr(runtime, "_offline_phase", None) == "first_pass":
        archive = getattr(runtime, "_offline_archive", None)
        if archive is not None:
            decisions = getattr(archive, "_model_aware_trust_forecasts", None)
            if not isinstance(decisions, dict):
                decisions = {}
                archive._model_aware_trust_forecasts = decisions
            decisions[step_id] = resolved
    return resolved


def _prediction_segments_with_trust(
    self: SpectrumH3Runtime,
    call: Any,
):
    if _ORIGINAL_RUNTIME_PREDICTION_SEGMENTS is None:
        raise RuntimeError("forecast trust controller was not installed correctly")
    if (
        _trust_live(self)
        and getattr(self, "_offline_phase", None) == "first_pass"
        and self.active_model_aware_decision is not None
    ):
        _ensure_step_trust(
            self,
            call,
            self.active_model_aware_decision,
            coordinate=float(self._step.coordinate),
        )
    return _ORIGINAL_RUNTIME_PREDICTION_SEGMENTS(self, call)


def _model_aware_weight_segments_with_trust(
    self: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
):
    if _ORIGINAL_RUNTIME_MODEL_AWARE_WEIGHT_SEGMENTS is None:
        raise RuntimeError("forecast trust controller was not installed correctly")
    weighted = _ORIGINAL_RUNTIME_MODEL_AWARE_WEIGHT_SEGMENTS(
        self,
        call,
        decision,
        coordinate=coordinate,
    )
    if not _trust_live(self) or getattr(self, "_offline_phase", None) is not None:
        return weighted

    trust = _ensure_step_trust(self, call, decision, coordinate=coordinate)
    if trust is None:
        return weighted
    state = _state(self)
    positions_by_id = {
        entry.anchor_id: index
        for index, entry in enumerate(self.forecaster._history)
        if entry.anchor_id is not None
    }
    latest_index = positions_by_id.get(trust.latest_anchor_id)
    if latest_index is None:
        state.trust.failures += 1
        return weighted

    by_range = {
        (start, end): name for name, start, end in self._stream_ranges(call)
    }
    updated = []
    apply_started = time.perf_counter()
    try:
        for start, end, weights in weighted:
            name = by_range.get((start, end))
            stream = getattr(trust, name, None) if name in {"audio", "video"} else None
            if not isinstance(stream, _StreamTrustDecision):
                updated.append((start, end, weights))
                continue
            try:
                trusted_weights = apply_trust_to_history_weights(
                    weights,
                    stream.kappa,
                    latest_index,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (RuntimeError, TypeError, ValueError, IndexError):
                state.trust.failures += 1
                updated.append((start, end, weights))
                continue
            updated.append((start, end, trusted_weights))
            key = (trust.step_id, name)
            if key not in state.applied_keys:
                state.applied_keys.add(key)
                state.trust.applications += 1
    finally:
        state.trust.weight_apply_seconds += time.perf_counter() - apply_started
    return tuple(updated)


def _record_replay_shadow_metadata(
    runtime: SpectrumH3Runtime,
    step: Any,
    decision: ModelAwareForecastDecision,
    *,
    stream_name: str,
    horizon: float,
    disagreement: float,
) -> None:
    if (
        not _trust_live(runtime)
        or getattr(runtime, "_offline_phase", None) != "first_pass"
        or stream_name not in {"audio", "video"}
        or not runtime.forecaster._history
    ):
        return
    latest_anchor_id = runtime.forecaster._history[-1].anchor_id
    archive = getattr(runtime, "_offline_archive", None)
    if latest_anchor_id is None or archive is None:
        return
    blend = (
        decision.audio_blend_weight
        if stream_name == "audio"
        else decision.video_blend_weight
    )
    gain = (
        decision.audio_correction_gain
        if stream_name == "audio"
        else decision.video_correction_gain
    )
    records = getattr(archive, "_model_aware_trust_replay_shadow_records", None)
    if not isinstance(records, list):
        records = []
        archive._model_aware_trust_replay_shadow_records = records
    records.append(
        _ReplayShadowRecord(
            step_id=int(step.step_id),
            coordinate=float(step.coordinate),
            latest_anchor_id=int(latest_anchor_id),
            stream_name=stream_name,
            degree=int(decision.degree),
            ridge_lambda=float(decision.ridge_lambda),
            blend_weight=float(blend),
            correction_gain=float(gain),
            disagreement=float(disagreement),
            kappa=trust_kappa(
                disagreement,
                horizon,
                theta=_APPLIED_THETA,
            ),
        )
    )


def _record_shadow_probe(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    evidence: AnchorEvidence,
) -> None:
    if runtime.config.model_aware_mode != "full":
        return
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or not forecaster._evidence_history:
        return
    if len(forecaster._evidence_history) != forecaster.history_length:
        return

    state = _state(runtime)
    horizon = _logical_horizon(step, forecaster)
    horizon_decay = math.exp(-_RACER_A * max(horizon - 1.0, 0.0))
    for name, start, end in runtime._stream_ranges(step.calls[0]):
        if name == "packed":
            continue
        stream_probe = getattr(state, name, None)
        stream_evidence = getattr(evidence, name, None)
        if not isinstance(stream_probe, _StreamProbe) or stream_evidence is None:
            continue
        blend = (
            decision.audio_blend_weight
            if name == "audio"
            else decision.video_blend_weight
        )
        gain = (
            decision.audio_correction_gain
            if name == "audio"
            else decision.video_correction_gain
        )
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        if not history_samples:
            continue
        actual = forecaster._sample_segment_device(combined, start, end)
        spectral_weights = forecaster._spectral_weights_configured(
            step.coordinate,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
        )
        linear_weights = forecaster._linear_weights(step.coordinate)
        spectral = _combine_samples(spectral_weights, history_samples)
        linear = _combine_samples(linear_weights, history_samples)
        blended_weights = (
            spectral_weights
            if blend >= 1.0 - 1e-12
            else linear_weights
            if blend <= 1e-12
            else blend * spectral_weights + (1.0 - blend) * linear_weights
        )
        predicted = _combine_samples(blended_weights, history_samples)
        latest = history_samples[-1]
        previous = history_samples[-2]
        delta = latest - previous
        proposal = predicted + float(gain) * delta

        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        disagreement = _tensor_rms(spectral - linear) / _tensor_rms(spectral).clamp_min(
            epsilon
        )
        corrected_ratio = _tensor_rms(actual - proposal) / hold_rms

        projection = float(stream_evidence.residual_projection)
        delta_bound_gain = projection / (
            1.0 + abs(projection) / _CORRECTION_GAIN_LIMIT
        )
        delta_bound = predicted + float(delta_bound_gain) * delta
        delta_bound_ratio = _tensor_rms(actual - delta_bound) / hold_rms

        oracle_kappa = oracle_segment_kappa(actual, latest, proposal)
        oracle = latest + oracle_kappa.to(dtype=proposal.dtype) * (proposal - latest)
        oracle_ratio = _tensor_rms(actual - oracle) / hold_rms

        candidate_ratios: dict[float, torch.Tensor] = {}
        for theta in _TRUST_THETAS:
            kappa = float(horizon_decay) * torch.sigmoid(
                disagreement.new_tensor(_RACER_B * float(theta))
                - _RACER_B * disagreement
            )
            candidate = latest + kappa.to(dtype=proposal.dtype) * (proposal - latest)
            candidate_ratios[theta] = _tensor_rms(actual - candidate) / hold_rms

        values = torch.stack(
            (
                disagreement,
                corrected_ratio,
                delta_bound_ratio,
                oracle_ratio,
                oracle_kappa,
                *[candidate_ratios[theta] for theta in _TRUST_THETAS],
            )
        )
        resolved = values.detach().to(device="cpu", dtype=torch.float32).tolist()
        stream_probe.record(
            horizon=horizon,
            disagreement=float(resolved[0]),
            corrected_ratio=float(resolved[1]),
            delta_bound_gain=delta_bound_gain,
            delta_bound_ratio=float(resolved[2]),
            oracle_ratio=float(resolved[3]),
            oracle_kappa=float(resolved[4]),
            candidate_ratios={
                theta: float(resolved[5 + index])
                for index, theta in enumerate(_TRUST_THETAS)
            },
        )
        _record_replay_shadow_metadata(
            runtime,
            step,
            decision,
            stream_name=name,
            horizon=horizon,
            disagreement=float(resolved[0]),
        )


def _generic_anchor_evidence_with_trust_probe(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    if _ORIGINAL_GENERIC_ANCHOR_EVIDENCE is None:
        raise RuntimeError("forecast trust probe was not installed correctly")
    evidence = _ORIGINAL_GENERIC_ANCHOR_EVIDENCE(runtime, step, combined, decision)
    if evidence is None:
        return None
    try:
        _record_shadow_probe(runtime, step, combined, decision, evidence)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
        _state(runtime).failures += 1
    return evidence


def _sample_archive_stream(
    smoother: OfflineSmoother,
    start_row: int,
    end_row: int,
) -> torch.Tensor:
    branch_count = int(smoother.archive.anchors[0].feature.shape[0])
    per_branch = max(1, _REPLAY_SHADOW_SAMPLE_ELEMENTS // max(1, branch_count))
    sampled_anchors = []
    for anchor in smoother.archive.anchors:
        branch_samples = []
        for branch in range(branch_count):
            flat = anchor.feature[branch, start_row:end_row].detach().reshape(-1)
            if flat.numel() == 0:
                raise ValueError("replay trust shadow cannot sample an empty stream")
            stride = max(1, int(flat.numel()) // per_branch)
            branch_samples.append(
                flat[::stride][:per_branch].to(device="cpu", dtype=torch.float32)
            )
        sampled_anchors.append(torch.stack(branch_samples, dim=0))
    return torch.stack(sampled_anchors, dim=0)


def _validate_replay_transfer(
    smoother: OfflineSmoother,
    aggregate: _TrustAggregate,
) -> None:
    records = getattr(
        smoother.archive,
        "_model_aware_trust_replay_shadow_records",
        None,
    )
    if not isinstance(records, list) or not records:
        return
    try:
        ranges = {name: (start, end) for name, start, end in smoother._stream_ranges}
        samples_by_stream = {
            name: _sample_archive_stream(smoother, start, end)
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
        if not isinstance(record, _ReplayShadowRecord):
            continue
        samples = samples_by_stream.get(record.stream_name)
        if samples is None:
            continue
        try:
            target_index = anchor_ids.index(record.step_id)
            latest_index = anchor_ids.index(record.latest_anchor_id)
            if (
                target_index <= 0
                or target_index >= len(anchor_ids) - 1
                or latest_index >= target_index
            ):
                continue
            retained = [
                index for index in range(len(anchor_ids)) if index != target_index
            ]
            if len(retained) < max(2, record.degree + 1):
                continue
            validator = HistoryWeightForecaster(
                degree=record.degree,
                ridge_lambda=record.ridge_lambda,
                max_history=max(len(retained), record.degree + 1, 2),
                history_storage="system_ram",
            )
            for index in retained:
                validator.update(
                    smoother.archive.anchors[index].coordinate,
                    samples[index],
                    anchor_id=anchor_ids[index],
                    take_ownership=True,
                )
            spectral = smoother._affine_spectral_weights(
                validator.model_aware_weights(
                    record.coordinate,
                    1.0,
                    degree=record.degree,
                    ridge_lambda=record.ridge_lambda,
                    correction_gain=0.0,
                )
            )
            left_id = anchor_ids[target_index - 1]
            right_id = anchor_ids[target_index + 1]
            retained_ids = [anchor_ids[index] for index in retained]
            left_position = retained_ids.index(left_id)
            right_position = retained_ids.index(right_id)
            latest_position = retained_ids.index(record.latest_anchor_id)
            left = smoother.archive.anchors[target_index - 1]
            right = smoother.archive.anchors[target_index + 1]
            spacing = right.coordinate - left.coordinate
            if abs(spacing) <= 1e-12:
                raise RuntimeError("replay trust shadow bracket has duplicate coordinates")
            ratio = (record.coordinate - left.coordinate) / spacing
            local = torch.zeros(len(retained), dtype=torch.float32)
            local[left_position] = 1.0 - ratio
            local[right_position] = ratio
            weights = (
                record.blend_weight * spectral
                + (1.0 - record.blend_weight) * local
            )
            if record.correction_gain != 0.0:
                weights = weights.clone()
                weights[left_position] -= record.correction_gain
                weights[right_position] += record.correction_gain
            trusted_weights = apply_trust_to_history_weights(
                weights,
                record.kappa,
                latest_position,
            )
            retained_samples = samples[retained]
            baseline = torch.einsum("k,kbs->bs", weights, retained_samples)
            trusted = torch.einsum("k,kbs->bs", trusted_weights, retained_samples)
            actual = samples[target_index]
            hold = samples[latest_index]
            epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_EPS)
            hold_rms = _tensor_rms(actual - hold).clamp_min(epsilon)
            baseline_ratio = _tensor_rms(actual - baseline) / hold_rms
            trust_ratio = _tensor_rms(actual - trusted) / hold_rms
            target = (
                aggregate.replay_shadow_audio
                if record.stream_name == "audio"
                else aggregate.replay_shadow_video
            )
            target.record(float(baseline_ratio), float(trust_ratio))
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
            aggregate.replay_shadow_failures += 1


def _offline_build_forecast_weights_with_trust(
    self: OfflineSmoother,
):
    if _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS is None:
        raise RuntimeError("forecast trust controller was not installed correctly")
    weights_by_step = _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS(self)
    decisions = getattr(self.archive, "_model_aware_trust_forecasts", None)
    aggregate = getattr(self.archive, "_model_aware_trust_aggregate", None)
    if not isinstance(decisions, dict) or not isinstance(aggregate, _TrustAggregate):
        return weights_by_step

    apply_started = time.perf_counter()
    try:
        for step_id, trust in decisions.items():
            if not isinstance(trust, _ForecastTrustDecision):
                continue
            try:
                latest_index = self._anchor_ids.index(trust.latest_anchor_id)
            except ValueError:
                aggregate.failures += 1
                continue
            for stream_index, (name, _start, _end) in enumerate(self._stream_ranges):
                stream = getattr(trust, name, None) if name in {"audio", "video"} else None
                if not isinstance(stream, _StreamTrustDecision):
                    continue
                staged: dict[tuple[int, int, int], torch.Tensor] = {}
                try:
                    for branch in range(self._branch_count):
                        key = (int(step_id), branch, stream_index)
                        baseline = weights_by_step[key]
                        staged[key] = apply_trust_to_history_weights(
                            baseline,
                            stream.kappa,
                            latest_index,
                        )
                except torch.cuda.OutOfMemoryError:
                    raise
                except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
                    aggregate.failures += 1
                    continue
                weights_by_step.update(staged)
                aggregate.applications += 1
    finally:
        aggregate.weight_apply_seconds += time.perf_counter() - apply_started

    _validate_replay_transfer(self, aggregate)
    self._model_aware_trust_aggregate = aggregate
    return weights_by_step


def _stream_summary(name: str, probe: _StreamProbe) -> str:
    fields = [
        f"trust_probe_{name}_samples={probe.count}",
        f"trust_probe_{name}_horizon_mean={probe.mean(probe.horizon_sum):.6f}",
        f"trust_probe_{name}_horizon_max={probe.horizon_max:.6f}",
        f"trust_probe_{name}_disagreement_mean={probe.mean(probe.disagreement_sum):.6f}",
        f"trust_probe_{name}_disagreement_max={probe.disagreement_max:.6f}",
        f"trust_probe_{name}_corrected_ratio_mean={probe.mean(probe.corrected_ratio_sum):.6f}",
        f"trust_probe_{name}_delta_bound_gain_mean={probe.mean(probe.delta_bound_gain_sum):.6f}",
        f"trust_probe_{name}_delta_bound_ratio_mean={probe.mean(probe.delta_bound_ratio_sum):.6f}",
        f"trust_probe_{name}_delta_bound_advantage_mean={probe.mean(probe.delta_bound_advantage_sum):.6f}",
        f"trust_probe_{name}_oracle_ratio_mean={probe.mean(probe.oracle_ratio_sum):.6f}",
        f"trust_probe_{name}_oracle_kappa_mean={probe.mean(probe.oracle_kappa_sum):.6f}",
        f"trust_probe_{name}_oracle_advantage_mean={probe.mean(probe.oracle_advantage_sum):.6f}",
        f"trust_probe_{name}_error_corr={probe.error_correlation.correlation():.6f}",
        f"trust_probe_{name}_shrink_corr={probe.shrink_correlation.correlation():.6f}",
    ]
    for theta in _TRUST_THETAS:
        suffix = f"{theta:.2f}".replace(".", "p")
        fields.extend(
            (
                f"trust_probe_{name}_theta_{suffix}_ratio_mean={probe.mean(probe.candidate_ratio_sums[theta]):.6f}",
                f"trust_probe_{name}_theta_{suffix}_advantage_mean={probe.mean(probe.candidate_advantage_sums[theta]):.6f}",
            )
        )
    return " ".join(fields)


def _applied_stream_summary(name: str, stream: _AppliedStream) -> str:
    return " ".join(
        (
            f"model_aware_trust_{name}_disagreement_mean={stream.disagreement_mean():.6f}",
            f"model_aware_trust_{name}_disagreement_max={stream.disagreement_max:.6f}",
            f"model_aware_trust_{name}_kappa_mean={stream.kappa_mean():.6f}",
            f"model_aware_trust_{name}_kappa_min={stream.resolved_kappa_min():.6f}",
            f"model_aware_trust_{name}_kappa_max={stream.resolved_kappa_max():.6f}",
        )
    )


def _replay_shadow_summary(name: str, stream: _ReplayShadowStream) -> str:
    return " ".join(
        (
            f"model_aware_trust_replay_shadow_{name}_samples={stream.count}",
            f"model_aware_trust_replay_shadow_{name}_baseline_ratio_mean={stream.mean(stream.baseline_ratio_sum):.6f}",
            f"model_aware_trust_replay_shadow_{name}_ratio_mean={stream.mean(stream.trust_ratio_sum):.6f}",
            f"model_aware_trust_replay_shadow_{name}_advantage_mean={stream.mean(stream.advantage_sum):.6f}",
        )
    )


def _debug_summary(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("forecast trust probe was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    state = _state(self)
    aggregate = _active_trust_aggregate(self)
    requested = _trust_requested(self)
    if not requested:
        path = "disabled"
    elif getattr(self, "_offline_phase", None) in {"first_pass", "replay"}:
        path = "offline_replay_causal_kappa_transfer"
    else:
        path = "causal_single_pass"
    total_seconds = (
        aggregate.compute_seconds
        + aggregate.scalar_transfer_seconds
        + aggregate.weight_apply_seconds
    )
    replay_shadow_mode = (
        "loo_unattenuated_future_bracket"
        if requested and aggregate.replay_shadow_audio.count + aggregate.replay_shadow_video.count
        else "inactive"
    )
    return (
        f"{summary} "
        "trust_probe=shadow_only "
        "trust_probe_observer=unblended_spectral_vs_linear "
        "trust_probe_applied=0 "
        f"trust_probe_failures={state.failures} "
        "trust_probe_extra_transformer_nfe=0 "
        f"{_stream_summary('audio', state.audio)} "
        f"{_stream_summary('video', state.video)} "
        f"model_aware_trust_enabled={int(requested)} "
        f"model_aware_trust_applied={int(aggregate.applications > 0)} "
        f"model_aware_trust_path={path} "
        f"model_aware_trust_applications={aggregate.applications} "
        f"{_applied_stream_summary('audio', aggregate.audio)} "
        f"{_applied_stream_summary('video', aggregate.video)} "
        f"model_aware_trust_failures={aggregate.failures} "
        f"model_aware_trust_compute_s={aggregate.compute_seconds:.6f} "
        f"model_aware_trust_scalar_transfer_s={aggregate.scalar_transfer_seconds:.6f} "
        f"model_aware_trust_weight_apply_s={aggregate.weight_apply_seconds:.6f} "
        f"model_aware_trust_total_s={total_seconds:.6f} "
        "model_aware_trust_extra_transformer_nfe=0 "
        f"model_aware_trust_replay_shadow={replay_shadow_mode} "
        f"model_aware_trust_replay_shadow_failures={aggregate.replay_shadow_failures} "
        f"{_replay_shadow_summary('audio', aggregate.replay_shadow_audio)} "
        f"{_replay_shadow_summary('video', aggregate.replay_shadow_video)}"
    )


def install_forecast_trust_probe() -> None:
    """Install shadow telemetry plus the opt-in scalar trust controller once."""
    global _ORIGINAL_GENERIC_ANCHOR_EVIDENCE
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    global _ORIGINAL_RUNTIME_PREDICTION_SEGMENTS
    global _ORIGINAL_RUNTIME_MODEL_AWARE_WEIGHT_SEGMENTS
    global _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS
    if getattr(SpectrumH3Runtime, "_forecast_trust_probe_installed", False):
        return

    _ORIGINAL_GENERIC_ANCHOR_EVIDENCE = _generic._generic_anchor_evidence
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    _ORIGINAL_RUNTIME_PREDICTION_SEGMENTS = SpectrumH3Runtime._prediction_segments
    _ORIGINAL_RUNTIME_MODEL_AWARE_WEIGHT_SEGMENTS = (
        SpectrumH3Runtime._model_aware_weight_segments
    )
    _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS = OfflineSmoother._build_forecast_weights

    _generic._generic_anchor_evidence = _generic_anchor_evidence_with_trust_probe
    SpectrumH3Runtime._prediction_segments = _prediction_segments_with_trust
    SpectrumH3Runtime._model_aware_weight_segments = (
        _model_aware_weight_segments_with_trust
    )
    OfflineSmoother._build_forecast_weights = _offline_build_forecast_weights_with_trust
    SpectrumH3Runtime.debug_summary = _debug_summary
    SpectrumH3Runtime._forecast_trust_probe_installed = True


__all__ = [
    "apply_trust_to_history_weights",
    "install_forecast_trust_probe",
    "oracle_segment_kappa",
    "trust_kappa",
]
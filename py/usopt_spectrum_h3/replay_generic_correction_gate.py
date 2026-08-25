from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from .experiments import OfflineModelAwareDecision, OfflineSmoother, OfflineStepRecord
from .runtime import SpectrumH3Runtime

_ARCHIVE_GATE_ATTR = "_model_aware_replay_generic_correction_enabled"
_ARCHIVE_TELEMETRY_ATTR = "_model_aware_replay_generic_correction_telemetry"

_ORIGINAL_BEGIN_OFFLINE_CAPTURE: Callable[..., Any] | None = None
_ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(frozen=True, slots=True)
class _ReplayGenericCorrectionTelemetry:
    enabled: bool
    path: str
    applications: int
    skips: int
    extra_transformer_nfe: int = 0


def _generic_coefficients_for_stream(
    decision: OfflineModelAwareDecision,
    stream_name: str,
) -> tuple[float, ...]:
    if stream_name == "audio":
        return tuple(decision.audio_correction_coefficients)
    if stream_name == "video":
        return tuple(decision.video_correction_coefficients)
    audio = tuple(decision.audio_correction_coefficients)
    video = tuple(decision.video_correction_coefficients)
    if len(audio) != len(video):
        return ()
    return tuple(
        0.5 * (audio_value + video_value)
        for audio_value, video_value in zip(audio, video, strict=True)
    )


def _generic_application_opportunities(
    smoother: OfflineSmoother,
    steps: list[OfflineStepRecord],
) -> int:
    count = 0
    for record in steps:
        if record.actual or record.model_aware_decision is None:
            continue
        decision = record.model_aware_decision
        for stream_name, _, _ in smoother._stream_ranges:
            coefficients = _generic_coefficients_for_stream(decision, stream_name)
            if len(coefficients) == 1 and coefficients[0] != 0.0:
                count += smoother._branch_count
    return count


def _without_generic_scalar(
    decision: OfflineModelAwareDecision,
) -> OfflineModelAwareDecision:
    audio_coefficients = tuple(decision.audio_correction_coefficients)
    video_coefficients = tuple(decision.video_correction_coefficients)
    strip_audio = len(audio_coefficients) == 1
    strip_video = len(video_coefficients) == 1
    if not strip_audio and not strip_video:
        return decision

    remaining_audio = () if strip_audio else audio_coefficients
    remaining_video = () if strip_video else video_coefficients
    return replace(
        decision,
        audio_correction_gain=(0.0 if strip_audio else decision.audio_correction_gain),
        video_correction_gain=(0.0 if strip_video else decision.video_correction_gain),
        audio_correction_coefficients=remaining_audio,
        video_correction_coefficients=remaining_video,
        correction_anchor_ids=(
            ()
            if not remaining_audio and not remaining_video
            else decision.correction_anchor_ids
        ),
    )


def _without_generic_replay_correction(
    steps: list[OfflineStepRecord],
) -> list[OfflineStepRecord]:
    stripped: list[OfflineStepRecord] = []
    for record in steps:
        decision = record.model_aware_decision
        if record.actual or decision is None:
            stripped.append(record)
            continue
        stripped.append(
            replace(
                record,
                model_aware_decision=_without_generic_scalar(decision),
            )
        )
    return stripped


def _begin_offline_capture_with_replay_generic_gate(
    self: SpectrumH3Runtime,
    *,
    total_steps: int,
    sampler_name: str,
) -> None:
    if _ORIGINAL_BEGIN_OFFLINE_CAPTURE is None:
        raise RuntimeError("replay generic-correction gate was not installed correctly")
    _ORIGINAL_BEGIN_OFFLINE_CAPTURE(
        self,
        total_steps=total_steps,
        sampler_name=sampler_name,
    )
    archive = getattr(self, "_offline_archive", None)
    if archive is None:
        return
    requested = bool(getattr(self.config, "model_aware_replay_generic_correction", False))
    effective = bool(self.config.model_aware_mode != "full" or requested)
    setattr(archive, _ARCHIVE_GATE_ATTR, effective)


def _build_forecast_weights_with_replay_generic_gate(
    self: OfflineSmoother,
):
    if _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS is None:
        raise RuntimeError("replay generic-correction gate was not installed correctly")

    # Every production archive is stamped by begin_offline_capture. Treat an
    # unstamped hand-constructed archive as legacy-enabled so low-level archive
    # consumers and historical tests retain their pre-gate semantics.
    enabled = bool(getattr(self.archive, _ARCHIVE_GATE_ATTR, True))
    original_steps = self.archive.steps
    opportunities = _generic_application_opportunities(self, original_steps)

    if enabled or opportunities == 0:
        weights_by_step = _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS(self)
    else:
        # Preserve the first-pass archive verbatim. Only the production replay
        # weight build sees a temporary view with the transplanted scalar removed.
        # Replay-native A/B/C/D diagnostics use their separately persisted shadow
        # records, so D remains available as a counterfactual while production is B.
        self.archive.steps = _without_generic_replay_correction(original_steps)
        try:
            weights_by_step = _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS(self)
        finally:
            self.archive.steps = original_steps

    observed_applications = int(
        getattr(self, "model_aware_offline_correction_applications", opportunities)
    )
    telemetry = _ReplayGenericCorrectionTelemetry(
        enabled=enabled,
        path=(
            "current_causal_gain_transfer"
            if enabled
            else "disabled_replay_geometry_experiment"
        ),
        applications=(observed_applications if enabled else 0),
        skips=(0 if enabled else opportunities),
    )
    setattr(self.archive, _ARCHIVE_TELEMETRY_ATTR, telemetry)
    return weights_by_step


def _debug_summary_with_replay_generic_gate(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("replay generic-correction gate was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    if archive is None:
        return summary
    telemetry = getattr(archive, _ARCHIVE_TELEMETRY_ATTR, None)
    if not isinstance(telemetry, _ReplayGenericCorrectionTelemetry):
        enabled = bool(getattr(archive, _ARCHIVE_GATE_ATTR, True))
        telemetry = _ReplayGenericCorrectionTelemetry(
            enabled=enabled,
            path=(
                "current_causal_gain_transfer"
                if enabled
                else "disabled_replay_geometry_experiment"
            ),
            applications=0,
            skips=0,
        )
    return (
        f"{summary} "
        f"model_aware_replay_generic_correction_enabled={int(telemetry.enabled)} "
        f"model_aware_replay_generic_correction_path={telemetry.path} "
        f"model_aware_replay_generic_correction_applications={telemetry.applications} "
        f"model_aware_replay_generic_correction_skips={telemetry.skips} "
        "model_aware_replay_generic_correction_extra_transformer_nfe=0"
    )


def install_replay_generic_correction_gate() -> None:
    """Install the explicit replay-only generic-correction experiment gate."""
    global _ORIGINAL_BEGIN_OFFLINE_CAPTURE
    global _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_generic_correction_gate_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_shadow_composition_installed", False):
        raise RuntimeError("install replay shadow composition before replay correction gate")

    _ORIGINAL_BEGIN_OFFLINE_CAPTURE = SpectrumH3Runtime.begin_offline_capture
    _ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS = OfflineSmoother._build_forecast_weights
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary

    SpectrumH3Runtime.begin_offline_capture = (
        _begin_offline_capture_with_replay_generic_gate
    )
    OfflineSmoother._build_forecast_weights = (
        _build_forecast_weights_with_replay_generic_gate
    )
    SpectrumH3Runtime.debug_summary = _debug_summary_with_replay_generic_gate
    SpectrumH3Runtime._replay_generic_correction_gate_installed = True


__all__ = ["install_replay_generic_correction_gate"]

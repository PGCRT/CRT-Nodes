from __future__ import annotations

from typing import Any, Callable

from . import replay_calibration as _calibration
from .runtime import SpectrumH3Runtime

_ORIGINAL_CALIBRATION_ROW: Callable[..., dict[str, Any] | None] | None = None


def _calibration_row_with_interior_guard(
    smoother,
    record,
    samples,
    anchor_ids: list[int],
    *,
    run_id: int | None,
):
    if _ORIGINAL_CALIBRATION_ROW is None:
        raise RuntimeError("replay calibration validation was not installed correctly")
    if record.stream_name == "video" and record.blend_weight > 1e-12:
        try:
            target_index = anchor_ids.index(record.step_id)
        except ValueError as exc:
            raise ValueError(
                f"replay calibration target step {record.step_id} is not an anchor"
            ) from exc
        if target_index <= 0 or target_index >= len(anchor_ids) - 1:
            raise ValueError(
                "replay calibration requires an interior withheld target with both "
                f"left and right anchors; step={record.step_id} index={target_index} "
                f"anchor_count={len(anchor_ids)}"
            )
    return _ORIGINAL_CALIBRATION_ROW(
        smoother,
        record,
        samples,
        anchor_ids,
        run_id=run_id,
    )


def install_replay_calibration_validation() -> None:
    """Install structural guards around the debug-only replay calibration row."""
    global _ORIGINAL_CALIBRATION_ROW
    if getattr(SpectrumH3Runtime, "_replay_calibration_validation_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_calibration_installed", False):
        raise RuntimeError("install replay calibration before calibration validation")
    _ORIGINAL_CALIBRATION_ROW = _calibration._calibration_row
    _calibration._calibration_row = _calibration_row_with_interior_guard
    SpectrumH3Runtime._replay_calibration_validation_installed = True


__all__ = ["install_replay_calibration_validation"]

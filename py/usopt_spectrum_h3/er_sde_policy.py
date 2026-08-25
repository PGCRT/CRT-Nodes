from __future__ import annotations

from typing import Any, Callable

from .runtime import SpectrumH3Runtime

_ORIGINAL_RUNTIME_BEGIN_STEP: Callable[..., dict[str, Any]] | None = None


def _begin_step(self: SpectrumH3Runtime, timestep: Any) -> dict[str, Any]:
    if _ORIGINAL_RUNTIME_BEGIN_STEP is None:
        raise RuntimeError("ER-SDE terminal replay policy was not installed correctly")

    decision = _ORIGINAL_RUNTIME_BEGIN_STEP(self, timestep)
    run = getattr(self, "_run", None)
    step = getattr(self, "_step", None)
    if (
        not bool(decision.get("actual", False))
        and getattr(self, "_offline_phase", None) == "first_pass"
        and run is not None
        and step is not None
        and run.sampler_name == "sample_er_sde"
        and run.total_steps >= 2
        and step.step_id == run.total_steps - 2
    ):
        # Offline replay can only interpolate a forecast safely when a later exact
        # anchor exists. The reproduced 25-step ER-SDE failure was specifically a
        # naturally forecast penultimate step bracketed by exact steps 22 and 24.
        # Promote only that vulnerable decision. Schedules whose penultimate step
        # is already actual (including the normal 20- and 32-step schedules) are
        # left completely untouched. Reuse the existing terminal-tail reason so
        # diagnostics and saved expectations do not invent a second tail category.
        reason = "final actual tail"
        step.mode = "actual"
        step.reason = reason
        step.adaptive_recompute = False
        step.bootstrap_forecast = False
        step.model_aware_decision = None
        step.model_aware_forced_actual = False
        decision = dict(decision)
        decision["actual"] = True
        decision["reason"] = reason
    return decision


def install_er_sde_tail_policy() -> None:
    """Protect only an ER-SDE offline-replay penultimate step that would forecast."""
    global _ORIGINAL_RUNTIME_BEGIN_STEP
    if getattr(SpectrumH3Runtime, "_er_sde_tail_policy_installed", False):
        return
    _ORIGINAL_RUNTIME_BEGIN_STEP = SpectrumH3Runtime.begin_step
    SpectrumH3Runtime.begin_step = _begin_step
    SpectrumH3Runtime._er_sde_tail_policy_installed = True


__all__ = ["install_er_sde_tail_policy"]

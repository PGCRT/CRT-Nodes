from __future__ import annotations

from typing import Any, Callable

from . import replay_component_shadow as _component
from . import replay_trust_shadow as _replay
from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .runtime import SpectrumH3Runtime

_NATIVE_REPLAY_VALIDATOR: Callable[..., Any] | None = None


def _validate_composed_replay_shadows(
    smoother: OfflineSmoother,
    aggregate: _trust._TrustAggregate,
) -> None:
    if _NATIVE_REPLAY_VALIDATOR is None:
        raise RuntimeError("replay shadow composition was not installed correctly")

    failures_before = aggregate.replay_shadow_failures
    _NATIVE_REPLAY_VALIDATOR(smoother, aggregate)
    if aggregate.replay_shadow_failures != failures_before:
        # The component decomposition depends on the same replay-native sampled
        # cache setup. Do not cascade one upstream diagnostic failure into a
        # second failure count or attempt a dependent decomposition with invalid
        # evidence. The production replay weights were already built unchanged.
        return
    _component._validate_replay_decomposition(smoother, aggregate)


def install_replay_shadow_composition() -> None:
    """Compose native replay and component shadows without cascading failures."""
    global _NATIVE_REPLAY_VALIDATOR
    if getattr(SpectrumH3Runtime, "_replay_shadow_composition_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_component_shadow_installed", False):
        raise RuntimeError("install replay component shadow before replay composition")
    native = _component._ORIGINAL_REPLAY_VALIDATOR
    if native is None:
        raise RuntimeError("native replay shadow validator is unavailable")

    _NATIVE_REPLAY_VALIDATOR = native
    _replay._validate_replay_native_shadow = _validate_composed_replay_shadows
    SpectrumH3Runtime._replay_shadow_composition_installed = True


__all__ = ["install_replay_shadow_composition"]

"""Pure inference invariants for the CRT LTX 2.3 distilled sampler."""

from __future__ import annotations

import math


DISTILLED_MAIN_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)

# Lightricks' official ComfyUI two-stage distilled workflow starts the
# refinement pass at 0.85 after latent upscaling.
DISTILLED_REFINEMENT_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)

# Self-refinement pass on an already full-resolution latent (no upsampling):
# matches the official inpaint/outpaint high-res schedule.
DISTILLED_POLISH_SIGMAS = (0.7250, 0.4219, 0.0)
MAX_LTX_FRAME_COUNT = 4089  # Largest 8k+1 value within the UI's 4096 limit.


def _sigma_text(values: tuple[float, ...]) -> str:
    return ", ".join(str(float(value)) for value in values)


DISTILLED_MAIN_SIGMAS_TEXT = _sigma_text(DISTILLED_MAIN_SIGMAS)
DISTILLED_REFINEMENT_SIGMAS_TEXT = _sigma_text(DISTILLED_REFINEMENT_SIGMAS)
DISTILLED_POLISH_SIGMAS_TEXT = _sigma_text(DISTILLED_POLISH_SIGMAS)


def normalize_frame_count(frame_count: int, strategy: str = "nearest") -> int:
    """Return a valid LTX pixel-frame count satisfying ``frames = 8k + 1``.

    ``nearest`` is intended for user/audio-derived targets. ``floor`` is used
    for source video batches so normalization never invents input frames.
    """

    frames = max(1, int(frame_count))
    if frames == 1:
        return 1

    if strategy == "floor":
        groups = (frames - 1) // 8
    elif strategy == "nearest":
        groups = math.floor(((frames - 1) / 8.0) + 0.5)
    else:
        raise ValueError(f"Unknown LTX frame normalization strategy: {strategy}")

    return min(MAX_LTX_FRAME_COUNT, max(1, groups * 8 + 1))


def normalize_dimension(value: int | float, multiple: int) -> int:
    """Round a pixel dimension to the nearest positive model-safe multiple."""

    divisor = max(1, int(multiple))
    pixels = max(divisor, int(round(float(value))))
    return max(divisor, int(math.floor((pixels / divisor) + 0.5)) * divisor)

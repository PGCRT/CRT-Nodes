from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HELPERS = Path(__file__).resolve().parents[1] / "py" / "_ltx23_inference.py"
SPEC = importlib.util.spec_from_file_location("crt_ltx23_inference", HELPERS)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LTX23InferenceInvariantTests(unittest.TestCase):
    def test_fixed_main_schedule(self):
        self.assertEqual(
            MODULE.DISTILLED_MAIN_SIGMAS,
            (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        )

    def test_fixed_refinement_schedule(self):
        self.assertEqual(MODULE.DISTILLED_REFINEMENT_SIGMAS, (0.85, 0.725, 0.4219, 0.0))

    def test_valid_frame_counts_are_preserved(self):
        for frames in (1, 9, 17, 121, 161, 4089):
            self.assertEqual(MODULE.normalize_frame_count(frames), frames)

    def test_nearest_frame_normalization(self):
        self.assertEqual(MODULE.normalize_frame_count(160), 161)
        self.assertEqual(MODULE.normalize_frame_count(158), 161)
        self.assertEqual(MODULE.normalize_frame_count(157), 161)
        self.assertEqual(MODULE.normalize_frame_count(156), 153)

    def test_frame_count_is_clamped_to_the_largest_valid_ui_value(self):
        self.assertEqual(MODULE.normalize_frame_count(4096), 4089)
        self.assertEqual(MODULE.normalize_frame_count(10000), 4089)

    def test_floor_normalization_never_invents_video_frames(self):
        for frames in range(1, 200):
            normalized = MODULE.normalize_frame_count(frames, strategy="floor")
            self.assertLessEqual(normalized, frames)
            self.assertEqual((normalized - 1) % 8, 0)

    def test_dimensions_snap_to_model_grid(self):
        self.assertEqual(MODULE.normalize_dimension(750, 32), 736)
        self.assertEqual(MODULE.normalize_dimension(750, 64), 768)
        self.assertEqual(MODULE.normalize_dimension(16, 32), 32)

    def test_invalid_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.normalize_frame_count(10, strategy="ceil")


if __name__ == "__main__":
    unittest.main()

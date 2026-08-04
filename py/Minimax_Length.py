class CRT_MinimaxLength:
    """
    Frame count for MiniMax-H3 video generation.

    MiniMax-H3 runs at a fixed 24 fps and its video VAE can only decode
    frame counts of the form 17 * n + 5: 5, 22, 39, ..., 362.

    The official diffusers pipeline additionally restricts duration to
    5-15 seconds (124-362 frames), but ComfyUI does not enforce that
    window, so the full decodable grid is exposed here. Values below
    124 frames are outside the model's trained duration range and may
    produce weaker results. The 362-frame ceiling (15 s) is kept.

    The widget steps by 17 from the minimum of 5, so only valid values
    are selectable. The function additionally snaps any out-of-grid input
    (e.g. from a converted widget) to the nearest valid frame count.
    """

    MIN_FRAMES = 5    # 17 * 0 + 5 -> 0.21 s (VAE grid floor; below 5 s is out-of-distribution)
    MAX_FRAMES = 362  # 17 * 21 + 5 -> 15.08 s at 24 fps (model maximum: 15 s)
    STEP = 17

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "INT",
                    {
                        "default": 124,
                        "min": cls.MIN_FRAMES,
                        "max": cls.MAX_FRAMES,
                        "step": cls.STEP,
                        "tooltip": (
                            "MiniMax-H3 frame count at 24 fps. Valid values are "
                            "17*n+5 (5 to 362). Official range is 124-362 (5-15 s); "
                            "shorter runs but is out-of-distribution."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "get_value"
    CATEGORY = "CRT/Utils/Logic & Values"
    DESCRIPTION = "Frame count snapped to MiniMax-H3's valid grid (17n+5, 24 fps, 5-15 s)."

    def get_value(self, frames):
        n = round((int(frames) - self.MIN_FRAMES) / self.STEP)
        n = max(0, min((self.MAX_FRAMES - self.MIN_FRAMES) // self.STEP, n))
        return (self.MIN_FRAMES + n * self.STEP,)


NODE_CLASS_MAPPINGS = {"CRT_MinimaxLength": CRT_MinimaxLength}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_MinimaxLength": "Minimax Length (CRT)"}

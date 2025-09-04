class VideoDurationCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("INT", {"default": 30, "min": 1}),
                "frame_count": ("INT", {"default": 300, "min": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "calculate_duration"
    CATEGORY = "CRT/Utils/Logic & Values"

    def calculate_duration(self, fps, frame_count):
        duration = frame_count / fps
        duration_rounded = round(duration, 2)
        return (duration_rounded,)

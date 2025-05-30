# video_duration_calculator.py

class VideoDurationCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("INT", {"default": 30, "min": 1}),
                "frame_count": ("INT", {"default": 300, "min": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)  # Change output type to FLOAT
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "calculate_duration"
    CATEGORY = "Video/Utilities"

    def calculate_duration(self, fps, frame_count):
        duration = frame_count / fps
        duration_rounded = round(duration, 2)  # round to 2 decimals
        return (duration_rounded,)

import math


class ResolutionBySide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "side_mode": (
                    ["Shorter Side", "Longer Side"],
                    {"default": "Longer Side"},
                ),
                "side_pixels": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 1}),
                "aspect_ratio": (
                    [
                        "1:1 (Square)",
                        "2:3 (Portrait)",
                        "3:4 (Portrait)",
                        "4:5 (Portrait)",
                        "5:7 (Portrait)",
                        "5:8 (Portrait)",
                        "7:9 (Portrait)",
                        "9:16 (Portrait)",
                        "9:19 (Portrait)",
                        "9:21 (Portrait)",
                        "3:2 (Landscape)",
                        "4:3 (Landscape)",
                        "5:3 (Landscape)",
                        "5:4 (Landscape)",
                        "7:5 (Landscape)",
                        "8:5 (Landscape)",
                        "9:7 (Landscape)",
                        "16:9 (Landscape)",
                        "19:9 (Landscape)",
                        "21:9 (Landscape)",
                    ],
                    {"default": "3:2 (Landscape)"},
                ),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "megapixels")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "CRT/Utils/Logic & Values"

    def calculate_dimensions(self, side_mode, side_pixels, aspect_ratio, divisible_by):
        ratio_str = aspect_ratio.split(" ")[0]

        try:
            w_ratio, h_ratio = map(int, ratio_str.split(":"))
        except ValueError:
            w_ratio, h_ratio = 1, 1

        w_ratio = max(w_ratio, 1)
        h_ratio = max(h_ratio, 1)

        if side_mode == "Shorter Side":
            # shorter side = min(w_ratio, h_ratio) direction
            if w_ratio <= h_ratio:
                # width is the shorter side
                raw_width = float(side_pixels)
                raw_height = side_pixels * h_ratio / w_ratio
            else:
                # height is the shorter side
                raw_height = float(side_pixels)
                raw_width = side_pixels * w_ratio / h_ratio
        else:  # "Longer Side"
            if w_ratio >= h_ratio:
                # width is the longer side
                raw_width = float(side_pixels)
                raw_height = side_pixels * h_ratio / w_ratio
            else:
                # height is the longer side
                raw_height = float(side_pixels)
                raw_width = side_pixels * w_ratio / h_ratio

        if divisible_by > 1:
            final_width = max(divisible_by, round(raw_width / divisible_by) * divisible_by)
            final_height = max(divisible_by, round(raw_height / divisible_by) * divisible_by)
        else:
            final_width = max(1, round(raw_width))
            final_height = max(1, round(raw_height))

        megapixels = round(final_width * final_height / 1_000_000, 3)

        return (final_width, final_height, megapixels)


NODE_CLASS_MAPPINGS = {
    "ResolutionBySide": ResolutionBySide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionBySide": "Resolution By Side (CRT)",
}

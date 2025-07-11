import math

class Resolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "longer_side": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 8}),
                "aspect_ratio": ([
                    "1:1 (Square)",
                    "2:3 (Portrait)", "3:4 (Portrait)", "4:5 (Portrait)", "5:7 (Portrait)", "5:8 (Portrait)",
                    "7:9 (Portrait)", "9:16 (Portrait)", "9:19 (Portrait)", "9:21 (Portrait)",
                    "3:2 (Landscape)", "4:3 (Landscape)", "5:3 (Landscape)", "5:4 (Landscape)", "7:5 (Landscape)", "8:5 (Landscape)",
                    "9:7 (Landscape)", "16:9 (Landscape)", "19:9 (Landscape)", "21:9 (Landscape)"
                ], {"default": "3:2 (Landscape)"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "longer_side")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "CRT/Utils/Logic & Values"

    def calculate_dimensions(self, longer_side, aspect_ratio):
        ratio_str = aspect_ratio.split(' ')[0]
        
        try:
            width_ratio, height_ratio = map(int, ratio_str.split(':'))
        except ValueError:
            width_ratio, height_ratio = 1, 1

        if width_ratio > height_ratio:
            width = longer_side
            height = (longer_side * height_ratio) / width_ratio
        elif height_ratio > width_ratio:
            height = longer_side
            width = (longer_side * width_ratio) / height_ratio
        else:
            width = longer_side
            height = longer_side

        final_width = int(round(width))
        final_height = int(round(height))
        
        return (final_width, final_height, longer_side)

NODE_CLASS_MAPPINGS = {
    "Resolution": Resolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Resolution": "Resolution",
}
import torch


class CRTPctCropCalculator:
    """
    A node that crops an input image based on percentages for width and height.
    It provides direct control over the crop amount for each dimension and outputs
    the processed image, simplifying workflows.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_percent_width": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "number"},
                ),
                "crop_percent_height": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "number"},
                ),
                "location": (
                    (
                        "center",
                        "top",
                        "bottom",
                        "left",
                        "right",
                        "top-left",
                        "top-right",
                        "bottom-left",
                        "bottom-right",
                    ),
                    {"default": "center"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "CRT/Image"

    def crop(self, image, crop_percent_width, crop_percent_height, location):
        _, original_height, original_width, _ = image.shape
        scale_factor_width = (100.0 - max(0.0, min(100.0, crop_percent_width))) / 100.0
        scale_factor_height = (100.0 - max(0.0, min(100.0, crop_percent_height))) / 100.0
        new_width = max(1, int(original_width * scale_factor_width))
        new_height = max(1, int(original_height * scale_factor_height))
        width_diff = original_width - new_width
        height_diff = original_height - new_height
        if 'left' in location:
            x_start = 0
        elif 'right' in location:
            x_start = width_diff
        else:  # center, top, bottom
            x_start = width_diff // 2
        if 'top' in location:
            y_start = 0
        elif 'bottom' in location:
            y_start = height_diff
        else:  # center, left, right
            y_start = height_diff // 2
        x_end = x_start + new_width
        y_end = y_start + new_height
        cropped_image = image[:, y_start:y_end, x_start:x_end, :]
        print(
            f"[CRTPctCrop] Original: {original_width}x{original_height} -> Cropped: {new_width}x{new_height} from '{location}'"
        )
        return (cropped_image, new_width, new_height)


NODE_CLASS_MAPPINGS = {"CRTPctCropCalculator": CRTPctCropCalculator}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- THIS LINE IS THE FIX ---
    "CRTPctCropCalculator": "Percentage Crop Calculator"
}

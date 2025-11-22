import math

class ImageDimensionsFromMegaPixels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 200.0, "step": 0.01}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 512}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "CRT/Image"

    def calculate(self, image, megapixels, multiple_of):
        _, h, w, _ = image.shape
        aspect_ratio = w / h
        target_pixels = megapixels * 1000000

        new_h = math.sqrt(target_pixels / aspect_ratio)
        new_w = new_h * aspect_ratio

        if multiple_of > 1:
            new_w = round(new_w / multiple_of) * multiple_of
            new_h = round(new_h / multiple_of) * multiple_of
        else:
            new_w = round(new_w)
            new_h = round(new_h)

        return (int(new_w), int(new_h))
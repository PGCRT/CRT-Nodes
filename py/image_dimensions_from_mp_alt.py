import math

class ImageDimensionsFromMegaPixelsAlt:
    @classmethod
    def INPUT_TYPES(cls):
        megapixel_options = [f"{i/10:.1f}" for i in range(1, 26)]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": (megapixel_options, {"default": "1.0"}),
                "multiple_of": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution")
    FUNCTION = "calculate"
    CATEGORY = "CRT/Image"

    def calculate(self, image, megapixels, multiple_of):
        _, h, w, _ = image.shape
        aspect_ratio = w / h
        
        megapixel = float(megapixels)
        target_pixels = megapixel * 1000000

        new_h = math.sqrt(target_pixels / aspect_ratio)
        new_w = new_h * aspect_ratio

        div = int(multiple_of)
        if div > 1:
            new_w = round(new_w / div) * div
            new_h = round(new_h / div) * div
        else:
            new_w = round(new_w)
            new_h = round(new_h)

        width = int(new_w)
        height = int(new_h)

        resolution = f"{width} × {height}"

        return (width, height, resolution)
import torch


COLOR_PRESETS = {
    "White":        (1.000, 1.000, 1.000),
    "Light Gray":   (0.753, 0.753, 0.753),
    "Gray":         (0.502, 0.502, 0.502),
    "Dark Gray":    (0.251, 0.251, 0.251),
    "Black":        (0.000, 0.000, 0.000),
    "Red":          (1.000, 0.000, 0.000),
    "Dark Red":     (0.502, 0.000, 0.000),
    "Orange":       (1.000, 0.502, 0.000),
    "Yellow":       (1.000, 1.000, 0.000),
    "Lime":         (0.000, 1.000, 0.000),
    "Green":        (0.000, 0.502, 0.000),
    "Cyan":         (0.000, 1.000, 1.000),
    "Sky Blue":     (0.529, 0.808, 0.922),
    "Blue":         (0.000, 0.000, 1.000),
    "Navy":         (0.000, 0.000, 0.502),
    "Magenta":      (1.000, 0.000, 1.000),
    "Pink":         (1.000, 0.753, 0.796),
    "Purple":       (0.502, 0.000, 0.502),
    "Brown":        (0.647, 0.165, 0.165),
    "Beige":        (0.961, 0.961, 0.863),
}


class SolidColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width":  ("INT",   {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT",   {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "color":  (list(COLOR_PRESETS.keys()), {"default": "White"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "CRT/Image"

    def generate(self, width, height, color):
        r, g, b = COLOR_PRESETS[color]

        image = torch.zeros([1, height, width, 3], dtype=torch.float32)
        image[..., 0] = r
        image[..., 1] = g
        image[..., 2] = b

        luma = 0.299 * r + 0.587 * g + 0.114 * b
        mask = torch.full([1, height, width], luma, dtype=torch.float32)

        return (image, mask)


NODE_CLASS_MAPPINGS = {
    "SolidColor": SolidColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SolidColor": "Solid Color (CRT)",
}

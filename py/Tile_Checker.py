import torch


class ImageTileChecker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tile"
    CATEGORY = "CRT/Image"

    def tile(self, image):
        # image: [B, H, W, C]
        row = torch.cat([image, image], dim=2)  # 2 tiles wide
        grid = torch.cat([row, row], dim=1)  # 2 tiles tall
        return (grid,)


NODE_CLASS_MAPPINGS = {
    "Image Tile Checker (CRT)": ImageTileChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Tile Checker (CRT)": "Image Tile Checker (CRT)",
}


import torch
import torch.nn.functional as F

def _make_circular_conv2d(tile_x, tile_y):
    original = F.conv2d

    def circular(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h = padding[0]
            pad_w = padding[1] if len(padding) > 1 else padding[0]

        if tile_x and pad_w > 0:
            input = F.pad(input, (pad_w, pad_w, 0, 0), mode='circular')
            pad_w = 0
        if tile_y and pad_h > 0:
            input = F.pad(input, (0, 0, pad_h, pad_h), mode='circular')
            pad_h = 0

        return original(input, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    return original, circular


def _blend_seam(image, blend_width, tile_x, tile_y):
    """Cross-fade each edge with its opposite edge over blend_width pixels."""
    img = image.clone()  # [B, H, W, C]
    B, H, W, _ = img.shape

    if tile_x and 0 < blend_width < W // 2:
        # ramp: 0.0 at the very edge → 1.0 at blend_width pixels in
        ramp = torch.linspace(0.0, 1.0, blend_width, device=img.device).view(1, 1, blend_width, 1)

        left  = img[:, :, :blend_width,  :].clone()
        right = img[:, :, W-blend_width:, :].clone()

        # left edge fades toward the right edge (reversed so adjacent pixels align)
        img[:, :, :blend_width,  :] = left  * ramp       + right.flip(2) * (1.0 - ramp)
        # right edge fades toward the left edge
        img[:, :, W-blend_width:, :] = right * ramp.flip(2) + left.flip(2)  * (1.0 - ramp.flip(2))

    if tile_y and 0 < blend_width < H // 2:
        ramp = torch.linspace(0.0, 1.0, blend_width, device=img.device).view(1, blend_width, 1, 1)

        top    = img[:, :blend_width,  :, :].clone()
        bottom = img[:, H-blend_width:, :, :].clone()

        img[:, :blend_width,  :, :] = top    * ramp       + bottom.flip(1) * (1.0 - ramp)
        img[:, H-blend_width:, :, :] = bottom * ramp.flip(1) + top.flip(1)   * (1.0 - ramp.flip(1))

    return img


class Flux2KleinSeamlessTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":       ("MODEL",),
                "vae":         ("VAE",),
                "bypass":      ("BOOLEAN", {"default": False}),
                "tiling":      (["both", "x_only", "y_only"],),
                "blend_width": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION     = "apply"
    CATEGORY     = "CRT/Flux2"

    def apply(self, model, vae, bypass, tiling, blend_width):
        if bypass:
            return (model, vae)

        tile_x = tiling in ("both", "x_only")
        tile_y = tiling in ("both", "y_only")
        original_conv2d, circular_conv2d = _make_circular_conv2d(tile_x, tile_y)

        m = model.clone()

        def unet_wrapper(model_function, params):
            F.conv2d = circular_conv2d
            try:
                return model_function(params["input"], params["timestep"], **params["c"])
            finally:
                F.conv2d = original_conv2d

        m.set_model_unet_function_wrapper(unet_wrapper)

        original_decode = vae.decode
        original_encode = vae.encode

        def seamless_decode(*args, **kwargs):
            F.conv2d = circular_conv2d
            try:
                result = original_decode(*args, **kwargs)
            finally:
                F.conv2d = original_conv2d
            if blend_width > 0:
                result = _blend_seam(result, blend_width, tile_x, tile_y)
            return result

        def seamless_encode(*args, **kwargs):
            F.conv2d = circular_conv2d
            try:
                return original_encode(*args, **kwargs)
            finally:
                F.conv2d = original_conv2d

        vae.decode = seamless_decode
        vae.encode = seamless_encode

        return (m, vae)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinSeamlessTile": Flux2KleinSeamlessTile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinSeamlessTile": "Flux2Klein Seamless Tile (CRT)",
}

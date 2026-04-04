"""
tiny_flux2_vae.py  –  ComfyUI custom nodes for FLUX.2-Tiny-AutoEncoder
=======================================================================
Nodes:
  • Tiny FLUX.2 VAE Loader  –  loads diffusion_pytorch_model.safetensors
  • Tiny FLUX.2 VAE Encode  –  IMAGE  →  LATENT  (128-ch, 16× downscale)
  • Tiny FLUX.2 VAE Decode  –  LATENT →  IMAGE

Architecture notes
------------------
fal/FLUX.2-Tiny-AutoEncoder wraps diffusers AutoencoderTiny (32-ch, 8×)
with an extra stride-2 Conv2d that maps 32 → 128 channels (+2× spatial),
giving a final 128-channel latent at H/16 × W/16.
This matches the FLUX.2 (Klein) full-VAE latent format.
"""

import os
import sys
import importlib.util

import torch
import torch.nn as nn
import safetensors.torch

import folder_paths
import comfy.model_management as mm

# ---------------------------------------------------------------------------
# Try to import diffusers (required by the model)
# ---------------------------------------------------------------------------
try:
    from diffusers.models import AutoencoderTiny
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    _DIFFUSERS_OK = True
except ImportError:
    _DIFFUSERS_OK = False

# ---------------------------------------------------------------------------
# Replicate Flux2TinyAutoEncoder inline so we don't need to sys.path-hack
# (identical to flux2_tiny_autoencoder.py shipped with the model weights)
# ---------------------------------------------------------------------------
if _DIFFUSERS_OK:
    class Flux2TinyAutoEncoder(ModelMixin, ConfigMixin):
        @register_to_config
        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            latent_channels: int = 128,
            encoder_block_out_channels=None,
            decoder_block_out_channels=None,
            act_fn: str = "silu",
            upsampling_scaling_factor: int = 2,
            num_encoder_blocks=None,
            num_decoder_blocks=None,
            latent_magnitude: float = 3.0,
            latent_shift: float = 0.5,
            force_upcast: bool = False,
            scaling_factor: float = 0.13025,
        ):
            if encoder_block_out_channels is None:
                encoder_block_out_channels = [64, 64, 64, 64]
            if decoder_block_out_channels is None:
                decoder_block_out_channels = [64, 64, 64, 64]
            if num_encoder_blocks is None:
                num_encoder_blocks = [1, 3, 3, 3]
            if num_decoder_blocks is None:
                num_decoder_blocks = [3, 3, 3, 1]

            super().__init__()
            self.tiny_vae = AutoencoderTiny(
                in_channels=in_channels,
                out_channels=out_channels,
                encoder_block_out_channels=encoder_block_out_channels,
                decoder_block_out_channels=decoder_block_out_channels,
                act_fn=act_fn,
                latent_channels=latent_channels // 4,      # 32
                upsampling_scaling_factor=upsampling_scaling_factor,
                num_encoder_blocks=num_encoder_blocks,
                num_decoder_blocks=num_decoder_blocks,
                latent_magnitude=latent_magnitude,
                latent_shift=latent_shift,
                force_upcast=force_upcast,
                scaling_factor=scaling_factor,
            )
            self.extra_encoder = nn.Conv2d(
                latent_channels // 4, latent_channels,
                kernel_size=4, stride=2, padding=1,
            )
            self.extra_decoder = nn.ConvTranspose2d(
                latent_channels, latent_channels // 4,
                kernel_size=4, stride=2, padding=1,
            )
            self.residual_encoder = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, latent_channels),
                nn.SiLU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            )
            self.residual_decoder = nn.Sequential(
                nn.Conv2d(latent_channels // 4, latent_channels // 4, kernel_size=3, padding=1),
                nn.GroupNorm(8, latent_channels // 4),
                nn.SiLU(),
                nn.Conv2d(latent_channels // 4, latent_channels // 4, kernel_size=3, padding=1),
            )

        def encode(self, x: torch.Tensor):
            encoded    = self.tiny_vae.encode(x, return_dict=False)[0]
            compressed = self.extra_encoder(encoded)
            enhanced   = self.residual_encoder(compressed) + compressed
            return enhanced

        def decode(self, z: torch.Tensor):
            decompressed = self.extra_decoder(z)
            enhanced     = self.residual_decoder(decompressed) + decompressed
            decoded      = self.tiny_vae.decode(enhanced, return_dict=False)[0]
            return decoded

else:
    Flux2TinyAutoEncoder = None   # nodes will report the missing dependency

# ---------------------------------------------------------------------------
# Model folder
# ---------------------------------------------------------------------------
_MODEL_SUBDIR = "FLUX.2-Tiny-AutoEncoder"
_MODEL_DIR    = os.path.join(folder_paths.models_dir, "vae_approx", _MODEL_SUBDIR)
_MODEL_FILE   = "diffusion_pytorch_model.safetensors"

# ---------------------------------------------------------------------------
# Internal wrapper (holds model + keeps track of device)
# ---------------------------------------------------------------------------
class _TinyFlux2VAE:
    def __init__(self, model: "Flux2TinyAutoEncoder"):
        self.model   = model.eval()
        self._device = torch.device("cpu")
        self._dtype  = torch.float32

    def to(self, device, dtype=torch.float32):
        self.model   = self.model.to(device=device, dtype=dtype)
        self._device = device
        self._dtype  = dtype
        return self

# ---------------------------------------------------------------------------
# NODE: Loader
# ---------------------------------------------------------------------------
class TinyFlux2VAELoader:
    """Load the FLUX.2-Tiny-AutoEncoder from models/vae_approx/FLUX.2-Tiny-AutoEncoder/."""

    @classmethod
    def INPUT_TYPES(cls):
        model_path = os.path.join(_MODEL_DIR, _MODEL_FILE)
        exists     = os.path.isfile(model_path)
        label      = _MODEL_FILE if exists else f"[NOT FOUND] {_MODEL_FILE}"
        return {
            "required": {
                "model_file": ([label],),
            }
        }

    RETURN_TYPES  = ("TINY_FLUX2_VAE",)
    RETURN_NAMES  = ("tiny_vae",)
    FUNCTION      = "load"
    CATEGORY      = "CRT/Flux2"

    def load(self, model_file):
        if not _DIFFUSERS_OK:
            raise RuntimeError(
                "diffusers is not installed. Run:  pip install diffusers"
            )

        model_path = os.path.join(_MODEL_DIR, _MODEL_FILE)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download from: https://huggingface.co/fal/FLUX.2-Tiny-AutoEncoder"
            )

        print(f"[TinyFlux2VAE] Loading from {model_path}")
        model = Flux2TinyAutoEncoder()
        sd    = safetensors.torch.load_file(model_path, device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[TinyFlux2VAE] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[TinyFlux2VAE] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[TinyFlux2VAE] Loaded successfully.")
        return (_TinyFlux2VAE(model),)


# ---------------------------------------------------------------------------
# NODE: Encode
# ---------------------------------------------------------------------------
class TinyFlux2VAEEncode:
    """
    Fast approximate encode: IMAGE → LATENT (128-ch, 16× downscale).
    Equivalent to the full FLUX.2 VAE Encode, but ~10× faster.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "tiny_vae": ("TINY_FLUX2_VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION     = "encode"
    CATEGORY     = "CRT/Flux2"

    def encode(self, image, tiny_vae: _TinyFlux2VAE):
        device = mm.get_torch_device()
        dtype  = mm.unet_dtype()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.float32

        tiny_vae.to(device, dtype)

        # IMAGE: (B, H, W, C) float32 [0, 1]
        # model expects  (B, C, H, W) float  [-1, 1]
        x = image.permute(0, 3, 1, 2).to(device, dtype)
        x = x * 2.0 - 1.0

        with torch.no_grad():
            latent = tiny_vae.model.encode(x)          # (B, 128, H/16, W/16)

        return ({"samples": latent.cpu().float()},)


# ---------------------------------------------------------------------------
# NODE: Decode
# ---------------------------------------------------------------------------
class TinyFlux2VAEDecode:
    """
    Fast approximate decode: LATENT (128-ch) → IMAGE.
    Equivalent to the full FLUX.2 VAE Decode, but ~10× faster.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples":  ("LATENT",),
                "tiny_vae": ("TINY_FLUX2_VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "decode"
    CATEGORY     = "CRT/Flux2"

    def decode(self, samples, tiny_vae: _TinyFlux2VAE):
        device = mm.get_torch_device()
        dtype  = mm.unet_dtype()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.float32

        tiny_vae.to(device, dtype)

        latent = samples["samples"].to(device, dtype)

        with torch.no_grad():
            img = tiny_vae.model.decode(latent)           # (B, C, H, W) [-1, 1]

        # → (B, H, W, C) [0, 1]
        img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
        img = img.permute(0, 2, 3, 1).cpu().float()
        return (img,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "TinyFlux2VAELoader": TinyFlux2VAELoader,
    "TinyFlux2VAEEncode": TinyFlux2VAEEncode,
    "TinyFlux2VAEDecode": TinyFlux2VAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TinyFlux2VAELoader": "Tiny FLUX.2 VAE Loader",
    "TinyFlux2VAEEncode": "Tiny FLUX.2 VAE Encode",
    "TinyFlux2VAEDecode": "Tiny FLUX.2 VAE Decode",
}

"""
@author: CRT
@title: CRT-Nodes
@version: 1.2.0
@project: "https://github.com/plugcrypt/CRT-Nodes",
@description: Single Blocks Arguments for LoRA Training + Professional Post-Processing + Flux Guidance Scheduling
https://discord.gg/8wYS9MBQqp
"""
from .toggle_lora_unet_blocks_L1 import ToggleLoraUnetBlocksNodeL1
from .toggle_lora_unet_blocks_L2 import ToggleLoraUnetBlocksNodeL2
from .remove_trailing_comma_node import RemoveTrailingCommaNode
from .boolean_transform_node import BooleanTransformNode
from .lora_loader_str import LoraLoaderStr
from .video_duration_calculator import VideoDurationCalculator
from .crt_post_process_node import CRTPostProcessNode
from .FluxTiledSamplerCustom import FluxTiledSamplerCustomAdvanced
from .FancySeed import FancySeed
from .FancyNoteNode import FancyNoteNode

NODE_CLASS_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": ToggleLoraUnetBlocksNodeL1,
    "Toggle Lora Unet Blocks L2": ToggleLoraUnetBlocksNodeL2,
    "Remove Trailing Comma": RemoveTrailingCommaNode,
    "Boolean Transform": BooleanTransformNode,
    "Lora Loader Str": LoraLoaderStr,
    "Video Duration Calculator": VideoDurationCalculator,
    "CRT Post-Process Suite": CRTPostProcessNode,
    "FluxTiledSamplerCustomAdvanced": FluxTiledSamplerCustomAdvanced,
    "FancySeed": FancySeed,
    "FancyNoteNode": FancyNoteNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": "Toggle Lora Unet Blocks L1",
    "Toggle Lora Unet Blocks L2": "Toggle Lora Unet Blocks L2",
    "Remove Trailing Comma": "Remove Trailing Comma",
    "Boolean Transform": "Boolean Transform",
    "Lora Loader Str": "Lora Loader Str",
    "Video Duration Calculator": "Video Duration Calculator",
    "CRT Post-Process Suite": "CRT Post-Process Suite",
    "FluxTiledSamplerCustomAdvanced": "Flux Tiled Sampler (Advanced)",
    "FancySeed": "FancySeed",
    "FancyNoteNode": "Fancy Note",
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
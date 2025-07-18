"""
@author: CRT
@title: CRT-Nodes
@version: 1.5.0
@project: "https://github.com/plugcrypt/CRT-Nodes",
@description: Set of nodes for ComfyUI
https://discord.gg/8wYS9MBQqp
"""

import folder_paths
import os

# Prevent double initialization
if 'CRT_NODES_INITIALIZED' not in globals():
    globals()['CRT_NODES_INITIALIZED'] = True
    
    print("[CRT-Nodes __init__] Importing node classes...")
    # Existing Imports
    from .py.toggle_lora_unet_blocks_L1 import ToggleLoraUnetBlocksNodeL1
    from .py.toggle_lora_unet_blocks_L2 import ToggleLoraUnetBlocksNodeL2
    from .py.remove_trailing_comma_node import RemoveTrailingCommaNode
    from .py.boolean_transform_node import BooleanTransformNode
    from .py.lora_loader_str import LoraLoaderStr
    from .py.video_duration_calculator import VideoDurationCalculator
    from .py.crt_post_process_node import CRTPostProcessNode
    from .py.FluxLoraBlocksPatcher import FluxLoraBlocksPatcher
    from .py.FluxTiledSamplerCustom import FluxTiledSamplerCustomAdvanced
    from .py.FancyNoteNode import FancyNoteNode
    from .py.Semantic_EQ_COND import FluxSemanticEncoder
    from .py.Flux_AIO_Node import FluxAIO_CRT
    from .py.FileLoaderCrawl import FileLoaderCrawl
    from .py.ImageLoaderCrawl import ImageLoaderCrawl
    from .py.MaskEmptyFloatNode import MaskEmptyFloatNode
    from .py.MaskPassOrPlaceholder import MaskPassOrPlaceholder
    from .py.latent_injection_sampler import LatentNoiseInjectionSampler
    from .py.face_enhancement_pipeline import FaceEnhancementPipeline
    from .py.face_enhancement_pipeline_with_injection import FaceEnhancementPipelineWithInjection
    from .py.flux_controlnet_sampler import FluxControlnetSampler
    from .py.flux_controlnet_sampler_with_injection import FluxControlnetSamplerWithInjection
    from .py.pony_upscale_sampler_with_injection import PonyUpscaleSamplerWithInjection
    from .py.pony_face_enhancement_pipeline_with_injection import PonyFaceEnhancementPipelineWithInjection
    from .py.SamplerSchedulerSelector import SamplerSchedulerSelector
    from .py.resolution import Resolution
    from .py.simple_knob import SimpleKnobNode
    from .py.simple_toggle import SimpleToggleNode
    from .py.simple_flux_shift_node import SimpleFluxShiftNode
    from .py.UpscaleModelAdv import CRT_UpscaleModelAdv
    from .py.smart_controlnet_apply import SmartControlNetApply
    from .py.smart_style_model_apply_dual import SmartStyleModelApplyDual
    from .py.CLIPTextEncodeFluxMerged import CLIPTextEncodeFluxMerged
    from .py.load_image_resize import LoadImageResize
    from .py.autoprompt_processor import AutopromptProcessor
    from .py.smart_preprocessor import SmartPreprocessor
    from .py.chroma_key_overlay import CRTChromaKeyOverlay
    from .py.get_first_last_frame import CRTFirstLastFrameSelector
    from .py.AdvancedStringReplace import AdvancedStringReplace
    from .py.seamless_loop_blender import SeamlessLoopBlender
    from .py.crop_by_percent import CRTPctCropCalculator
    from .py.AudioPreviewer import AudioPreviewer
    from .py.AudioCompressor import AudioCompressor
    from .py.eq_node import ParametricEQNode
    from .py.LoadVideo_ForVCaptioning import LoadVideoForVCaptioning
    from .py.load_last_media import CRTLoadLastMedia
    from .py.load_last_video import CRTLoadLastVideo
    from .py.SaveImageWithPath import SaveImageWithPath
    from .py.SaveTextWithPath import SaveTextWithPath
    from .py.VideoLoaderCrawl import VideoLoaderCrawl
    from .py.SaveVideoWithPath import SaveVideoWithPath

    print("[CRT-Nodes __init__] Registering custom model paths...")
    try:
        comfy_dir = os.path.dirname(folder_paths.__file__)
        models_dir = os.path.join(comfy_dir, "models")
        bbox_path = os.path.join(models_dir, "ultralytics", "bbox")
        segm_path = os.path.join(models_dir, "ultralytics", "segm")

        if os.path.isdir(bbox_path):
            folder_paths.add_model_folder_path("ultralytics_bbox", bbox_path)
            print("... 'ultralytics_bbox' path registered.")
        if os.path.isdir(segm_path):
            folder_paths.add_model_folder_path("ultralytics_segm", segm_path)
            print("... 'ultralytics_segm' path registered.")
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Could not register ultralytics paths. Error: {e}")

    print("[CRT-Nodes __init__] INITIALIZATION COMPLETE.")
else:
    print("[CRT-Nodes __init__] Already initialized, skipping...")

NODE_CLASS_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": ToggleLoraUnetBlocksNodeL1,
    "Toggle Lora Unet Blocks L2": ToggleLoraUnetBlocksNodeL2,
    "Remove Trailing Comma": RemoveTrailingCommaNode,
    "Boolean Transform": BooleanTransformNode,
    "Lora Loader Str": LoraLoaderStr,
    "Video Duration Calculator": VideoDurationCalculator,
    "CRT Post-Process Suite": CRTPostProcessNode,
    "FluxLoraBlocksPatcher": FluxLoraBlocksPatcher,
    "FluxTiledSamplerCustomAdvanced": FluxTiledSamplerCustomAdvanced,
    "FancyNoteNode": FancyNoteNode,
    "FluxSemanticEncoder": FluxSemanticEncoder,
    "FluxAIO_CRT": FluxAIO_CRT,
    "FileLoaderCrawl": FileLoaderCrawl,
    "ImageLoaderCrawl": ImageLoaderCrawl,
    "MaskEmptyFloatNode": MaskEmptyFloatNode,
    "MaskPassOrPlaceholder": MaskPassOrPlaceholder,
    "LatentNoiseInjectionSampler": LatentNoiseInjectionSampler,
    "FaceEnhancementPipeline": FaceEnhancementPipeline,
    "FaceEnhancementPipelineWithInjection": FaceEnhancementPipelineWithInjection,
    "FluxControlnetSampler": FluxControlnetSampler,
    "FluxControlnetSamplerWithInjection": FluxControlnetSamplerWithInjection,
    "PonyUpscaleSamplerWithInjection": PonyUpscaleSamplerWithInjection,
    "PonyFaceEnhancementPipelineWithInjection": PonyFaceEnhancementPipelineWithInjection,
    "SamplerSchedulerSelector": SamplerSchedulerSelector,
    "Resolution": Resolution,
    "SimpleKnobNode": SimpleKnobNode,
    "SimpleToggleNode": SimpleToggleNode,
    "SimpleFluxShiftNode": SimpleFluxShiftNode,
    "CRT_UpscaleModelAdv": CRT_UpscaleModelAdv,
    "SmartControlNetApply": SmartControlNetApply,
    "SmartStyleModelApplyDual": SmartStyleModelApplyDual,
    "CLIPTextEncodeFluxMerged": CLIPTextEncodeFluxMerged,
    "LoadImageResize": LoadImageResize,
    "AutopromptProcessor": AutopromptProcessor,
    "SmartPreprocessor": SmartPreprocessor,
    "CRTChromaKeyOverlay": CRTChromaKeyOverlay,
    "CRTFirstLastFrameSelector": CRTFirstLastFrameSelector,
    "AdvancedStringReplace": AdvancedStringReplace,
    "SeamlessLoopBlender": SeamlessLoopBlender,
    "CRTPctCropCalculator": CRTPctCropCalculator,
    "AudioPreviewer": AudioPreviewer,
    "AudioCompressor": AudioCompressor,
    "ParametricEQNode": ParametricEQNode,
    "LoadVideoForVCaptioning": LoadVideoForVCaptioning,
    "CRTLoadLastMedia": CRTLoadLastMedia,
    "CRTLoadLastVideo": CRTLoadLastVideo,
    "SaveImageWithPath": SaveImageWithPath,
    "SaveTextWithPath": SaveTextWithPath,
    "VideoLoaderCrawl": VideoLoaderCrawl,
    "SaveVideoWithPath": SaveVideoWithPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": "Toggle Lora Unet Blocks L1 (CRT)",
    "Toggle Lora Unet Blocks L2": "Toggle Lora Unet Blocks L2 (CRT)",
    "Remove Trailing Comma": "Remove Trailing Comma (CRT)",
    "Boolean Transform": "Boolean Transform (CRT)",
    "Lora Loader Str": "Lora Loader Str (CRT)",
    "Video Duration Calculator": "Video Duration Calculator (CRT)",
    "CRT Post-Process Suite": "Post-Process Suite (CRT)",
    "FluxLoraBlocksPatcher": "Flux LoRA Blocks Patcher (CRT)",
    "FluxTiledSamplerCustomAdvanced": "Flux Tiled Sampler Advanced (CRT)",
    "FancyNoteNode": "Fancy Note (CRT)",
    "FluxSemanticEncoder": "FLUX Semantic Encoder (CRT)",
    "FluxAIO_CRT": "FLUX All-In-One (CRT)",
    "FileLoaderCrawl": "File Loader Crawl (CRT)",
    "ImageLoaderCrawl": "Image Loader Crawl (CRT)",
    "MaskEmptyFloatNode": "Mask Empty Float (CRT)",
    "MaskPassOrPlaceholder": "Mask Pass or Placeholder (CRT)",
    "LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)",
    "FaceEnhancementPipeline": "Face Enhancement Pipeline (CRT)",
    "FaceEnhancementPipelineWithInjection": "Face Enhancement Pipeline with Injection (CRT)",
    "FluxControlnetSampler": "Flux Controlnet Sampler (CRT)",
    "FluxControlnetSamplerWithInjection": "Flux Controlnet Sampler with Injection (CRT)",
    "PonyUpscaleSamplerWithInjection": "Pony Upscale Sampler with Injection & Tiling (CRT)",
    "PonyFaceEnhancementPipelineWithInjection": "Pony Face Enhancement Pipeline with Injection (CRT)",
    "SamplerSchedulerSelector": "Sampler & Scheduler Selector (CRT)",
    "Resolution": "Resolution (CRT)",
    "SimpleKnobNode": "K",
    "SimpleToggleNode": "T",
    "SimpleFluxShiftNode": "Simple FLUX Shift (CRT)",
    "CRT_UpscaleModelAdv": "Upscale using model adv (CRT)",
    "SmartControlNetApply": "Smart ControlNet Apply (CRT)",
    "SmartStyleModelApplyDual": "Smart Style Model Apply DUAL (CRT)",
    "CLIPTextEncodeFluxMerged": "CLIP Text Encode FLUX Merged (CRT)",
    "LoadImageResize": "Load Image Resize (CRT)",
    "AutopromptProcessor": "Autoprompt Processor (CRT)",
    "SmartPreprocessor": "Smart Preprocessor (CRT)",
    "CRTChromaKeyOverlay": "Chroma Key Overlay (CRT)",
    "CRTFirstLastFrameSelector": "Get First & Last Frame (CRT)",
    "AdvancedStringReplace": "Advanced String Replace (CRT)",
    "SeamlessLoopBlender": "Seamless Loop Blender (CRT)",
    "CRTPctCropCalculator": "Percentage Crop Calculator (CRT)",
    "AudioPreviewer": "Preview Audio (CRT)",
    "AudioCompressor": "Tube Compressor (CRT)",
    "ParametricEQNode": "Parametric EQ (CRT)",
    "LoadVideoForVCaptioning": "Load Video For VCaptioning (CRT)",
    "CRTLoadLastMedia": "Load Last Image (CRT)",
    "CRTLoadLastVideo": "Load Last Video (CRT)",
    "SaveImageWithPath": "Save Image With Path (CRT)",
    "SaveTextWithPath": "Save Text With Path (CRT)",
    "VideoLoaderCrawl": "Video Loader Crawl (CRT)",
    "SaveVideoWithPath": "Save Video With Path (CRT)",
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


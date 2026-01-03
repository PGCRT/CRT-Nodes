"""
@author: CRT
@title: CRT-Nodes
@version: 1.9.9
@project: "https://github.com/plugcrypt/CRT-Nodes",
@description: Set of nodes for ComfyUI
https://discord.gg/8wYS9MBQqp
"""

import folder_paths
import os

if 'CRT_NODES_INITIALIZED' not in globals():
    globals()['CRT_NODES_INITIALIZED'] = True
    
    print("[CRT-Nodes __init__] Importing node classes...")
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
    from .py.AudioLoaderCrawl import AudioLoaderCrawl
    from .py.MaskEmptyFloatNode import MaskEmptyFloatNode
    from .py.MaskPassOrPlaceholder import MaskPassOrPlaceholder
    from .py.latent_injection_sampler import LatentNoiseInjectionSampler
    from .py.face_enhancement_pipeline_with_injection import FaceEnhancementPipelineWithInjection
    from .py.flux_controlnet_sampler import FluxControlnetSampler
    from .py.flux_controlnet_sampler_with_injection import FluxControlnetSamplerWithInjection
    from .py.pony_upscale_sampler_with_injection import PonyUpscaleSamplerWithInjection
    from .py.pony_face_enhancement_pipeline_with_injection import PonyFaceEnhancementPipelineWithInjection
    from .py.SamplerSchedulerSelector import SamplerSchedulerSelector
    from .py.SamplerSchedulerCrawler import SamplerSchedulerCrawler
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
    from .py.SaveAudioWithPath import SaveAudioWithPath
    from .py.VideoLoaderCrawl import VideoLoaderCrawl
    from .py.SaveVideoWithPath import SaveVideoWithPath
    from .py.SaveLatentWithPath import SaveLatentWithPath
    from .py.LoadLastLatent import LoadLastLatent
    from .py.EnableLatent import EnableLatent
    from .py.BooleanInvert import BooleanInvert
    from .py.StrengthToStepsNode import StrengthToStepsNode
    from .py.ClarityFX import ClarityFX
    from .py.ColourfulnessFX import ColourfulnessFX
    from .py.FilmGrainFX import FilmGrainFX
    from .py.Technicolor2FX import Technicolor2FX
    from .py.AdvancedBloomFX import AdvancedBloomFX
    from .py.LensFX import LensFX
    from .py.ContourFX import ContourFX
    from .py.ColorIsolationFX import ColorIsolationFX
    from .py.LensDistortFX import LensDistortFX
    from .py.SmartDeNoiseFX import SmartDeNoiseFX
    from .py.ArcaneBloomFX import ArcaneBloomFX
    from .py.FancyTimerNode import FancyTimerNode
    from .py.wan_compare_sampler_crt import WAN2_2LoraCompareSampler
    from .py.Add_Settings_and_Prompt import CRT_AddSettingsAndPrompt
    from .py.crt_wan_batch_sampler import CRT_WAN_BatchSampler
    from .py.crt_dynamic_prompt_scheduler import CRT_DynamicPromptScheduler
    from .py.crt_file_batch_prompt_scheduler import CRT_FileBatchPromptScheduler
    from .py.FileLoaderCrawlBatch import FileLoaderCrawlBatch
    from .py.AudioDataToFrameCount import AudioOrManualFrameCount
    from .py.EmptyContext import EmptyContext
    from .py.crt_quantize_and_crop import CRT_QuantizeAndCropImage
    from .py.crt_string_batcher import CRT_StringBatcher
    from .py.crt_string_splitter import CRT_StringSplitter
    from .py.image_dimensions_from_mp import ImageDimensionsFromMegaPixels
    from .py.WanVideoLoraSelectMultiImproved import WanVideoLoraSelectMultiImproved
    from .py.crt_ksampler_batch import CRT_KSamplerBatch
    from .py.crt_string_line_counter import CRT_StringLineCounter
    from .py.line_spot import CRT_LineSpot


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
    "AudioLoaderCrawl": AudioLoaderCrawl,
    "MaskEmptyFloatNode": MaskEmptyFloatNode,
    "MaskPassOrPlaceholder": MaskPassOrPlaceholder,
    "LatentNoiseInjectionSampler": LatentNoiseInjectionSampler,
    "FaceEnhancementPipelineWithInjection": FaceEnhancementPipelineWithInjection,
    "FluxControlnetSampler": FluxControlnetSampler,
    "FluxControlnetSamplerWithInjection": FluxControlnetSamplerWithInjection,
    "PonyUpscaleSamplerWithInjection": PonyUpscaleSamplerWithInjection,
    "PonyFaceEnhancementPipelineWithInjection": PonyFaceEnhancementPipelineWithInjection,
    "SamplerSchedulerSelector": SamplerSchedulerSelector,
    "SamplerSchedulerCrawler": SamplerSchedulerCrawler,
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
    "SaveAudioWithPath": SaveAudioWithPath,
    "VideoLoaderCrawl": VideoLoaderCrawl,
    "SaveVideoWithPath": SaveVideoWithPath,
    "SaveLatentWithPath": SaveLatentWithPath,
    "LoadLastLatent": LoadLastLatent,
    "EnableLatent": EnableLatent,
    "BooleanInvert": BooleanInvert,
    "Strength To Steps": StrengthToStepsNode,
    "ClarityFX": ClarityFX,
    "ColourfulnessFX": ColourfulnessFX,
    "FilmGrainFX": FilmGrainFX,
    "Technicolor2FX": Technicolor2FX,
    "AdvancedBloomFX": AdvancedBloomFX,
    "LensFX": LensFX,
    "ContourFX": ContourFX,
    "ColorIsolationFX": ColorIsolationFX,
    "LensDistortFX": LensDistortFX,
    "SmartDeNoiseFX": SmartDeNoiseFX,
    "ArcaneBloomFX": ArcaneBloomFX,
    "FancyTimerNode": FancyTimerNode,
    "WAN2.2 LoRA Compare Sampler": WAN2_2LoraCompareSampler,
    "CRT_AddSettingsAndPrompt": CRT_AddSettingsAndPrompt,
    "CRT_WAN_BatchSampler": CRT_WAN_BatchSampler,
    "CRT_DynamicPromptScheduler": CRT_DynamicPromptScheduler,
    "CRT_FileBatchPromptScheduler": CRT_FileBatchPromptScheduler,
    "FileLoaderCrawlBatch": FileLoaderCrawlBatch,
    "AudioOrManualFrameCount": AudioOrManualFrameCount,
    "EmptyContext": EmptyContext,
    "CRT_QuantizeAndCropImage": CRT_QuantizeAndCropImage,
    "CRT_StringBatcher": CRT_StringBatcher,
    "CRT_StringSplitter": CRT_StringSplitter,
    "ImageDimensionsFromMegaPixels": ImageDimensionsFromMegaPixels,
    "WanVideoLoraSelectMultiImproved": WanVideoLoraSelectMultiImproved,
    "CRT_KSamplerBatch": CRT_KSamplerBatch,
    "CRT_StringLineCounter": CRT_StringLineCounter,
    "Text Box line spot": CRT_LineSpot,
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
    "AudioLoaderCrawl": "Audio Loader Crawl (CRT)",
    "MaskEmptyFloatNode": "Mask Empty Float (CRT)",
    "MaskPassOrPlaceholder": "Mask Pass or Placeholder (CRT)",
    "LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)",
    "FaceEnhancementPipelineWithInjection": "Face Enhancement Pipeline with Injection (CRT)",
    "FluxControlnetSampler": "Flux Controlnet Sampler (CRT)",
    "FluxControlnetSamplerWithInjection": "Flux Controlnet Sampler with Injection (CRT)",
    "PonyUpscaleSamplerWithInjection": "Pony Upscale Sampler with Injection & Tiling (CRT)",
    "PonyFaceEnhancementPipelineWithInjection": "Pony Face Enhancement Pipeline with Injection (CRT)",
    "SamplerSchedulerSelector": "Sampler & Scheduler Selector (CRT)",
    "SamplerSchedulerCrawler": "Sampler & Scheduler Crawler (CRT)",
    "Resolution": "Resolution (CRT)",
    "SimpleKnobNode": "K",
    "SimpleToggleNode": "T",
    "SimpleFluxShiftNode": "Simple FLUX Shift (CRT)",
    "CRT_UpscaleModelAdv": "Upscale using model adv (CRT)",
    "SmartControlNetApply": "Smart ControlNet Apply (CRT)",
    "SmartStyleModelApplyDual": "Smart Style Model Apply DUAL (CRT)",
    "CLIPTextEncodeFluxMerged": "CLIP Text Encode FLUX Merged (CRT)",
    "LoadImageResize": "Load Image Resize (CRT)",
    "AutopromptProcessor": "AutopromptProcessor (CRT)",
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
    "SaveAudioWithPath": "Save Audio With Path (CRT)",
    "VideoLoaderCrawl": "Video Loader Crawl (CRT)",
    "SaveVideoWithPath": "Save Video With Path (CRT)",
    "SaveLatentWithPath": "Save Latent With Path (CRT)",
    "LoadLastLatent": "Load Last Latent (CRT)",
    "EnableLatent": "Enable Latent (CRT)",
    "BooleanInvert": "Boolean Invert (CRT)",
    "Strength To Steps": "Strength to Steps (CRT)",
    "ClarityFX": "Clarity FX (CRT)",
    "ColourfulnessFX": "Colourfulness FX (CRT)",
    "FilmGrainFX": "Film Grain FX (CRT)",
    "Technicolor2FX": "Technicolor 2 FX (CRT)",
    "AdvancedBloomFX": "Advanced Bloom FX (CRT)",
    "LensFX": "Lens FX (CRT)",
    "ContourFX": "Contour FX (CRT)",
    "ColorIsolationFX": "Color Isolation FX (CRT)",
    "LensDistortFX": "Lens Distort FX (CRT)",
    "SmartDeNoiseFX": "Smart DeNoise FX (CRT)",
    "ArcaneBloomFX": "Arcane Bloom FX (CRT)",
    "FancyTimerNode": "Fancy Timer Node",
    "WAN2.2 LoRA Compare Sampler": "WAN 2.2 LoRA Compare Sampler (CRT)",
    "CRT_AddSettingsAndPrompt": "Add Settings and Prompt (CRT)",
    "CRT_WAN_BatchSampler": "WAN 2.2 Batch Sampler (CRT)",
    "CRT_DynamicPromptScheduler": "Dynamic Prompt Scheduler (CRT)",
    "CRT_FileBatchPromptScheduler": "File Batch Prompt Scheduler (CRT)",
    "FileLoaderCrawlBatch": "File Loader Crawl Batch (CRT)",
    "AudioOrManualFrameCount": "Frame Count (Audio or Manual) (CRT)",
    "EmptyContext": "Empty Context (CRT)",
    "CRT_QuantizeAndCropImage": "Quantize and Crop Image (CRT)",
    "CRT_StringBatcher": "String Batcher (CRT)",
    "CRT_StringSplitter": "String Splitter (CRT)",
    "ImageDimensionsFromMegaPixels": "Image Dimensions From Megapixels (CRT)",
    "WanVideoLoraSelectMultiImproved": "Wan Video Multi-LoRA Select (CRT)",
    "CRT_KSamplerBatch": "KSampler Batch (CRT)",
    "CRT_StringLineCounter": "String Line Counter (CRT)",
    "Text Box line spot": "Text Box line spot (CRT)",
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

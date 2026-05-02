"""
@author: CRT
@title: CRT-Nodes
@version: 2.4.3
@project: "https://github.com/PGCRT/CRT-Nodes",
@description: Set of nodes for ComfyUI
https://discord.gg/8wYS9MBQqp
"""

import folder_paths
import os
import sys

sys.modules["crt_nodes"] = sys.modules[__name__]
__package__ = "crt_nodes"

if True:

    from .py.Remove_Trailing_Comma_Node import RemoveTrailingCommaNode
    from .py.Boolean_Transform_Node import BooleanTransformNode
    from .py.Video_Duration_Calculator import VideoDurationCalculator
    from .py.Post_Process_Node import CRTPostProcessNode
    from .py.Flux_Lora_Blocks_Patcher import FluxLoraBlocksPatcher
    from .py.Fancy_Note_Node import FancyNoteNode
    from .py.Text_Loader_Crawl import TextLoaderCrawl
    from .py.Image_Loader_Crawl import ImageLoaderCrawl
    from .py.Image_Loader_Crawl_Batch import CRT_ImageLoaderCrawlBatch
    from .py.Audio_Loader_Crawl import AudioLoaderCrawl
    from .py.Mask_Empty_Float_Node import MaskEmptyFloatNode
    from .py.Mask_Pass_Or_Placeholder import MaskPassOrPlaceholder
    from .py.Latent_Injection_Sampler import LatentNoiseInjectionSampler
    from .py.Face_Enhancement_Pipeline_With_Injection import (
        UltralyticsEnhancer as FaceEnhancementWithInjection,
    )
    from .py.Pony_Upscale_Sampler_With_Injection import PonyUpscaleSamplerWithInjection
    from .py.SEGS_Enhancer_Multi import (
        FaceEnhancementWithInjectionSEGS,
    )
    from .py.Mask_Censor import MaskCensor
    from .py.Sampler_Scheduler_Selector import SamplerSchedulerSelector
    from .py.Sampler_Scheduler_Crawler import SamplerSchedulerCrawler
    from .py.Resolution import Resolution
    from .py.Solid_Color import SolidColor
    from .py.Simple_Knob import SimpleKnobNode
    from .py.Simple_Toggle import SimpleToggleNode
    from .py.Upscale_Model_Adv import CRT_UpscaleModelAdv
    from .py.Smart_Controlnet_Apply import SmartControlNetApply
    from .py.Smart_Style_Model_Apply_Dual import SmartStyleModelApplyDual
    from .py.CLIP_Text_Encode_Flux_Merged import CLIPTextEncodeFluxMerged
    from .py.Load_Image_Resize import LoadImageResize
    from .py.Autoprompt_Processor import AutopromptProcessor
    from .py.Smart_Preprocessor import SmartPreprocessor
    from .py.Chroma_Key_Overlay import CRTChromaKeyOverlay
    from .py.Get_First_Last_Frame import CRTFirstLastFrameSelector
    from .py.Advanced_String_Replace import AdvancedStringReplace
    from .py.Seamless_Loop_Blender import SeamlessLoopBlender
    from .py.Crop_By_Percent import CRTPctCropCalculator
    from .py.Audio_Previewer import AudioPreviewer
    from .py.Audio_Compressor import AudioCompressor
    from .py.Eq_Node import ParametricEQNode
    from .py.Load_Last_Image import LoadLastImage
    from .py.Load_Last_Video import CRTLoadLastVideo
    from .py.Save_Image_With_Path import SaveImageWithPath
    from .py.Save_Text_With_Path import SaveTextWithPath
    from .py.Save_Audio_With_Path import SaveAudioWithPath
    from .py.Video_Loader_Crawl import VideoLoaderCrawl
    from .py.Save_Video_With_Path import SaveVideoWithPath
    from .py.Save_Latent_With_Path import SaveLatentWithPath
    from .py.Load_Last_Latent import LoadLastLatent
    from .py.Enable_Latent import EnableLatent
    from .py.Boolean_Invert import BooleanInvert
    from .py.Strength_To_Steps_Node import StrengthToStepsNode
    from .py.Clarity_FX import ClarityFX
    from .py.Colourfulness_FX import ColourfulnessFX
    from .py.Film_Grain_FX import FilmGrainFX
    from .py.Technicolor2_FX import Technicolor2FX
    from .py.Advanced_Bloom_FX import AdvancedBloomFX
    from .py.Lens_FX import LensFX
    from .py.Contour_FX import ContourFX
    from .py.Color_Isolation_FX import ColorIsolationFX
    from .py.Lens_Distort_FX import LensDistortFX
    from .py.Smart_De_Noise_FX import SmartDeNoiseFX
    from .py.Arcane_Bloom_FX import ArcaneBloomFX
    from .py.Fancy_Timer_Node import FancyTimerNode
    from .py.Wan_Compare_Sampler import WAN2_2LoraCompareSampler
    from .py.Add_Settings_And_Prompt import CRT_AddSettingsAndPrompt
    from .py.Wan_Batch_Sampler import CRT_WAN_BatchSampler
    from .py.Dynamic_Prompt_Scheduler import CRT_DynamicPromptScheduler
    from .py.File_Batch_Prompt_Scheduler import CRT_FileBatchPromptScheduler
    from .py.Text_Loader_Crawl_Batch import TextLoaderCrawlBatch
    from .py.Audio_Data_To_Frame_Count import AudioOrManualFrameCount
    from .py.Quantize_And_Crop import CRT_QuantizeAndCropImage
    from .py.String_Batcher import CRT_StringBatcher
    from .py.String_Splitter import CRT_StringSplitter
    from .py.Image_Dimensions_From_MP import ImageDimensionsFromMegaPixels
    from .py.Image_Dimensions_From_MP_Alt import ImageDimensionsFromMegaPixelsAlt
    from .py.Wan_Video_Lora_Select_Multi_Improved import WanVideoLoraSelectMultiImproved
    from .py.Ksampler_Batch import CRT_KSamplerBatch
    from .py.Ksampler_Batch_Advanced import CRT_KSamplerBatchAdvanced
    from .py.String_Line_Counter import CRT_StringLineCounter
    from .py.Text_Box_Line_Spot import CRT_LineSpot
    from .py.Remove_Lines import CRT_RemoveLines
    from .py.Mono_To_Stereo_Converter import MonoToStereoConverter
    from .py.Any_Trigger import AnyTrigger
    from .py.Depth_Anything_Tensorrt_Format import DepthAnythingTensorrtFormat
    from .py.Audio_Frame_Adjuster import AudioFrameAdjuster
    from .py.Batch_Brightness_Curve import BatchBrightnessCurve
    from .py.Image_Scale_Range_From_MP import ImageScaleRangeFromMp
    from .py.Load_Image_Base64 import LoadImageBase64
    from .py.Reference_Latent_Batch import ReferenceLatentBatch
    from .py.Save_Jpeg_Websocket import SaveJpegWebsocket
    from .py.Seamless_Tile import Flux2KleinSeamlessTile
    from .py.Tile_Checker import ImageTileChecker
    from .py.Scale_Latent_To_Megapixels import ScaleLatentToMegapixels
    from .py.Resolution_By_Side import ResolutionBySide
    from .py.LTX23_Unified_Sampler import (
        CRT_LTX23USConfig,
        CRT_LTX23USModelsPipe,
        CRT_LTX23UnifiedSampler,
    )
    from .py.Isolate import (
        CRT_IsolateInput,
        CRT_IsolateOutput,
    )
    from .py.Isolate_CLIPSeg import CRT_IsolateInputCLIPSeg

    CRT_LTX23AutoDownload = None
    LTX23AutoDownloadAPI = None
    try:
        from .py.LTX23_AutoDownload import (
            CRT_LTX23AutoDownload,
            LTX23AutoDownloadAPI,
        )
    except Exception as e:
        print(f"[CRT-Nodes] Warning: LTX23 AutoDownload node unavailable: {e}")

    try:
        from .py.Tiny_Flux2_VAE import (
            TinyFlux2VAELoader,
            TinyFlux2VAEEncode,
            TinyFlux2VAEDecode,
        )

        _tiny_flux2_vae_available = True
    except Exception as e:
        _tiny_flux2_vae_available = False
        print(f"[CRT-Nodes] Warning: Tiny Flux2 VAE nodes unavailable: {e}")
    SaveImageBase64 = None
    MagicLoraLoader = None
    SaveMergedLora = None
    _crt_pll_setup_routes = None
    try:
        from .py.Save_Image_Base64 import SaveImageBase64
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Save Image Base64 node unavailable: {e}")
    try:
        from .py.Magic_Lora_Loader import (
            MagicLoraLoader,
            SaveMergedLora,
            setup_routes as _crt_pll_setup_routes,
        )
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Magic LoRA Loader node unavailable: {e}")

    CRT_AudioTranscript = None
    CRT_AudioTranscriptPipeOut = None
    try:
        from .py.Audio_Transcript import CRT_AudioTranscript, CRT_AudioTranscriptPipeOut
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Audio Transcript node unavailable: {e}")

    try:
        comfy_dir = os.path.dirname(folder_paths.__file__)
        models_dir = os.path.join(comfy_dir, "models")
        bbox_path = os.path.join(models_dir, "ultralytics", "bbox")
        segm_path = os.path.join(models_dir, "ultralytics", "segm")

        if os.path.isdir(bbox_path):
            folder_paths.add_model_folder_path("ultralytics_bbox", bbox_path)
        if os.path.isdir(segm_path):
            folder_paths.add_model_folder_path("ultralytics_segm", segm_path)
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Could not register ultralytics paths. Error: {e}")

    if _crt_pll_setup_routes is not None:
        try:
            _crt_pll_setup_routes(os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            print(f"[CRT-Nodes] Warning: Could not setup CRT PLL routes. Error: {e}")

else:
    pass

NODE_CLASS_MAPPINGS = {
    "Remove Trailing Comma": RemoveTrailingCommaNode,
    "Boolean Transform": BooleanTransformNode,
    "Video Duration Calculator": VideoDurationCalculator,
    "CRT Post-Process Suite": CRTPostProcessNode,
    "FluxLoraBlocksPatcher": FluxLoraBlocksPatcher,
    "FancyNoteNode": FancyNoteNode,
    "TextLoaderCrawl": TextLoaderCrawl,
    "ImageLoaderCrawl": ImageLoaderCrawl,
    "CRT_ImageLoaderCrawlBatch": CRT_ImageLoaderCrawlBatch,
    "AudioLoaderCrawl": AudioLoaderCrawl,
    "MaskEmptyFloatNode": MaskEmptyFloatNode,
    "MaskPassOrPlaceholder": MaskPassOrPlaceholder,
    "LatentNoiseInjectionSampler": LatentNoiseInjectionSampler,
    "PonyUpscaleSamplerWithInjection": PonyUpscaleSamplerWithInjection,
    "FaceEnhancementWithInjection": FaceEnhancementWithInjection,
    "FaceEnhancementWithInjectionSEGS": FaceEnhancementWithInjectionSEGS,
    "MaskCensor": MaskCensor,
    "SamplerSchedulerSelector": SamplerSchedulerSelector,
    "SamplerSchedulerCrawler": SamplerSchedulerCrawler,
    "Resolution": Resolution,
    "SolidColor": SolidColor,
    "SimpleKnobNode": SimpleKnobNode,
    "SimpleToggleNode": SimpleToggleNode,
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
    "LoadLastImage": LoadLastImage,
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
    "TextLoaderCrawlBatch": TextLoaderCrawlBatch,
    "AudioOrManualFrameCount": AudioOrManualFrameCount,
    "CRT_QuantizeAndCropImage": CRT_QuantizeAndCropImage,
    "CRT_StringBatcher": CRT_StringBatcher,
    "CRT_StringSplitter": CRT_StringSplitter,
    "ImageDimensionsFromMegaPixels": ImageDimensionsFromMegaPixels,
    "ImageDimensionsFromMegaPixelsAlt": ImageDimensionsFromMegaPixelsAlt,
    "WanVideoLoraSelectMultiImproved": WanVideoLoraSelectMultiImproved,
    "CRT_KSamplerBatch": CRT_KSamplerBatch,
    "CRT_KSamplerBatchAdvanced": CRT_KSamplerBatchAdvanced,
    "CRT_StringLineCounter": CRT_StringLineCounter,
    "Text Box line spot": CRT_LineSpot,
    "CRT_RemoveLines": CRT_RemoveLines,
    "MonoToStereoConverter": MonoToStereoConverter,
    "AnyTrigger": AnyTrigger,
    "DepthAnythingTensorrtFormat": DepthAnythingTensorrtFormat,
    "AudioFrameAdjuster": AudioFrameAdjuster,
    "BatchBrightnessCurve": BatchBrightnessCurve,
    "ImageScaleRangeFromMp": ImageScaleRangeFromMp,
    "LoadImageBase64": LoadImageBase64,
    "ReferenceLatentBatch": ReferenceLatentBatch,
    "SaveJpegWebsocket": SaveJpegWebsocket,
    "Flux2KleinSeamlessTile": Flux2KleinSeamlessTile,
    "ImageTileChecker": ImageTileChecker,
    "ScaleLatentToMegapixels": ScaleLatentToMegapixels,
    "ResolutionBySide": ResolutionBySide,
    "CRT_LTX23USModelsPipe": CRT_LTX23USModelsPipe,
    "CRT_LTX23USConfig": CRT_LTX23USConfig,
    "CRT_LTX23UnifiedSampler": CRT_LTX23UnifiedSampler,
    "CRT_LTX23AutoDownload": CRT_LTX23AutoDownload,
    "CRT_IsolateInput": CRT_IsolateInput,
    "CRT_IsolateOutput": CRT_IsolateOutput,
    "CRT_IsolateInputCLIPSeg": CRT_IsolateInputCLIPSeg,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove Trailing Comma": "Remove Trailing Comma (CRT)",
    "Boolean Transform": "Boolean Transform (CRT)",
    "Video Duration Calculator": "Video Duration Calculator (CRT)",
    "CRT Post-Process Suite": "Post-Process Suite (CRT)",
    "FluxLoraBlocksPatcher": "Flux LoRA Blocks Patcher (CRT)",
    "FancyNoteNode": "Fancy Note (CRT)",
    "TextLoaderCrawl": "Text Loader Crawl (CRT)",
    "ImageLoaderCrawl": "Image Loader Crawl (CRT)",
    "CRT_ImageLoaderCrawlBatch": "Image Loader Crawl Batch (CRT)",
    "AudioLoaderCrawl": "Audio Loader Crawl (CRT)",
    "MaskEmptyFloatNode": "Mask Empty Float (CRT)",
    "MaskPassOrPlaceholder": "Mask Pass or Placeholder (CRT)",
    "LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)",
    "PonyUpscaleSamplerWithInjection": "Image Upscale Sampler (CRT)",
    "FaceEnhancementWithInjection": "Ultralytics Enhancer (CRT)",
    "FaceEnhancementWithInjectionSEGS": "SEGS Enhancer Multi (CRT)",
    "MaskCensor": "Mask Censor (CRT)",
    "SamplerSchedulerSelector": "Sampler & Scheduler Selector (CRT)",
    "SamplerSchedulerCrawler": "Sampler & Scheduler Crawler (CRT)",
    "Resolution": "Resolution (CRT)",
    "SimpleKnobNode": "K",
    "SimpleToggleNode": "T",
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
    "LoadLastImage": "Load Last Image (CRT)",
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
    "TextLoaderCrawlBatch": "Text Loader Crawl Batch (CRT)",
    "AudioOrManualFrameCount": "Frame Count (Audio or Manual) (CRT)",
    "CRT_QuantizeAndCropImage": "Quantize and Crop Image (CRT)",
    "CRT_StringBatcher": "String Batcher (CRT)",
    "CRT_StringSplitter": "String Splitter (CRT)",
    "ImageDimensionsFromMegaPixels": "Image Dimensions From Megapixels (CRT)",
    "ImageDimensionsFromMegaPixelsAlt": "Image Dimensions From MP alt (CRT)",
    "WanVideoLoraSelectMultiImproved": "Wan Video Multi-LoRA Select (CRT)",
    "CRT_KSamplerBatch": "KSampler Batch (CRT)",
    "CRT_KSamplerBatchAdvanced": "KSampler Batch Advanced (CRT)",
    "CRT_StringLineCounter": "String Line Counter (CRT)",
    "Text Box line spot": "Text Box line spot (CRT)",
    "CRT_RemoveLines": "Remove Lines (CRT)",
    "MonoToStereoConverter": "Mono to Stereo Converter (CRT)",
    "AnyTrigger": "Any Trigger (CRT)",
    "DepthAnythingTensorrtFormat": "Depth Anything Tensorrt Format (CRT)",
    "AudioFrameAdjuster": "Audio Frame Adjuster (CRT)",
    "BatchBrightnessCurve": "Batch Brightness Curve (U-Shape) (CRT)",
    "ImageScaleRangeFromMp": "Image Scale Range From MP (CRT)",
    "LoadImageBase64": "Load Image Base64 (CRT)",
    "ReferenceLatentBatch": "Reference Latent Batch (CRT)",
    "SaveJpegWebsocket": "Save JPEG Websocket (CRT)",
    "Flux2KleinSeamlessTile": "Flux2Klein Seamless Tile (CRT)",
    "ImageTileChecker": "Image Tile Checker (CRT)",
    "ScaleLatentToMegapixels": "Scale Latent To Megapixels (CRT)",
    "ResolutionBySide": "Resolution By Side (CRT)",
    "CRT_LTX23USModelsPipe": "LTX 2.3 US Models Pipe (CRT)",
    "CRT_LTX23USConfig": "LTX 2.3 US Config (CRT)",
    "CRT_LTX23UnifiedSampler": "LTX 2.3 Unified Sampler (CRT)",
    "CRT_LTX23AutoDownload": "LTX 2.3 AutoDownload (CRT)",
    "CRT_IsolateInput": "Isolate Input SAM3.1 (CRT)",
    "CRT_IsolateOutput": "Isolate Output (CRT)",
    "CRT_IsolateInputCLIPSeg": "Isolate Input CLIPSeg (CRT)",
}

if SaveImageBase64 is not None:
    NODE_CLASS_MAPPINGS["SaveImageBase64"] = SaveImageBase64
    NODE_DISPLAY_NAME_MAPPINGS["SaveImageBase64"] = "Save Image Base64 (CRT)"

if MagicLoraLoader is not None:
    NODE_CLASS_MAPPINGS["Magic LoRA Loader"] = MagicLoraLoader
    NODE_DISPLAY_NAME_MAPPINGS["Magic LoRA Loader"] = "Magic LoRA Loader (CRT)"

if SaveMergedLora is not None:
    NODE_CLASS_MAPPINGS["Magic Save Merged LoRA"] = SaveMergedLora
    NODE_DISPLAY_NAME_MAPPINGS["Magic Save Merged LoRA"] = (
        "Magic Save Merged LoRA (CRT)"
    )

if CRT_AudioTranscript is not None:
    NODE_CLASS_MAPPINGS["CRT_AudioTranscript"] = CRT_AudioTranscript
    NODE_DISPLAY_NAME_MAPPINGS["CRT_AudioTranscript"] = "Audio Transcript (CRT)"

if CRT_AudioTranscriptPipeOut is not None:
    NODE_CLASS_MAPPINGS["CRT_AudioTranscriptPipeOut"] = CRT_AudioTranscriptPipeOut
    NODE_DISPLAY_NAME_MAPPINGS["CRT_AudioTranscriptPipeOut"] = (
        "Audio Transcript Pipe Out (CRT)"
    )

if globals().get("_tiny_flux2_vae_available", False):
    NODE_CLASS_MAPPINGS.update(
        {
            "TinyFlux2VAELoader": TinyFlux2VAELoader,
            "TinyFlux2VAEEncode": TinyFlux2VAEEncode,
            "TinyFlux2VAEDecode": TinyFlux2VAEDecode,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "TinyFlux2VAELoader": "Tiny FLUX.2 VAE Loader (CRT)",
            "TinyFlux2VAEEncode": "Tiny FLUX.2 VAE Encode (CRT)",
            "TinyFlux2VAEDecode": "Tiny FLUX.2 VAE Decode (CRT)",
        }
    )

# Filter out None values from mappings
NODE_CLASS_MAPPINGS = {k: v for k, v in NODE_CLASS_MAPPINGS.items() if v is not None}
NODE_DISPLAY_NAME_MAPPINGS = {
    k: v for k, v in NODE_DISPLAY_NAME_MAPPINGS.items() if v is not None
}

# Setup LTX23 AutoDownload API routes
_LTX23_API_ROUTES_REGISTERED = globals().get("_LTX23_API_ROUTES_REGISTERED", False)
if LTX23AutoDownloadAPI is not None and not _LTX23_API_ROUTES_REGISTERED:
    try:
        import server
        from aiohttp import web

        @server.PromptServer.instance.routes.post("/crt/ltx23/check_models")
        async def api_check_models(request):
            try:
                result = LTX23AutoDownloadAPI.check_models()
                return web.json_response(result)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @server.PromptServer.instance.routes.post("/crt/ltx23/download_model")
        async def api_download_model(request):
            try:
                data = await request.json()
                model_type = data.get("model_type")
                result = LTX23AutoDownloadAPI.download_model_endpoint(model_type)
                return web.json_response(result)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @server.PromptServer.instance.routes.post("/crt/ltx23/download_status")
        async def api_download_status(request):
            try:
                data = await request.json()
                model_type = data.get("model_type")
                result = LTX23AutoDownloadAPI.get_download_status_endpoint(model_type)
                return web.json_response(result)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        _LTX23_API_ROUTES_REGISTERED = True

    except Exception as e:
        print(
            f"[CRT-Nodes] Warning: Could not setup LTX23 AutoDownload API routes: {e}"
        )

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


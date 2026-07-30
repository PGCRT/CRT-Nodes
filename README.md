# CRT-Nodes for ComfyUI

CRT-Nodes is a production-focused custom node suite for ComfyUI covering
image and video I/O, sampling, post-processing, model loading, LoRA workflows,
audio, image scoring, local LLM integration, and graph utilities.

Current version: **2.6.0**

## What is in this pack

- Up to 150 node registrations when optional nodes are available.
- 27 categories covering AutoDL model families, Load, Save, Text,
  Conditioning, Sampling, FX, Image, Image Scorer, LLM, Audio, LoRA, Latent,
  Video, Flux2, LTX2.3, Isolate, model patches, and utility nodes.
- Automatic Hugging Face downloads with live console progress.

## What's new in 2.6.0

### New nodes and model support

- Added `ERNIE Image Aesthetic Score (CRT)` under `CRT/Image Scorer`.
  It scores IMAGE batches from 0–100, returns the best image and images above
  a threshold, and can save qualifying images.
- Added `Unsloth Studio Bridge (CRT)` under `CRT/LLM`.
  It connects ComfyUI to the model currently loaded in Unsloth Studio and
  supports optional image input with vision-capable models.
- Added six ChronoEdit AutoDL nodes for the model, distill LoRA, upscaler LoRA,
  WAN VAE, WAN text encoder, and CLIP Vision model.

### Download experience

- All CRT Hugging Face auto-download paths now show real-time console progress:
  percentage, downloaded and total size, transfer speed, and ETA.
- Progress reporting covers general CRT AutoDL models, LTX 2.3, SAM 3.1,
  CLIPSeg, Audio Transcript models, OmniVoice, and ERNIE Image Aesthetic.
- Hugging Face snapshot progress is forced on even when the Hub client would
  otherwise suppress its progress display.

### Performance and reliability

- `Image Loader Crawl Batch (CRT)` now resizes in Pillow before float tensor
  conversion, prefetches one predicted sequential batch, honors EXIF
  orientation, and adds a `No resize` mode with center-padding for mixed sizes.
- `LTX 2.3 Unified Sampler (CRT)` now performs more reliable Depth Anything V3
  and SAM VRAM cleanup and has safer depth-preview request/resource handling.
- `KSampler Batch (CRT)` now supports additional 2D/video VAE compression APIs
  and handles independent image batches with 3D VAEs.
- `Magic LoRA Loader (CRT)` adds a `lora_stack_info` JSON output describing
  configured/effective weights, block settings, applied LoRAs, and skipped
  entries.
- `Image Scale Range From MP (CRT)` preserves aspect ratio while quantizing
  dimensions with a centered crop.
- `Save Image With Path (CRT)` saves JPG files at quality 98 with 4:4:4 chroma
  subsampling.
- Refreshed the LTX 2.3 and WAN 2.2 example workflows and removed obsolete
  SAM3/Ultralytics enhancer examples.

## Installation

### ComfyUI Manager

Search for `CRT-Nodes` and install it.

### Manual

Clone into the ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/PGCRT/CRT-Nodes.git
```

Install the requirements with the Python environment used by ComfyUI:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation or an update.

## Dependencies and optional features

Base requirements in `requirements.txt`:

- `opencv-contrib-python`
- `scipy`
- `ultralytics`
- `color-matcher`
- `spandrel`
- `pedalboard`
- `wordcloud`
- `librosa`
- `imageio-ffmpeg`
- `soundfile`
- `huggingface_hub`
- `einops`
- `rotary-embedding-torch`
- `openai-whisper`
- `omnivoice`
- `accelerate`
- `sentencepiece`
- `timm`
- `transformers`
- `tqdm`

Optional and conditional features:

- `Audio Transcript (CRT)` and `Audio Transcript Pipe Out (CRT)` depend on
  their optional runtime imports, `torchaudio`, and the bundled or compatible
  MelBand runtime. Translation is optional and requires a separate
  `llama-cpp-python` installation only when `enable_translation` is used.
- Tiny FLUX.2 VAE nodes depend on `diffusers` and FLUX.2 Tiny VAE weights in
  `models/vae_approx/FLUX.2-Tiny-AutoEncoder/`.
- `Magic LoRA Loader (CRT)` and `Magic Save Merged LoRA (CRT)` are registered
  only when their imports succeed.
- `Save Image Base64 (CRT)` is registered only when its import succeeds.
- `LTX 2.3 AutoDownload (CRT)` is registered only when its import succeeds.
- Isolate features require the corresponding SAM 3.1 or CLIPSeg support.

If an optional import fails, CRT-Nodes keeps loading and skips only the affected
nodes.

## External model and service setup

### ERNIE Image Aesthetic Score

On first use, `ERNIE Image Aesthetic Score (CRT)` downloads
`baidu/ERNIE-Image-Aes` to:

```text
ComfyUI/models/aesthetic/ERNIE-Image-Aes
```

The scorer is an 8B model. A 24 GB GPU is recommended. FlashAttention 2 is
optional.

### Unsloth Studio Bridge

Before running `Unsloth Studio Bridge (CRT)`:

1. Open Unsloth Studio.
2. Load a model and keep Studio running.
3. Normally leave `unsloth_server_url` at `http://127.0.0.1:8888`.
4. The bridge discovers the active llama-server port from Unsloth Studio logs.
5. Use a vision-capable model when connecting an IMAGE input.

The merged CRT node contains only the Studio bridge; the standalone
`prompt_deck` node and its state/word assets are not included.

## Example workflows

The `workflows/` directory contains drag-and-drop PNG workflows with embedded
ComfyUI graph metadata:

- `LTX2.3 Unified Sampler.png`
- `LTX2.3 Unified Sampler I2V.png`
- `LTX2.3 Unified Sampler_V2V_EDIT.png`
- `LTX2.3 Unified Sampler_V2V_ISOLATE.png`
- `LTX2.3 Unified Sampler_V2V_ISOLATE_TRANSLATE.png`
- `WAN 2.2 LORA COMPARE.png`

The LTX and WAN examples were refreshed for 2.6.0. Older SAM3 and Ultralytics
enhancer examples were removed because they no longer represented the current
recommended workflows.

## Verified node catalog

The catalog below is generated from the node registrations available in
version 2.6.0. Conditional nodes are marked explicitly.

### CRT/Audio (8)

- `Audio Frame Adjuster (CRT)`
- `Audio Transcript (CRT)` (conditional)
- `Audio Transcript Pipe Out (CRT)` (conditional)
- `Frame Count (Audio or Manual) (CRT)`
- `Mono to Stereo Converter (CRT)`
- `Parametric EQ (CRT)`
- `Preview Audio (CRT)`
- `Tube Compressor (CRT)`

### CRT/AutoDL/ChronoEdit (6)

- `ChronoEdit CLIP - WAN (CRT AutoDL)`
- `ChronoEdit CLIP Vision (CRT AutoDL)`
- `ChronoEdit Distill LoRA (CRT AutoDL)`
- `ChronoEdit Model (CRT AutoDL)`
- `ChronoEdit Upscaler LoRA (CRT AutoDL)`
- `ChronoEdit VAE (CRT AutoDL)`

### CRT/AutoDL/ERNIE (5)

- `ERNIE CLIP (CRT AutoDL)`
- `ERNIE Model (CRT AutoDL)`
- `ERNIE Turbo Model (CRT AutoDL)`
- `ERNIE Turbo NVFP4 Model (CRT AutoDL)`
- `ERNIE VAE (CRT AutoDL)`

### CRT/AutoDL/FLUXKLEIN (4)

- `Flux2Klein CLIP (CRT AutoDL)`
- `Flux2Klein HDRI LoRA (CRT AutoDL)`
- `Flux2Klein Model (CRT AutoDL)`
- `Flux2Klein VAE (CRT AutoDL)`

### CRT/AutoDL/KREA2 (4)

- `Krea 2 CLIP (CRT AutoDL)`
- `Krea 2 Raw Model (CRT AutoDL)`
- `Krea 2 Turbo Model (CRT AutoDL)`
- `Krea 2 VAE (CRT AutoDL)`

### CRT/AutoDL/LTX2.3 (11)

- `LTX2.3 AUDIO VAE (CRT AutoDL)`
- `LTX2.3 CLIP (CRT AutoDL)`
- `LTX2.3 IC Cnet LoRA (CRT AutoDL)`
- `LTX2.3 IC Outpaint LoRA (CRT AutoDL)`
- `LTX2.3 IC Upscale LoRA (CRT AutoDL)`
- `LTX2.3 Latent Upscaler (CRT AutoDL)`
- `LTX2.3 Model (CRT AutoDL)`
- `LTX2.3 Model GGUF Q4_K_M (CRT AutoDL)`
- `LTX2.3 Model GGUF Q5_K_M (CRT AutoDL)`
- `LTX2.3 Model NVFP4 (CRT AutoDL)`
- `LTX2.3 VIDEO VAE (CRT AutoDL)`

### CRT/AutoDL/ZIMAGETURBO (3)

- `Z-Image Turbo CLIP (CRT AutoDL)`
- `Z-Image Turbo Model (CRT AutoDL)`
- `Z-Image Turbo VAE (CRT AutoDL)`

### CRT/Conditioning (6)

- `CLIP Text Encode + Unload (CRT)`
- `CLIP Text Encode FLUX Merged (CRT)`
- `Dynamic Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler (CRT)`
- `Smart ControlNet Apply (CRT)`
- `Smart Style Model Apply DUAL (CRT)`

### CRT/Flux2 (4)

- `Flux2Klein Seamless Tile (CRT)`
- `Tiny FLUX.2 VAE Decode (CRT)` (conditional)
- `Tiny FLUX.2 VAE Encode (CRT)` (conditional)
- `Tiny FLUX.2 VAE Loader (CRT)` (conditional)

### CRT/FX (12)

- `Advanced Bloom FX (CRT)`
- `Arcane Bloom FX (CRT)`
- `Clarity FX (CRT)`
- `Color Isolation FX (CRT)`
- `Colourfulness FX (CRT)`
- `Contour FX (CRT)`
- `Film Grain FX (CRT)`
- `Lens Distort FX (CRT)`
- `Lens FX (CRT)`
- `Post-Process Suite (CRT)`
- `Smart DeNoise FX (CRT)`
- `Technicolor 2 FX (CRT)`

### CRT/Image (12)

- `Batch Brightness Curve (U-Shape) (CRT)`
- `Chroma Key Overlay (CRT)`
- `Depth Anything Tensorrt Format (CRT)`
- `Image Dimensions From Megapixels (CRT)`
- `Image Dimensions From MP alt (CRT)`
- `Image Scale Range From MP (CRT)`
- `Image Tile Checker (CRT)`
- `Percentage Crop Calculator (CRT)`
- `Quantize and Crop Image (CRT)`
- `Smart Preprocessor (CRT)`
- `Solid Color (CRT)`
- `Upscale Model Advanced (CRT)`

### CRT/Image Scorer (1)

- `ERNIE Image Aesthetic Score (CRT)`

### CRT/Latent (3)

- `Enable Latent (CRT)`
- `Reference Latent Batch (CRT)`
- `Scale Latent To Megapixels (CRT)`

### CRT/LLM (1)

- `Unsloth Studio Bridge (CRT)`

### CRT/Load (11)

- `Audio Loader Crawl (CRT)`
- `Image Loader Crawl (CRT)`
- `Image Loader Crawl Batch (CRT)`
- `Load Image Base64 (CRT)`
- `Load Image Resize (CRT)`
- `Load Last Image (CRT)`
- `Load Last Latent (CRT)`
- `Load Last Video (CRT)`
- `Text Loader Crawl (CRT)`
- `Text Loader Crawl Batch (CRT)`
- `Video Loader Crawl (CRT)`

### CRT/Logic (3)

- `Any Trigger (CRT)`
- `Boolean Invert (CRT)`
- `Strength to Steps (CRT)`

### CRT/LoRA (4)

- `Flux LoRA Blocks Patcher (CRT)`
- `Magic LoRA Loader (CRT)`
- `Magic Save Merged LoRA (CRT)`
- `Wan Video Multi-LoRA Select (CRT)`

### CRT/LTX2.3 (4)

- `LTX 2.3 AutoDownload (CRT)` (conditional)
- `LTX 2.3 Unified Sampler (CRT)`
- `LTX 2.3 US Config (CRT)`
- `LTX 2.3 US Models Pipe (CRT)`

### CRT/Mask (2)

- `Mask Censor (CRT)`
- `Mask Temporal Enhancer (CRT)`

### CRT/Model Patches (1)

- `Ideogram 4 FlashAttention (CRT)`

### CRT/Sampling (8)

- `Image Upscale Sampler (CRT)`
- `KSampler Batch (CRT)`
- `KSampler Batch Advanced (CRT)`
- `Latent Noise Injection Sampler (CRT)`
- `SEGS Enhancer Multi (CRT)`
- `Ultralytics Enhancer (CRT)`
- `WAN 2.2 Batch Sampler (CRT)`
- `WAN 2.2 LoRA Compare Sampler (CRT)`

### CRT/Save (7)

- `Save Audio With Path (CRT)`
- `Save Image Base64 (CRT)` (conditional)
- `Save Image With Path (CRT)`
- `Save JPEG Websocket (CRT)`
- `Save Latent With Path (CRT)`
- `Save Text With Path (CRT)`
- `Save Video With Path (CRT)`

### CRT/Text (11)

- `Add Settings and Prompt (CRT)`
- `Advanced String Replace (CRT)`
- `AutopromptProcessor (CRT)`
- `Join Strings (CRT)`
- `Remove Lines (CRT)`
- `Remove Trailing Comma (CRT)`
- `String Batcher (CRT)`
- `String Line Counter (CRT)`
- `String Splitter (CRT)`
- `Text Box line spot (CRT)`
- `Textbox (CRT)`

### CRT/Utils/Isolate (3)

- `Isolate Input CLIPSeg (CRT)`
- `Isolate Input SAM3.1 (CRT)`
- `Isolate Output (CRT)`

### CRT/Utils/Logic & Values (9)

- `Boolean Transform (CRT)`
- `Int Value (CRT)`
- `Mask Empty Float (CRT)`
- `Mask Pass or Placeholder (CRT)`
- `Resolution (CRT)`
- `Resolution By Side (CRT)`
- `Sampler & Scheduler Crawler (CRT)`
- `Sampler & Scheduler Selector (CRT)`
- `Video Duration Calculator (CRT)`

### CRT/Utils/UI (4)

- `Fancy Note (CRT)`
- `Fancy Timer (CRT)`
- `K`
- `T`

### CRT/Video (3)

- `Even Batch Picker (CRT)`
- `Get First & Last Frame (CRT)`
- `Seamless Loop Blender (CRT)`

## Links

- Repository: [PGCRT/CRT-Nodes](https://github.com/PGCRT/CRT-Nodes)
- Comfy Registry package: `crt-nodes`
- Community: [Discord](https://discord.gg/MqQeQvYcPA)

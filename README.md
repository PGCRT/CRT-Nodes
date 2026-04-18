# CRT-Nodes for ComfyUI

CRT-Nodes is a large custom node suite for ComfyUI focused on production workflows: image/video loading and saving, FX and post, FLUX/WAN/LTX sampling, LoRA utilities, audio tools, and graph helpers.

## What is in this pack

- Up to 106 total node registrations when optional nodes are available
- Categories covering Load, Save, Text, Conditioning, Sampling, FX, Image, Audio, LoRA, Latent, Video, Flux2, LTX2.3, and utility logic/UI

## Installation

### ComfyUI Manager

Search for `CRT-Nodes` and install.

### Manual

Clone into your custom nodes folder:

```bash
git clone https://github.com/PGCRT/CRT-Nodes.git
```

Install requirements:

```bash
pip install -r requirements.txt
```

Restart ComfyUI.

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

Optional/conditional nodes:

- `Audio Transcript (CRT)` depends on `whisper`, `torchaudio`, and MelBand runtime sources from `custom_nodes/ComfyUI-MelBandRoFormer/model/*.py`.
- Tiny FLUX.2 VAE nodes depend on `diffusers` and FLUX.2 Tiny VAE weights in `models/vae_approx/FLUX.2-Tiny-AutoEncoder/`.
- `Magic LoRA Loader (CRT)` and `Magic Save Merged LoRA (CRT)` are conditionally registered if their import succeeds.
- `Save Image Base64 (CRT)` is conditionally registered if its import succeeds.
- LTX 2.3 AutoDownload uses built-in HTTP download utilities and adds API routes for model status/download.

If optional imports fail, CRT-Nodes keeps loading and skips only affected nodes. The full catalog below lists every node that can be registered, with conditional nodes marked explicitly.

## Verified node catalog

Below is the full catalog grouped by each node `CATEGORY`.

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

### CRT/Save (6)

- `Save Audio With Path (CRT)`
- `Save Image Base64 (CRT)` (conditional)
- `Save Image With Path (CRT)`
- `Save Latent With Path (CRT)`
- `Save Text With Path (CRT)`
- `Save Video With Path (CRT)`

### CRT/Text (9)

- `Add Settings and Prompt (CRT)`
- `Advanced String Replace (CRT)`
- `AutopromptProcessor (CRT)`
- `Remove Lines (CRT)`
- `Remove Trailing Comma (CRT)`
- `String Batcher (CRT)`
- `String Line Counter (CRT)`
- `String Splitter (CRT)`
- `Text Box line spot (CRT)`

### CRT/Conditioning (5)

- `CLIP Text Encode FLUX Merged (CRT)`
- `Dynamic Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler (CRT)`
- `Smart ControlNet Apply (CRT)`
- `Smart Style Model Apply DUAL (CRT)`

### CRT/Utils/Logic & Values (8)

- `Boolean Transform (CRT)`
- `Mask Empty Float (CRT)`
- `Mask Pass or Placeholder (CRT)`
- `Resolution (CRT)`
- `Resolution By Side (CRT)`
- `Sampler & Scheduler Crawler (CRT)`
- `Sampler & Scheduler Selector (CRT)`
- `Video Duration Calculator (CRT)`

### CRT/Utils/UI (4)

- `Fancy Note (CRT)`
- `Fancy Timer Node`
- `K`
- `T`

### CRT/Logic (3)

- `Any Trigger (CRT)`
- `Boolean Invert (CRT)`
- `Strength to Steps (CRT)`

### CRT/Audio (7)

- `Audio Frame Adjuster (CRT)`
- `Audio Transcript (CRT)` (conditional)
- `Frame Count (Audio or Manual) (CRT)`
- `Mono to Stereo Converter (CRT)`
- `Parametric EQ (CRT)`
- `Preview Audio (CRT)`
- `Tube Compressor (CRT)`

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
- `SolidColor`
- `Upscale using model adv (CRT)`

### CRT/Sampling (10)

- `Flux1 Cnet Sampler with Injection (CRT)`
- `Flux1 Cnet Ultralytics Multi (CRT)`
- `Image Upscale Sampler (CRT)`
- `KSampler Batch (CRT)`
- `KSampler Batch Advanced (CRT)`
- `Latent Noise Injection Sampler (CRT)`
- `SEGS Enhancer Multi (CRT)`
- `Ultralytics Enhancer (CRT)`
- `WAN 2.2 Batch Sampler (CRT)`
- `WAN 2.2 LoRA Compare Sampler (CRT)`

### CRT/LoRA (3)

- `Flux LoRA Blocks Patcher (CRT)`
- `Magic LoRA Loader (CRT)` (conditional)
- `Magic Save Merged LoRA (CRT)` (conditional)

### CRT/Latent (3)

- `Enable Latent (CRT)`
- `Reference Latent Batch (CRT)`
- `Scale Latent To Megapixels (CRT)`

### CRT/Video (2)

- `Get First & Last Frame (CRT)`
- `Seamless Loop Blender (CRT)`

### CRT/Mask (1)

- `Mask Censor (CRT)`

### CRT/Flux2 (4)

- `Flux2Klein Seamless Tile (CRT)`
- `Tiny FLUX.2 VAE Loader (CRT)` (conditional)
- `Tiny FLUX.2 VAE Encode (CRT)` (conditional)
- `Tiny FLUX.2 VAE Decode (CRT)` (conditional)

### CRT/LTX2.3 (4)

- `LTX 2.3 AutoDownload (CRT)` (conditional)
- `LTX 2.3 Unified Sampler (CRT)`
- `LTX 2.3 US Config (CRT)`
- `LTX 2.3 US Models Pipe (CRT)`

### image (1)

- `Save JPEG Websocket (CRT)`

### WanVideo/Loaders (1)

- `Wan Video Multi-LoRA Select (CRT)`




- Repository: `https://github.com/PGCRT/CRT-Nodes`
- Comfy Registry package name: `crt-nodes`
- https://discord.gg/MqQeQvYcPA

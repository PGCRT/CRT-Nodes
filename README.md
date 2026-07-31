# CRT-Nodes for ComfyUI

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

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

> **Note on updates:** Nodes that have changed inputs, outputs, or widget names may appear red or show `NaN` values in existing workflows. Right-click the node and select **Fix node (recreate)** to refresh its sockets.

### CRT/Audio (8)

- `Audio Frame Adjuster (CRT)`
- `Audio Transcript Batch (CRT)`
- `Frame Count (Audio or Manual) (CRT)`
- `Mono to Stereo Converter (CRT)`
- `Parametric EQ (CRT)`
- `Preview Audio (CRT)`
- `Tube Compressor (CRT)`
- `YuE Music Generator (CRT)`

### CRT/AutoDL/ChronoEdit (6)

- `ChronoEdit CLIP - WAN (CRT AutoDL)`
- `ChronoEdit CLIP Vision (CRT AutoDL)`
- `ChronoEdit Distill LoRA (CRT AutoDL)`
- `ChronoEdit Model (CRT AutoDL)`
- `ChronoEdit Upscaler LoRA (CRT AutoDL)`
- `ChronoEdit VAE (CRT AutoDL)`

### CRT/AutoDL/ERNIE (4)

- `ERNIE CLIP (CRT AutoDL)`
- `ERNIE Model (CRT AutoDL)`
- `ERNIE Turbo Model (CRT AutoDL)`
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

### CRT/AutoDL/LTX2.5 (11)

- `LTX2.5 AUDIO VAE (CRT AutoDL)`
- `LTX2.5 CLIP w4a8 Light (CRT AutoDL)`
- `LTX2.5 Duration Head (CRT AutoDL)`
- `LTX2.5 IC Cnet LoRA (CRT AutoDL)`
- `LTX2.5 IC Outpaint LoRA (CRT AutoDL)`
- `LTX2.5 IC Pixel Spatial Upscale LoRA (CRT AutoDL)`
- `LTX2.5 IC Upscale LoRA (CRT AutoDL)`
- `LTX2.5 Model (CRT AutoDL)`
- `LTX2.5 Spatial Upscaler (CRT AutoDL)`
- `LTX2.5 Temporal Upscaler (CRT AutoDL)`
- `LTX2.5 VIDEO VAE (CRT AutoDL)`

### CRT/AutoDL/MINIMAXH3 (7)

- `MiniMax H3 AUDIO VAE (CRT AutoDL)`
- `MiniMax H3 CLIP (CRT AutoDL)`
- `MiniMax H3 FL2VA Turbo LoRA (CRT AutoDL)`
- `MiniMax H3 Fun ControlNet (CRT AutoDL)`
- `MiniMax H3 Model (CRT AutoDL)`
- `MiniMax H3 REF2VA Turbo LoRA (CRT AutoDL)`
- `MiniMax H3 VIDEO VAE (CRT AutoDL)`

### CRT/AutoDL/MOSSIMAGE (3)

- `MOSS-Image CLIP Qwen3.5-9B (CRT AutoDL)`
- `MOSS-Image Model (CRT AutoDL)`
- `MOSS-Image VAE (CRT AutoDL)`

### CRT/AutoDL/Pixal3D (6)

- `BiRefNet Background Removal (CRT AutoDL)`
- `MoGe-2 Model (CRT AutoDL)`
- `Pixal3D CLIP Vision DINOv3 (CRT AutoDL)`
- `Pixal3D Model (CRT AutoDL)`
- `Pixal3D Shape VAE (CRT AutoDL)`
- `Pixal3D Texture VAE (CRT AutoDL)`

### CRT/AutoDL/QWEN2.1 (4)

- `Qwen-Image 2.1 CLIP Qwen3VL-8B (CRT AutoDL)`
- `Qwen-Image 2.1 Model (CRT AutoDL)`
- `Qwen-Image 2.1 Turbo LoRA 4-Step Lightx2v (CRT AutoDL)`
- `Qwen-Image 2.1 VAE (CRT AutoDL)`

### CRT/AutoDL/SAM3Body (4)

- `MoGe-2 Model (CRT AutoDL)`
- `SAM 3.1 Checkpoint (CRT AutoDL)`
- `SAM 3D Body Model (CRT AutoDL)`
- `SAM3DBody RT-DETR Detector (CRT AutoDL)`

### CRT/AutoDL/ZIMAGETURBO (3)

- `Z-Image Turbo CLIP (CRT AutoDL)`
- `Z-Image Turbo Model (CRT AutoDL)`
- `Z-Image Turbo VAE (CRT AutoDL)`

### CRT/Conditioning (4)

- `CLIP Text Encode + Unload (CRT)`
- `Dynamic Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler KREA2 (CRT)`

### CRT/Depth (2)

- `Fast Depth Anything v3 (CRT)`
- `Heavy Depth Marigold v2 (CRT)`

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

### CRT/Image (11)

- `Batch Brightness Curve (U-Shape) (CRT)`
- `Chroma Key Overlay (CRT)`
- `Depth Anything Tensorrt Format (CRT)`
- `Image Dimensions From Megapixels (CRT)`
- `Image Dimensions From MP alt (CRT)`
- `Image Scale Range From MP (CRT)`
- `Image Tile Checker (CRT)`
- `Image Unbatch (CRT)`
- `Percentage Crop Calculator (CRT)`
- `Quantize and Crop Image (CRT)`
- `Solid Color (CRT)`

### CRT/Image Scorer (1)

- `ERNIE Image Aesthetic Score (CRT)`

### CRT/Latent (4)

- `Enable Latent (CRT)`
- `Reference Latent Batch (CRT)`
- `Scale Latent To Megapixels (CRT)`
- `VAE Decode Last Frame (CRT)`

### CRT/LLM (2)

- `OpenCode Go Bridge (CRT)`
- `OpenCode Go Thinking Display (CRT)`

### CRT/Load (14)

- `Audio Loader Crawl (CRT)`
- `Audio Loader Crawl Batch (CRT)`
- `FileSearch Crawl (CRT)`
- `Image Loader Crawl (CRT)`
- `Image Loader Crawl Batch (CRT)`
- `Load Image Base64 (CRT)`
- `Load Image Resize (CRT)`
- `Load Last Image (CRT)`
- `Load Last Latent (CRT)`
- `Load Last Video (CRT)`
- `Load Latents Conditioning (CRT)`
- `Text Loader Crawl (CRT)`
- `Text Loader Crawl Batch (CRT)`
- `Video Loader Crawl (CRT)`

### CRT/Logic (3)

- `Any Trigger (CRT)`
- `Boolean Invert (CRT)`
- `Strength to Steps (CRT)`

### CRT/LoRA (3)

- `Magic LoRA Loader (CRT)`
- `Magic Save Merged LoRA (CRT)`
- `Wan Video Multi-LoRA Select (CRT)`

### CRT/LTX2.5 (3)

- `LTX Unified Sampler (CRT)`
- `LTX US Config (CRT)`
- `LTX US Models Pipe (CRT)`

### CRT/Mask (2)

- `Mask Censor (CRT)`
- `Mask Temporal Enhancer (CRT)`

### CRT/MiniMaxH3 (5)

- `MiniMax H3 Fun Control Apply (CRT)`
- `MiniMax H3 Unified Sampler (CRT)`
- `MiniMax H3 US Config (CRT)`
- `MiniMax H3 US Models Pipe (CRT)`
- `MiniMax H3 US Models Pipe CNET (CRT)`

### CRT/Sampling (9)

- `Image Upscale Sampler (CRT)`
- `KSampler Batch (CRT)`
- `KSampler Batch Advanced (CRT)`
- `KSampler LoRA Batch (CRT)`
- `Latent Noise Injection Sampler (CRT)`
- `SEGS Enhancer Multi (CRT)`
- `Ultralytics Enhancer (CRT)`
- `WAN 2.2 Batch Sampler (CRT)`
- `WAN 2.2 LoRA Compare Sampler (CRT)`

### CRT/Save (8)

- `Save Audio With Path (CRT)`
- `Save Image Base64 (CRT)`
- `Save Image With Path (CRT)`
- `Save JPEG Websocket (CRT)`
- `Save Latent With Path (CRT)`
- `Save Latents Conditioning (CRT)`
- `Save Text With Path (CRT)`
- `Save Video With Path (CRT)`

### CRT/Text (14)

- `Add Settings and Prompt (CRT)`
- `AutopromptProcessor (CRT)`
- `Extract Q/A (CRT)`
- `Image Label (CRT)`
- `Join Strings (CRT)`
- `Merge Q/A (CRT)`
- `Remove Lines (CRT)`
- `String Batcher (CRT)`
- `String Line Counter (CRT)`
- `String Splitter (CRT)`
- `Text Add Rows (CRT)`
- `Text Rows Crawl (CRT)`
- `Textbox (CRT)`
- `TextBox line spot (CRT)`

### CRT/Utils/Isolate (3)

- `Isolate Input CLIPSeg (CRT)`
- `Isolate Input SAM3.1 (CRT)`
- `Isolate Output (CRT)`

### CRT/Utils/Logic & Values (11)

- `Int to String (CRT)`
- `Int Value (CRT)`
- `Mask Empty Float (CRT)`
- `Mask Pass or Placeholder (CRT)`
- `Minimax Length (CRT)`
- `Resolution (CRT)`
- `Resolution By Side (CRT)`
- `Sampler & Scheduler Crawler (CRT)`
- `Sampler & Scheduler Selector (CRT)`
- `String to Boolean (CRT)`
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

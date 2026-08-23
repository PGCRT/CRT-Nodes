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
- `Audio Transcript Batch (CRT)` (conditional)
- `Audio Transcript turbo (CRT ALT)`
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

### CRT/AutoDL/MINIMAXH3 (4)

- `MiniMax H3 CLIP (CRT AutoDL)`
- `MiniMax H3 Model (CRT AutoDL)`
- `MiniMax H3 AUDIO VAE (CRT AutoDL)`
- `MiniMax H3 VIDEO VAE (CRT AutoDL)`

### CRT/AutoDL/ZIMAGETURBO (3)

- `Z-Image Turbo CLIP (CRT AutoDL)`
- `Z-Image Turbo Model (CRT AutoDL)`
- `Z-Image Turbo VAE (CRT AutoDL)`

### CRT/Conditioning (4)

- `CLIP Text Encode + Unload (CRT)`
- `Dynamic Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler (CRT)`
- `File Batch Prompt Scheduler KREA2 (CRT)`

### CRT/DepthAnything3 (1)

- `DepthAnything3 (CRT)`

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

### CRT/Image (10)

- `Batch Brightness Curve (U-Shape) (CRT)`
- `Chroma Key Overlay (CRT)`
- `Depth Anything Tensorrt Format (CRT)`
- `Image Dimensions From Megapixels (CRT)`
- `Image Dimensions From MP alt (CRT)`
- `Image Scale Range From MP (CRT)`
- `Image Tile Checker (CRT)`
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

### CRT/LLM (3)

- `Kimi Inference Bridge (CRT)`
- `LM Studio Bridge (CRT)`
- `Unsloth Studio Bridge (CRT)`

### CRT/Load (13)

- `Audio Loader Crawl (CRT)`
- `Audio Loader Crawl Batch (CRT)`
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

### CRT/LoRA (6)

- `Flux LoRA Blocks Patcher (CRT)`
- `Magic LoRA Loader (CRT)`
- `Magic Save Merged LoRA (CRT)`
- `Seeded Persona LoRA Crawl Batch (CRT)`
- `Seeded Persona LoRA Loader (CRT)`
- `Wan Video Multi-LoRA Select (CRT)`

### CRT/LTX2.5 (3)

- `LTX US Models Pipe (CRT)`
- `LTX Unified Sampler (CRT)`
- `LTX US Config (CRT)`

### CRT/Mask (2)

- `Mask Censor (CRT)`
- `Mask Temporal Enhancer (CRT)`

### CRT/Sampling (8)

- `Image Upscale Sampler (CRT)`
- `KSampler Batch (CRT)`
- `KSampler Batch Advanced (CRT)`
- `Latent Noise Injection Sampler (CRT)`
- `SEGS Enhancer Multi (CRT)`
- `Ultralytics Enhancer (CRT)`
- `WAN 2.2 Batch Sampler (CRT)`
- `WAN 2.2 LoRA Compare Sampler (CRT)`

### CRT/Save (8)

- `Save Audio With Path (CRT)`
- `Save Image Base64 (CRT)` (conditional)
- `Save Image With Path (CRT)`
- `Save JPEG Websocket (CRT)`
- `Save Latent With Path (CRT)`
- `Save Latents Conditioning (CRT)`
- `Save Text With Path (CRT)`
- `Save Video With Path (CRT)`

### CRT/Text (14)

- `Add Settings and Prompt (CRT)`
- `AutopromptProcessor (CRT)`
- `Extract Dialogues MiniMaxH3 (CRT)`
- `Extract Q/A (CRT)`
- `Join Strings (CRT)`
- `Merge Q/A (CRT)`
- `Remove Lines (CRT)`
- `String Batcher (CRT)`
- `String Line Counter (CRT)`
- `String Splitter (CRT)`
- `Text Add Rows (CRT)`
- `TextBox line spot (CRT)`
- `Text Rows Crawl (CRT)`
- `Textbox (CRT)`

### CRT/Utils/Isolate (3)

- `Isolate Input CLIPSeg (CRT)`
- `Isolate Input SAM3.1 (CRT)`
- `Isolate Output (CRT)`

### CRT/Utils/Logic & Values (10)

- `String to Boolean (CRT)`
- `Int Value (CRT)`
- `Mask Empty Float (CRT)`
- `Mask Pass or Placeholder (CRT)`
- `Minimax Length (CRT)`
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

## Notes

### Unsloth Studio Bridge (CRT)

Connects to the model currently loaded in [Unsloth Studio](https://github.com/unslothai/studio) and chats with it through its local llama-server.

- `unload_model_after_run`: unloads the model from VRAM after the response is generated so the rest of the workflow gets the full GPU. The next run automatically reloads the same model via the Studio API with its previous settings (context, parallel slots, KV cache type, speculative decoding, GPU layers).
- `studio_api_key`: single optional credential. Paste an `sk-unsloth-...` API key or your Studio password, or leave empty for automatic local authentication. Required only for unload/reload.
- If llama-server is down when a run starts (e.g. unloaded after the previous run), the bridge reloads it and fails within seconds if Unsloth Studio itself is not running.

## Links

- Repository: [PGCRT/CRT-Nodes](https://github.com/PGCRT/CRT-Nodes)
- Comfy Registry package: `crt-nodes`
- Community: [Discord](https://discord.gg/MqQeQvYcPA)

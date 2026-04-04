# CRT-Nodes for ComfyUI

Custom node collection for ComfyUI covering FX/post, loaders/savers, audio, and workflow utilities.

**Note:** Some nodes require optional dependencies such as `pedalboard`, `ultralytics`, `color-matcher`, and `sageattention`.

---

## Node List

### Audio Processing
Tools for manipulating audio, visualizing waveforms, and syncing audio to frame counts.

Sorted by each node's `CATEGORY` in the Python scripts.

#### Load

| Node Name | Description |
| :--- | :--- |
| **Tube Compressor** | Audio compressor with threshold/ratio/attack/release and musical shaping controls. |
| **Parametric EQ** | Multi-band EQ with standard filter shapes for corrective and creative tone control. |
| **Preview Audio** | In-UI audio preview with waveform visualization and level metering. |
| **Frame Count (Audio or Manual)** | Computes frame count from audio length + FPS, or accepts manual values. |
| **Audio Frame Adjuster** | Utility for adjusting frame alignment from audio-driven timing. |
| **Mono to Stereo Converter** | Converts mono audio streams to stereo. |
| **Audio Loader Crawl** | Recursively scans folders and selects an audio file by seed/index logic. |
| **Image Loader Crawl** | Recursively scans folders and selects an image by seed/index logic. |
| **Load Image Base64** | Decodes Base64 image data into image output for pipeline use. |
| **Load Image Resize** | Loads an image and immediately resizes to target dimensions. |
| **Load Last Image** | Loads the most recently modified image from a target folder. |
| **Load Last Latent** | Loads the most recently modified latent (`.safetensors`) file. |
| **Load Last Video** | Loads the most recently modified video from a target folder. |
| **Text Loader Crawl** | Recursively scans folders and selects a text file by seed/index logic. |
| **Text Loader Crawl Batch** | Batch text loader that returns multiple prompts/strings. |
| **Video Loader Crawl** | Recursively scans folders and selects a video by seed/index logic. |

### Visual FX & Post-Processing
Image grading, cleanup, stylization, and effect nodes.

#### Save

| Node Name | Description |
| :--- | :--- |
| **Post-Process Suite** | Main all-in-one post node for grading, lens effects, sharpening, grain, and more. |
| **Save Audio With Path** | Saves audio output to explicit paths. |
| **Save Image With Path** | Saves images to explicit folder/subfolder paths. |
| **Save Latent With Path** | Saves latent tensors as `.safetensors` files to explicit paths. |
| **Save Text With Path** | Saves text output to explicit paths. |
| **Save Video With Path** | Saves videos to explicit folder/subfolder paths. |

#### Text

| Node Name | Description |
| :--- | :--- |
| **Add Settings and Prompt** | Renders prompt/settings overlays onto output images. |
| **Advanced String Replace** | Multi-rule find/replace utility for prompts and text. |
| **AutopromptProcessor** | Prompt preprocessor for replacement/prefix/suffix style operations. |
| **Remove Lines** | Removes selected lines from text blocks. |
| **Remove Trailing Comma** | Cleans trailing commas from prompt/text strings. |
| **String Batcher** | Combines multiple string inputs into a batched/merged output. |
| **String Line Counter** | Counts line totals in incoming text. |
| **String Splitter** | Splits large text into separate outputs by line or delimiter rules. |
| **Text Box line spot** | Picks/extracts a specific line from multi-line text. |

#### Utils

| Node Name | Description |
| :--- | :--- |
| **Boolean Transform** | Converts numeric/string values to boolean outputs. |
| **Mask Empty Float** | Outputs fallback float behavior based on mask emptiness. |
| **Mask Pass or Placeholder** | Passes a valid mask or emits a placeholder mask. |
| **Resolution** | Calculates output resolution from ratio and size targets. |
| **Sampler & Scheduler Crawler** | Iterates sampler/scheduler combos for comparison/testing. |
| **Sampler & Scheduler Selector** | Manual selector for sampler/scheduler combinations. |
| **Video Duration Calculator** | Calculates duration from FPS and frame counts. |
| **Fancy Note** | Graph note/sticky node for readable workflow annotation. |

#### FX

| Node Name | Description |
| :--- | :--- |
| **Advanced Bloom FX** | High-quality bloom with threshold and radius/intensity controls. |
| **Arcane Bloom FX** | Stylized bloom for glowing, dreamy highlights. |
| **Clarity FX** | Local contrast/detail enhancement without heavy global exposure shifts. |
| **Colourfulness FX** | Adaptive color intensity enhancement with clipping-safe behavior. |
| **Color Isolation FX** | Isolates selected hues while muting others. |
| **Contour FX** | Edge extraction/line-art style contour pass. |
| **Film Grain FX** | Procedural film grain with controllable texture strength. |
| **Lens Distort FX** | Barrel/pincushion style lens distortion simulation. |
| **Lens FX** | Combined lens treatment node (aberration/vignette style workflows). |
| **Post-Process Suite** | Main all-in-one post node for grading, lens effects, sharpening, grain, and more. |
| **Smart DeNoise FX** | Edge-aware denoise for cleanup with detail retention. |
| **Technicolor 2 FX** | Vintage two-strip Technicolor style color look. |
| **Chroma Key Overlay** | Basic chroma key/spill suppression and compositing helper. |
| **Batch Brightness Curve (U-Shape)** | Applies controlled U-shaped brightness remapping over image batches. |
| **Flux2Klein Seamless Tile** | Seamless tiling utility for texture generation workflows. |
| **Image Tile Checker** | Quick tile seam checking for pattern/texture validation. |

### FLUX / WAN / Sampling
Sampling, LoRA, and enhancement tools for FLUX and WAN pipelines.

#### LoRA

| Node Name | Description |
| :--- | :--- |
| **Flux LoRA Blocks Patcher** | Fine-grained block-level LoRA strength patching for FLUX models. |
| **Magic LoRA Loader** | Flexible multi-LoRA loader with optional wet/cap and block-weight controls. |
| **Wan Video Multi-LoRA Select** | Multi-LoRA selector/stack manager for WAN video workflows. |

#### Sampling

| Node Name | Description |
| :--- | :--- |
| **CLIP Text Encode FLUX Merged** | Prompt encoder for FLUX workflows with merged encoding behavior. |
| **Flux.1 Cnet Face Enhancement with Injection** | Face enhancement pipeline with detection, crop, and injection-based restore steps. |
| **Face Enhancement with Injection** | Face enhancement pipeline tuned for injection workflows. |
| **KSampler Batch** | Batch execution wrapper for repeated KSampler runs. |
| **Latent Noise Injection Sampler** | Injects controlled latent noise during sampling for texture/detail shaping. |
| **Pony Upscale Sampler with Injection & Tiling** | Pony-focused upscaler sampler with injection and tile support. |
| **WAN 2.2 Batch Sampler** | WAN video batch sampler for staged generation workflows. |
| **WAN 2.2 LoRA Compare Sampler** | Side-by-side WAN LoRA comparison helper. |
| **Flux Controlnet Sampler** | FLUX sampler variant with integrated ControlNet handling. |
| **Flux Controlnet Sampler with Injection** | ControlNet FLUX sampler with latent/noise injection controls. |
| **Upscale using model adv** | Model-based upscaler with advanced memory/tiling options. |

### Loaders & Savers
Directory crawl loaders and explicit path-based save nodes.

#### Image

| Node Name | Description |
| :--- | :--- |
| **Audio Loader Crawl** | Recursively scans folders and selects an audio file by seed/index logic. |
| **Image Loader Crawl** | Recursively scans folders and selects an image by seed/index logic. |
| **Video Loader Crawl** | Recursively scans folders and selects a video by seed/index logic. |
| **Text Loader Crawl** | Recursively scans folders and selects a text file by seed/index logic. |
| **Text Loader Crawl Batch** | Batch text loader that returns multiple prompts/strings. |
| **Load Last Image** | Loads the most recently modified image from a target folder. |
| **Load Last Video** | Loads the most recently modified video from a target folder. |
| **Load Last Latent** | Loads the most recently modified latent (`.safetensors`) file. |
| **Load Image Resize** | Loads an image and immediately resizes to target dimensions. |
| **Load Image Base64** | Decodes Base64 image data into image output for pipeline use. |
| **Save Image Base64** | Encodes image output to a Base64 string without writing to disk. |
| **Save Image With Path** | Saves images to explicit folder/subfolder paths. |
| **Save Video With Path** | Saves videos to explicit folder/subfolder paths. |
| **Save Audio With Path** | Saves audio output to explicit paths. |
| **Save Text With Path** | Saves text output to explicit paths. |
| **Save Latent With Path** | Saves latent tensors as `.safetensors` files to explicit paths. |
| **Batch Brightness Curve (U-Shape)** | Applies controlled U-shaped brightness remapping over image batches. |
| **Chroma Key Overlay** | Basic chroma key/spill suppression and compositing helper. |
| **Depth Anything Tensorrt Format** | Formats depth outputs for TensorRT-compatible depth workflows. |
| **Image Dimensions From Megapixels** | Computes dimensions from megapixel targets. |
| **Image Dimensions From MP alt** | Alternate megapixel-to-dimension calculator variant. |
| **Image Scale Range From MP** | Derives scaling ranges from megapixel constraints. |

### Logic & Utilities
Workflow helpers for strings, toggles, dimensions, scheduling, and graph organization.

#### Conditioning

| Node Name | Description |
| :--- | :--- |
| **String Batcher** | Combines multiple string inputs into a batched/merged output. |
| **String Splitter** | Splits large text into separate outputs by line or delimiter rules. |
| **String Line Counter** | Counts line totals in incoming text. |
| **Text Box line spot** | Picks/extracts a specific line from multi-line text. |
| **Remove Lines** | Removes selected lines from text blocks. |
| **Remove Trailing Comma** | Cleans trailing commas from prompt/text strings. |
| **Advanced String Replace** | Multi-rule find/replace utility for prompts and text. |
| **AutopromptProcessor** | Prompt preprocessor for replacement/prefix/suffix style operations. |
| **CLIP Text Encode FLUX Merged** | Prompt encoder for FLUX workflows with merged encoding behavior. |
| **Dynamic Prompt Scheduler** | Coordinates prompt changes over batches or frame sequences. |
| **File Batch Prompt Scheduler** | Builds prompt schedules from text file batches. |
| **Boolean Transform** | Converts numeric/string values to boolean outputs. |
| **Boolean Invert** | Inverts boolean values. |
| **Enable Latent** | Gate node that passes/blocks latent flow by boolean state. |
| **Mask Empty Float** | Outputs fallback float behavior based on mask emptiness. |
| **Mask Pass or Placeholder** | Passes a valid mask or emits a placeholder mask. |
| **Strength to Steps** | Converts denoise strength percentages to scheduler step ranges. |
| **Sampler & Scheduler Selector** | Manual selector for sampler/scheduler combinations. |
| **Sampler & Scheduler Crawler** | Iterates sampler/scheduler combos for comparison/testing. |
| **Smart ControlNet Apply** | Applies ControlNet with bypass logic when disabled/zeroed. |
| **Smart Preprocessor** | Preprocessor wrapper with smart passthrough behavior. |
| **Smart Style Model Apply DUAL** | Dual-input style application helper with internal caching behavior. |
| **Add Settings and Prompt** | Renders prompt/settings overlays onto output images. |

#### Video

| Node Name | Description |
| :--- | :--- |
| **Get First & Last Frame** | Extracts boundary frames from an image batch. |
| **Seamless Loop Blender** | Blends first/last frames for seamless loop generation. |
| **Percentage Crop Calculator** | Crop planning via percentage coordinates rather than absolute pixels. |
| **Quantize and Crop Image** | Crops/resizes to quantized bucket-friendly dimensions. |
| **Resolution** | Calculates output resolution from ratio and size targets. |
| **Image Dimensions From Megapixels** | Computes dimensions from megapixel targets. |
| **Image Dimensions From MP alt** | Alternate megapixel-to-dimension calculator variant. |
| **Image Scale Range From MP** | Derives scaling ranges from megapixel constraints. |
| **Video Duration Calculator** | Calculates duration from FPS and frame counts. |
| **Depth Anything Tensorrt Format** | Formats depth outputs for TensorRT-compatible depth workflows. |

#### Audio

| Node Name | Description |
| :--- | :--- |
| **Audio Frame Adjuster** | Utility for adjusting frame alignment from audio-driven timing. |
| **Frame Count (Audio or Manual)** | Computes frame count from audio length + FPS, or accepts manual values. |
| **Mono to Stereo Converter** | Converts mono audio streams to stereo. |
| **Parametric EQ** | Multi-band EQ with standard filter shapes for corrective and creative tone control. |
| **Preview Audio** | In-UI audio preview with waveform visualization and level metering. |
| **Tube Compressor** | Audio compressor with threshold/ratio/attack/release and musical shaping controls. |

#### Latent

| Node Name | Description |
| :--- | :--- |
| **Enable Latent** | Gate node that passes/blocks latent flow by boolean state. |
| **Reference Latent Batch** | Loads/manages latent reference batches for iterative workflows. |

#### Logic

| Node Name | Description |
| :--- | :--- |
| **Any Trigger** | Generic trigger/event pass-through utility node. |
| **Fancy Note** | Graph note/sticky node for readable workflow annotation. |
| **Fancy Timer Node** | Displays execution timing information in workflow runs. |
| **K** | Compact knob-style float control utility node. |
| **T** | Compact toggle-style boolean control utility node. |
| **Boolean Invert** | Inverts boolean values. |
| **Strength to Steps** | Converts denoise strength percentages to scheduler step ranges. |

#### Flux2

| Node Name | Description |
| :--- | :--- |
| **Flux2Klein Seamless Tile** | Seamless tiling utility for texture generation workflows. |
| **Tiny FLUX.2 VAE Loader** | Loads FLUX.2 Tiny AutoEncoder weights from `models/vae_approx/FLUX.2-Tiny-AutoEncoder/`. |
| **Tiny FLUX.2 VAE Encode** | Encodes `IMAGE` to FLUX.2-compatible latent format with fast Tiny VAE approximation. |
| **Tiny FLUX.2 VAE Decode** | Decodes FLUX.2-compatible latents back to `IMAGE` with fast Tiny VAE approximation. |

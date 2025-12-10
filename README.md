# CRT-Nodes for ComfyUI

This repository contains a comprehensive suite of custom nodes for ComfyUI, designed to provide advanced capabilities in Image Post-Processing, Audio Processing, FLUX/WanVideo workflows, and general utility management.

**Note:** Some nodes can require specific external libraries including `pedalboard` (audio), `ultralytics` (face detection), `color-matcher`, and `sageattention`.

---

## Node List

### Audio Processing
A collection of tools for manipulating audio, visualizing waveforms, and synchronizing audio with video frames.

| Node Name | Description |
| :--- | :--- |
| **Tube Compressor** | Advanced audio compressor with threshold, ratio, attack, release, and soft-clipping controls. Includes visual gain reduction metering. |
| **Parametric EQ (CRT)** | 8-band parametric equalizer with interactive frequency response visualization. Supports Bell, Shelf, Pass, and Notch filters. |
| **Preview Audio (CRT)** | Enhanced audio player with waveform visualization, VU meters, LUFS metering, and scrubbing. |
| **Audio/Manual Frame Count** | Calculates the number of frames required for a video based on audio duration and FPS, with optional quantization for WanVideo models. |
| **Video Duration Calculator** | Simple utility to calculate duration in seconds from FPS and frame count. |

### Visual FX & Post-Processing
Nodes focused on image grading, effects, and restoration. The suite includes a primary All-In-One node and standalone effects.

| Node Name | Description |
| :--- | :--- |
| **CRT Post-Process Suite** | Master node containing 12+ effect categories: Levels, Color Wheels, Temperature/Tint, Sharpening, Glow, Glare, Chromatic Aberration, Vignette, Radial Blur, Film Grain, and Lens Distortion. |
| **Advanced Bloom FX** | High-quality bloom effect with threshold, smoothing, radius, and intensity controls. |
| **Arcane Bloom FX** | Stylized bloom effect with saturation and exposure controls for a dreamlike look. |
| **Clarity FX** | Enhances local contrast and texture definition without affecting global brightness. |
| **Color Isolation FX** | Isolates specific hues while desaturating others, with adjustable falloff and background options. |
| **Colourfulness FX** | Adaptive saturation adjustment that prevents clipping using luma limits. |
| **Contour FX** | Edge detection node capable of generating line art or overlays. |
| **Film Grain FX** | Procedural film grain generator with shadow/highlight masking and color noise options. |
| **Lens Distort FX** | Simulates physical lens distortion including barrel/pincushion and chromatic separation. |
| **Lens FX** | Combined lens effects node offering chromatic aberration, vignette, and grain in a single pass. |
| **Prism FX** | Spectral dispersion effect that separates color channels radially. |
| **Smart DeNoise FX** | Reduces noise while preserving edges using sigma and threshold parameters. |
| **Technicolor 2 FX** | Emulates the 2-strip Technicolor process for vintage color grading. |
| **Chroma Key Overlay** | Basic green screen removal with spill suppression and background compositing. |

### FLUX & GenAI Tools
Workflows and samplers optimized for FLUX and WanVideo models.

| Node Name | Description |
| :--- | :--- |
| **FLUX All-In-One (CRT)** | Comprehensive node managing Models, LoRAs, ControlNet, Inpainting, Face Enhancement, and Upscaling in a tabbed interface. |
| **Flux Controlnet Sampler** | Specialized sampler for Flux models with integrated ControlNet support. |
| **Flux Controlnet Sampler (Injection)** | Extended sampler supporting noise injection for texture control. |
| **Flux Tiled Sampler (Advanced)** | Tiled sampling implementation for high-resolution generation with reduced VRAM usage. |
| **Flux LoRA Blocks Patcher** | Provides fine-grained control over Flux LoRA weights, allowing adjustment of specific Single and Double blocks via sliders. |
| **Flux Semantic Encoder** | CLIP/T5 text encoding node supporting "Semantic Shifting" and token injection. |
| **Wan Batch Sampler** | Specialized sampler for Wan2.1 video generation, managing High-Noise and Low-Noise model phases. |
| **Wan2.2 LoRA Compare Sampler** | Tool for comparing different LoRA configurations for Wan2.2 models side-by-side. |
| **Wan Video Multi-LoRA Select** | Interface for selecting and stacking multiple LoRAs for WanVideo workflows. |
| **Face Enhancement Pipeline** | Automated pipeline using YOLO models to detect, crop, upscale, and restore faces in images. |
| **Pony Face Enhancement Pipeline** | Variant of the face enhancement pipeline optimized for Pony-based models. |
| **Pony Upscale Sampler** | Upscaling sampler specialized for Pony models with noise injection capabilities. |
| **Latent Noise Injection Sampler** | Injects noise into latents at specific percentage steps during sampling to add detail or texture. |
| **KSampler Batch (CRT)** | Batch processing sampler supporting parallel and sequential modes. |
| **Upscale using model adv (CRT)** | Advanced model-based upscaler with tiling support and memory management. |

### Loaders & Crawlers
Smart loading nodes that can scan directories and select files based on seeds or specific criteria.

| Node Name | Description |
| :--- | :--- |
| **Audio Loader Crawl** | Recursively scans a folder for audio files and selects one based on a seed value. |
| **File Loader Crawl** | Recursively scans a folder for text files and selects one based on a seed. |
| **File Loader Crawl Batch** | Batch version of the file loader, returning multiple strings from a directory. |
| **Image Loader Crawl** | Recursively scans a folder for images and selects one based on a seed. |
| **Video Loader Crawl** | Recursively scans a folder for video files and selects one based on a seed. |
| **Load Last Image** | Automatically loads the most recently modified image file from a specified directory. |
| **Load Last Video** | Automatically loads the most recently modified video file from a specified directory. |
| **Load Last Latent** | Automatically loads the most recently modified .safetensors latent file from a directory. |
| **Load Video (V-Captioning)** | Efficient video loader designed for captioning workflows, supporting frame skipping and limits. |
| **Load & Resize Image** | Loads an image and immediately resizes it to a target dimension or multiple. |
| **LoRA Loader (String)** | LoRA loader that outputs the LoRA name as a string for logging or logic purposes. |

### Savers
Nodes for saving various data types with customizable paths and naming conventions.

| Node Name | Description |
| :--- | :--- |
| **Save Image With Path** | Saves images to a specific folder and subfolder structure. |
| **Save Video With Path** | Saves video files to a specific folder using FFmpeg. |
| **Save Audio With Path** | Saves audio waveforms to a specific folder as WAV files. |
| **Save Text With Path** | Saves text strings to a specific folder as TXT files. |
| **Save Latent With Path** | Saves latents to a specific folder as .safetensors files. |

### Logic & Utilities
Helper nodes for string manipulation, math, and workflow logic.

| Node Name | Description |
| :--- | :--- |
| **CRT String Batcher** | Combines multiple string inputs into a single batch or concatenated string. |
| **CRT String Splitter** | Splits a text block into multiple string outputs based on delimiters or newlines. |
| **CRT Dynamic Prompt Scheduler** | Coordinate prompts and images for batch generation sequences. |
| **File Batch Prompt Scheduler** | Reads text files from a folder to create a scheduled batch of prompts. |
| **Autoprompt Processor** | Processes prompt strings with search/replace functionality and prefixing. |
| **Advanced String Replace** | Performs multiple find/replace operations on a string with exclusion lists. |
| **Remove Trailing Comma** | Cleans up prompt strings by removing trailing commas. |
| **Smart Preprocessor** | ControlNet preprocessor wrapper that bypasses processing if strength is set to 0. |
| **Resolution Calculator** | Calculates width and height based on aspect ratio and target megapixel count or side length. |
| **Image Dimensions from MP** | Calculates target dimensions from a total megapixel count. |
| **Percentage Crop Calculator** | Crops an image based on percentage values rather than pixels. |
| **Quantize and Crop Image** | Crops and resizes images to standard bucket sizes (e.g., 512, 768, 1024). |
| **Get First & Last Frame** | Extracts the start and end frames from a batch of images. |
| **Seamless Loop Blender** | Blends the start and end of an image batch to create a seamless loop. |
| **Simple Toggle** | Boolean switch for controlling workflow logic. |
| **Simple Knob** | Float value controller with a knob UI. |
| **Boolean Invert** | Inverts a boolean value (True -> False). |
| **Boolean Transform** | Converts numbers or strings into boolean values. |
| **Strength to Steps** | Calculates start/end steps based on a percentage strength value. |
| **Mask Empty Float** | Logic gate that outputs a float value depending on whether a mask is empty. |
| **Mask Pass or Placeholder** | Passes a mask if valid, or generates a placeholder mask if missing. |
| **Enable Latent** | Gate node to pass or block a latent based on a boolean toggle. |
| **Empty Context** | Generates a null context object for WanVideo workflows. |
| **Toggle LoRA Blocks (L1/L2)** | Helper nodes for generating block selection strings for LoRA patching. |
| **Sampler Scheduler Selector** | Simple dropdown selector for sampler and scheduler names. |
| **Sampler & Scheduler Crawler** | Iterates through sampler/scheduler combinations based on a seed for testing. |
| **Smart Style Model Apply** | Applies style models with caching and dual-input support. |
| **Smart ControlNet Apply** | Applies ControlNet with intelligent bypassing if strength is zero. |
| **Clear Dual Style Cache** | Utility to clear the cache of the Smart Style Model node. |
| **Add Text Overlay** | Adds text/settings overlays to images for comparison grids. |
| **Fancy Note** | A sticky note node for the graph with customizable colors and text size. |
| **Fancy Timer** | Displays execution time for workflows. |

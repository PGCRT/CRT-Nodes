
FLUX SUPER
![SC](https://github.com/user-attachments/assets/6017ce56-b5eb-40ce-a9f9-eb63e3b0383e)

PONY SUPER
![SC](https://github.com/user-attachments/assets/81fa2a70-718b-4fde-8c20-ca690c147cd9)


1. Resolution

This is a utility node that standardizes image dimensions. By providing a target length for the longest side and selecting from a list of common aspect ratios (like 3:2 or 16:9), the node calculates and provides the corresponding width and height, making it easy to maintain consistent image proportions.

2. Simple Toggle & K (Simple Knob)

These are compact UI utility nodes.

Simple Toggle provides a simple True/False value, visualized as an intuitive on/off switch in the interface.

K (Simple Knob) offers a float value through a space-saving rotary knob widget, serving as a compact alternative to a full slider.

3. SamplerSchedulerSelector

A convenience node that centralizes the selection of ComfyUI's samplers and schedulers. It provides two dropdown menus containing all available options and outputs the selected names as strings for use in other sampler nodes.

4. Boolean Transform Node

This node acts as a simple logic gate, converting an input string into a boolean and an integer. It interprets any non-zero number as True (and 1), while zero or non-numeric strings become False (and 0).

5. FancyNoteNode

A purely visual tool for workflow annotation, this node displays highly customizable, stylized text directly on the graph. Users can adjust font size, colors, and glow effects to create clear, visually appealing notes for complex workflows.

![Capture d'écran 2025-06-19 101544](https://github.com/user-attachments/assets/093c1f05-5c98-465e-ba1f-516ab168b9bc)
![Capture d'écran 2025-06-19 101528](https://github.com/user-attachments/assets/0ef5e93a-8200-4a4a-a5c3-e59c0506b43a)

6. LoraLoaderStr

This is an enhanced LoRA loader with a convenient on/off switch. It applies a selected LoRA to the model and CLIP and also generates a descriptive string that can optionally include the strength settings, which is useful for tracking metadata.

7. RemoveTrailingCommaNode

A specific string utility node whose sole function is to remove a single comma from the very end of a string, if present. This is often used to clean up dynamically generated lists of keywords or block weights.

8. Toggle Lora Unet Blocks L1 / L2

These two nodes offer incredibly granular control over LoRA application. Each provides 38 boolean toggles, corresponding to specific blocks in the UNet architecture. Activating toggles generates a comma-separated string that can be fed into advanced LoRA nodes to apply weights to very specific parts of the model's structure. L1 targets the first linear layer in each block, while L2 targets the second.

9. Video Duration Calculator

A straightforward calculator that determines the total duration of a video clip in seconds, based on its frames per second (FPS) and total frame count.

10. Smart Style Model Apply & Clear Cache

This pair of nodes provides an efficient way to apply style models (like those used with FLUX).

The main node's key feature is its ability to bypass all computation if the style strength is zero, saving significant resources. It also includes an optional cache to speed up repeated generations with the same settings.

The accompanying Clear Style Model Cache node provides a simple way to manually clear this cache.

11. Flux Tiled Sampler (Advanced)

A powerful, self-contained tiled sampler designed specifically for high-resolution generation with FLUX models. It works by dividing an image or latent into a grid of overlapping tiles, running the sampling process on each tile individually, and then seamlessly stitching them back together using a blended mask. This technique allows for creating images at resolutions that would normally cause out-of-memory errors.

12. MaskEmptyFloatNode

A conditional utility node that outputs a float value based on a mask's content. If the input mask is completely empty (all black), it outputs 0.0; otherwise, it outputs a user-specified value.

13. FluxSemanticEncoder https://www.youtube.com/watch?v=it35sOMFxho

An extremely advanced text conditioning node for FLUX models that offers powerful, creative control over prompts through several modes:

Positional EQ: Functions like an audio EQ for prompts, allowing you to boost or cut the influence of tokens based on their position in the sentence using parametric bands (center, gain, Q-factor).

Semantic Shift: Uses a pre-computed "map" of related words to replace tokens with semantically similar ones. A distance parameter controls how far the meaning can drift.

Token Injection: Injects new, thematically relevant tokens (from the map file) into the prompt, either by replacing existing tokens or adding new ones.

Manual Keywords: Supports standard (word:scale) syntax for direct emphasis control.

![Capture d'écran 2025-06-19 105721](https://github.com/user-attachments/assets/9e68fe74-b036-4842-8451-df892809c4e6)

14. MaskPassOrPlaceholder

A simple workflow utility that ensures a node requiring a mask input doesn't fail. If a mask is connected, it passes it through. If not, it generates a small, default placeholder mask.

15. FileLoaderCrawl & ImageLoaderCrawl

These nodes deterministically load a random file from a folder using a seed for reproducible selection. They can optionally crawl through subfolders.

FileLoaderCrawl is designed for text files.

ImageLoaderCrawl is designed for all common image formats.

16. Flux Controlnet Sampler

A custom pipeline that integrates ControlNet guidance with a FLUX model. It takes a source, upscales it, applies ControlNet, runs the sampler, and can optionally perform color matching between the final image and the original ControlNet guide image to maintain color consistency.

17. Flux LoRA Blocks Patcher https://www.youtube.com/watch?v=uHIoEnLX38Q

This node allows for precise, per-block tuning of an applied LoRA's effect on the FLUX architecture. It exposes individual strength sliders for each of the 38 single_blocks and 19 double_blocks, enabling deep customization beyond a single global strength value.

![Capture d'écran 2025-06-19 100858](https://github.com/user-attachments/assets/e876c0e6-6e80-4807-bb0f-9a07e9a79fdf)

18. CRT Post-Process Suite https://www.youtube.com/watch?v=vZ8tgiSDf9Y

A massive, all-in-one suite for image post-processing. It consolidates a wide array of effects into a single node, where each effect group can be toggled on or off. Effects include:

Color Correction: Levels, Color Wheels (Lift/Gamma/Gain), Temperature & Tint.

Image Effects: Sharpening, Glow (small and large), Glare/Flares (star, anamorphic), and Film Grain.

Lens Simulation: Chromatic Aberration, Vignetting, Radial Blur (spin/zoom), and Barrel/Pincushion Distortion.

![Capture d'écran 2025-06-19 102131](https://github.com/user-attachments/assets/0aa1b0d9-bc55-4289-97b4-41099c8a0509)

19. Face Enhancement Pipeline

A comprehensive workflow-in-a-node for detecting and enhancing faces. Its pipeline involves upscaling the source image, using YOLO models to detect and crop a face, upscaling the face crop further, performing a guided diffusion pass on the face, color matching it, and then seamlessly compositing it back into the original image.

20. Upscale using model adv (CRT)

An advanced upscaler with superior memory management. Its key features are an output multiplier to achieve any target size regardless of the model's native scale, and a tiling system to process very large images without running out of VRAM. It also supports precision control and model offloading.

21. Flux Controlnet Sampler with Injection

An enhanced version of the FluxControlnetSampler that adds a "noise injection" step. Partway through the sampling process, it adds a new layer of noise (from a different seed) to the latent, which can introduce new details and creative variations before the final refinement steps.

22. Simple FLUX Shift

A user-friendly node for setting the shift parameter in FLUX models. Instead of manual calculation, it offers simple intensity presets ("-3" to "+3") and automatically determines the correct shift value based on the input image's resolution, simplifying a complex but important setting.

23. Latent Noise Injection Sampler

A generic KSampler wrapper that implements the two-stage noise injection technique. It can be used with any model to add creative chaos or detail midway through the generation process by running for a percentage of steps, adding noise, and then completing the remaining steps.

24. Face Enhancement Pipeline with Injection

This node combines the full FaceEnhancementPipeline with the noise injection technique. It adds the two-stage injection process to the face re-diffusion step, aiming to generate more detailed and varied facial textures.

25. Smart ControlNet Apply

A highly efficient node that integrates ControlNet preprocessing and application. It can automatically run a preprocessor (like Canny or Depth) on an image and then apply the ControlNet. Its "smart" features are bypassing all work if strength is zero and caching the preprocessed image to avoid re-computing it on subsequent runs.

26. Pony Upscale Sampler with Injection, Tiling & Color Matching

A sophisticated, all-in-one upscaling and resampling pipeline. It can optionally upscale an image (with an AI model or simple resize), then re-sample it using either a full or tiled approach. Both sampling methods support the two-stage noise injection technique. Finally, it can color-match the result to the original image.

27. Pony Face Enhancement Pipeline with Injection

The Pony-specific version of the Face Enhancement pipeline. It follows the same detect-crop-enhance-composite logic but also integrates the two-stage noise injection technique during the face re-diffusion step, tailored for Pony models.

28. CLIPTextEncodeFluxMerged

A simplified text encoder for FLUX models. It takes a single prompt and guidance value, then automatically tokenizes and encodes it for both the required CLIP-L and T5-XXL encoders, reducing workflow clutter.

29. FLUX All-In-One WIP

A monolithic "mega-node" that encapsulates an entire, complex generation workflow for FLUX models. It integrates model loading, conditioning, LoRA stacking, multi-pass generation (with upscaling and optional ControlNet), face enhancement, and a full suite of post-processing effects into a single, highly configurable unit. It also includes advanced performance optimizations like SageAttention patching and a custom TeaCache system to accelerate the sampling process.

![Capture d'écran 2025-06-19 101420](https://github.com/user-attachments/assets/b1c49938-5229-4472-a20e-05621a482d55)

And more (this list is already too long)..

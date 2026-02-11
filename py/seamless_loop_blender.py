import torch


class SeamlessLoopBlender:
    """
    A node to create a seamless loop. It follows the "boomerang" blending
    method where the end of the sequence is blended with a reversed copy
    of the beginning, ensuring the last frame is identical to the first.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend_frames": ("INT", {"default": 12, "min": 1, "max": 256, "step": 1}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images_for_loop"
    CATEGORY = "CRT/Video"

    def blend_images_for_loop(self, images: torch.Tensor, blend_frames: int, enabled: bool):
        if not enabled:
            return (images,)

        num_frames = images.shape[0]

        # We need at least blend_frames + 1 frames to have a section to blend
        # and a frame preceding it.
        if blend_frames <= 0 or num_frames <= blend_frames:
            print(
                f"[SeamlessLoopBlender] Warning: Not enough frames ({num_frames}) to perform a blend of {blend_frames} frames. Skipping."
            )
            return (images,)

        print("[SeamlessLoopBlender] Applying boomerang blend. Final frame will match the first frame.")

        # --- This implements the logic you described ---

        # 1. The frames at the end of the video that we will be modifying.
        # This is the "tail" that will be faded out.
        tail_section = images[num_frames - blend_frames :]

        # 2. The frames at the beginning of the video that we will fade in.
        head_section = images[:blend_frames]

        # 3. As you instructed, we "invert" the head section. This is the crucial step.
        # This aligns the first frame of the video to be the last frame of the overlay.
        # Example: [frame_0, frame_1, ..., frame_9] becomes [frame_9, frame_8, ..., frame_0]
        reversed_head_section = torch.flip(head_section, dims=[0])

        # 4. Create the alpha weights for the transition.
        # This goes from transparent (0.0) to fully opaque (1.0), as you said.
        alphas = torch.linspace(0.0, 1.0, steps=blend_frames, device=images.device, dtype=images.dtype)

        # Reshape for broadcasting with the image tensors.
        alphas = alphas.view(-1, 1, 1, 1)

        # 5. Perform the blend (linear interpolation).
        # We fade FROM the `tail_section` TO the `reversed_head_section`.
        # At alpha=0.0, the frame is 100% `tail_section`.
        # At alpha=1.0, the frame is 100% `reversed_head_section`.
        blended_section = torch.lerp(tail_section, reversed_head_section, alphas)

        # The last frame of `blended_section` is now calculated with alpha=1.0,
        # making it identical to the last frame of `reversed_head_section`,
        # which is `images[0]`. The loop is now perfect.

        # Clone the original tensor and replace the tail with our new blended section.
        output_images = images.clone()
        output_images[num_frames - blend_frames :] = blended_section

        return (output_images,)

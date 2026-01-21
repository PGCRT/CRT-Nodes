from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import comfy
from comfy_api.latest import io
import math

class LTXVAutoGuideSpacing(LTXVAddGuide):
    """
    Automatically distributes guide frames evenly across an image batch
    with easing curve support for strength interpolation.
    """

    EASING_FUNCTIONS = {
        "linear": lambda t: t,
        "ease_in_quad": lambda t: t * t,
        "ease_out_quad": lambda t: t * (2 - t),
        "ease_in_out_quad": lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t,
        "ease_in_cubic": lambda t: t * t * t,
        "ease_out_cubic": lambda t: (t - 1) * (t - 1) * (t - 1) + 1,
        "ease_in_out_cubic": lambda t: 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
        "ease_in_sine": lambda t: 1 - math.cos((t * math.pi) / 2),
        "ease_out_sine": lambda t: math.sin((t * math.pi) / 2),
        "ease_in_out_sine": lambda t: -(math.cos(math.pi * t) - 1) / 2,
    }

    @classmethod
    def define_schema(cls):
        easing_options = list(cls.EASING_FUNCTIONS.keys())
        
        return io.Schema(
            node_id="LTXVAutoGuideSpacing",
            category="CRT/Conditioning",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("image_batch", tooltip="Batch of images to use as guides"),
                io.Int.Input(
                    "num_guides",
                    default=5,
                    min=1,
                    max=50,
                    tooltip="Number of guide frames to evenly select from the batch",
                ),
                io.Int.Input(
                    "max_frames",
                    default=0,
                    min=0,
                    max=1000,
                    tooltip="Total frame count of the final video. If 0, attempts to auto-detect from latent.",
                ),
                io.Float.Input(
                    "first_frame_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength for the first frame (0 = disabled, 1 = full strength)",
                ),
                io.Float.Input(
                    "last_frame_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength for the last frame (0 = disabled, 1 = full strength)",
                ),
                io.Float.Input(
                    "min_strength",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Minimum strength for middle frames (start of easing curve)",
                ),
                io.Float.Input(
                    "max_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Maximum strength for middle frames (end of easing curve)",
                ),
                io.Combo.Input(
                    "easing_curve",
                    options=easing_options,
                    default="linear",
                    tooltip="Easing curve for strength interpolation across middle frames",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def calculate_strength(cls, index, total_middle_frames, min_strength, max_strength, easing_name):
        if total_middle_frames <= 1:
            return max_strength
        t = index / (total_middle_frames - 1)
        easing_func = cls.EASING_FUNCTIONS.get(easing_name, cls.EASING_FUNCTIONS["linear"])
        eased_t = easing_func(t)
        return min_strength + (max_strength - min_strength) * eased_t

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        image_batch,
        num_guides,
        max_frames,
        first_frame_strength,
        last_frame_strength,
        min_strength,
        max_strength,
        easing_curve,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        # Standard LTXV temporal compression is roughly 8x, but we rely on max_frames for spacing
        TEMPORAL_COMPRESSION = 8

        _, _, actual_latent_length, latent_height, latent_width = latent_image.shape
        batch_size = image_batch.shape[0]

        # 1. Determine the timeline length in VIDEO FRAMES
        if max_frames > 0:
            timeline_frames = max_frames
        else:
            # Fallback: Convert latent length back to approximate video frames
            timeline_frames = actual_latent_length * TEMPORAL_COMPRESSION
            print(f"[LTXVAutoGuideSpacing] max_frames=0. Inferred timeline: {timeline_frames} frames from latent length {actual_latent_length}.")

        if batch_size == 0:
            return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

        # Cap num_guides to batch_size
        num_guides = min(num_guides, batch_size)

        # 2. Calculate mappings: (Batch Index) -> (Target Video Frame Index)
        frame_mappings = []
        
        if num_guides == 1:
            frame_mappings.append((0, 0, "single"))
        elif num_guides == 2:
            frame_mappings.append((0, 0, "first"))
            # For the last frame, we usually want the very end of the video sequence
            frame_mappings.append((batch_size - 1, timeline_frames - 1, "last"))
        else:
            for i in range(num_guides):
                # Evenly space the batch index
                batch_idx = int((i / (num_guides - 1)) * (batch_size - 1))
                
                # Evenly space the VIDEO frame index
                target_frame_idx = int((i / (num_guides - 1)) * (timeline_frames - 1))
                
                if i == 0:
                    frame_type = "first"
                elif i == num_guides - 1:
                    frame_type = "last"
                else:
                    frame_type = "middle"
                
                frame_mappings.append((batch_idx, target_frame_idx, frame_type))

        processed_frames = []
        middle_frame_counter = 0
        total_middle_frames = sum(1 for _, _, ftype in frame_mappings if ftype == "middle")

        for batch_idx, target_frame_idx, frame_type in frame_mappings:
            if frame_type == "first":
                strength = first_frame_strength
            elif frame_type == "last":
                strength = last_frame_strength
            elif frame_type == "single":
                strength = max_strength
            else:
                strength = cls.calculate_strength(
                    middle_frame_counter,
                    total_middle_frames,
                    min_strength,
                    max_strength,
                    easing_curve
                )
                middle_frame_counter += 1
            
            if strength == 0.0:
                continue

            # Encode the image
            img = image_batch[batch_idx].unsqueeze(0)
            image_encoded, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            
            # FIX: Pass the VIDEO Frame Index (target_frame_idx) directly to get_latent_index.
            # The base class handles the division/compression.
            # We pass actual_latent_length to allow it to check bounds against the real tensor.
            frame_idx, latent_idx = cls.get_latent_index(
                positive, actual_latent_length, len(image_encoded), target_frame_idx, scale_factors
            )
            
            # If the calculated latent index is out of bounds (because actual latent is shorter than max_frames implies),
            # we warn and skip to prevent a crash.
            if latent_idx + t.shape[2] > actual_latent_length:
                print(f"[LTXVAutoGuideSpacing] SKIP: Guide for frame {target_frame_idx} maps to latent index {latent_idx}, which exceeds input latent size ({actual_latent_length}).")
                continue
            
            # Append keyframe
            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )
            
            processed_frames.append({
                'batch_idx': batch_idx,
                'target_frame': target_frame_idx,
                'latent_idx': latent_idx,
                'strength': strength,
                'frame_type': frame_type
            })

        print(f"[LTXVAutoGuideSpacing] Processed {len(processed_frames)} guide frames (Max Video Frames: {timeline_frames}).")
        for info in processed_frames:
            print(f"  {info['frame_type'].upper()}: Batch {info['batch_idx']} -> Frame {info['target_frame']} (Latent {info['latent_idx']}) | Str: {info['strength']:.2f}")

        return io.NodeOutput(
            positive, 
            negative, 
            {"samples": latent_image, "noise_mask": noise_mask}
        )
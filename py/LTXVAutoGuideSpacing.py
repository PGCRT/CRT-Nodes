from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import comfy
from comfy_api.latest import io
import math

class LTXVAutoGuideSpacing(LTXVAddGuide):
    """
    Automatically distributes guide frames evenly across an image batch.
    Applies a symmetrical "U-Shape" strength curve (High -> Low -> High) 
    controlled by a Q-factor.
    """

    @classmethod
    def define_schema(cls):
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
                    tooltip="Strength for the very first frame (Start of video)",
                ),
                io.Float.Input(
                    "last_frame_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength for the very last frame (End of video)",
                ),
                io.Float.Input(
                    "max_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength at the edges of the middle sequence (Connecting to first/last)",
                ),
                io.Float.Input(
                    "min_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength at the center of the video (Bottom of the U-shape)",
                ),
                io.Float.Input(
                    "q_factor",
                    default=2.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    tooltip="Curve Shape. 1.0=Linear(V), 2.0=Parabolic(U). Lower values = Narrow Dip. Higher values = Wide Dip.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

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
        max_strength,
        min_strength,
        q_factor,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        # Standard LTXV temporal compression
        TEMPORAL_COMPRESSION = 8

        _, _, actual_latent_length, latent_height, latent_width = latent_image.shape
        batch_size = image_batch.shape[0]

        # 1. Determine the timeline length in VIDEO FRAMES
        if max_frames > 0:
            timeline_frames = max_frames
        else:
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
                # --- U-SHAPE CALCULATION ---
                # 1. Normalize position in middle sequence (0.0 to 1.0)
                if total_middle_frames <= 1:
                    t = 0.5
                else:
                    t = middle_frame_counter / (total_middle_frames - 1)
                
                # 2. Calculate Distance from Center (1.0 = Edge, 0.0 = Center)
                # t=0.0 -> dist=1.0 | t=0.5 -> dist=0.0 | t=1.0 -> dist=1.0
                dist_from_center = abs(t - 0.5) * 2
                
                # 3. Apply Q-Factor (Power Curve)
                # Curve = Distance ^ Q
                curve_value = math.pow(dist_from_center, q_factor)
                
                # 4. Map to Strength
                # If curve_value is 1 (Edges), we get max_strength
                # If curve_value is 0 (Center), we get min_strength
                strength = min_strength + (max_strength - min_strength) * curve_value
                
                middle_frame_counter += 1
            
            # Skip if strength is effectively zero
            if strength < 0.001:
                processed_frames.append({
                    'batch_idx': batch_idx, 'target_frame': target_frame_idx, 'strength': 0.0, 'type': 'SKIPPED'
                })
                continue

            # Encode the image
            img = image_batch[batch_idx].unsqueeze(0)
            image_encoded, t_enc = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            
            # Get Latent Index
            frame_idx, latent_idx = cls.get_latent_index(
                positive, actual_latent_length, len(image_encoded), target_frame_idx, scale_factors
            )
            
            # Safety Check
            if latent_idx + t_enc.shape[2] > actual_latent_length:
                print(f"[LTXVAutoGuideSpacing] SKIP: Frame {target_frame_idx} (Latent {latent_idx}) exceeds bounds.")
                continue
            
            # Append keyframe
            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t_enc,
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

        print(f"[LTXVAutoGuideSpacing] Processed {len(processed_frames)} guide frames. Mode: U-Shape (Q={q_factor})")
        # Optional: Print summary of first few, middle, and last for verification
        if len(processed_frames) > 0:
            print(f"  Start: Str {processed_frames[0]['strength']:.2f}")
            mid = len(processed_frames)//2
            print(f"  Mid:   Str {processed_frames[mid]['strength']:.2f}")
            print(f"  End:   Str {processed_frames[-1]['strength']:.2f}")

        return io.NodeOutput(
            positive, 
            negative, 
            {"samples": latent_image, "noise_mask": noise_mask}
        )
        
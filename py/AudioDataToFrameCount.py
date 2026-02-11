import torch


class AudioOrManualFrameCount:
    """
    A custom node for ComfyUI that calculates frame count from audio
    OR provides a manual frame count, with an option to quantize
    the output for WAN video models (4n + 1 frames).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
                "bypass": ("BOOLEAN", {"default": False}),
                "manual_frame_count": ("INT", {"default": 120, "min": 1, "max": 99999, "step": 1}),
                "quantize_for_wan": ("BOOLEAN", {"default": False}),  # New input
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_count",)
    FUNCTION = "get_frame_count"
    CATEGORY = "CRT/Audio"

    def get_frame_count(self, audio, fps, bypass, manual_frame_count, quantize_for_wan):
        frame_count = 0

        # Step 1: Determine the base frame count from manual input or audio
        if bypass:
            frame_count = manual_frame_count
        else:
            # --- Audio calculation logic ---
            if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
                print("Warning (Frame Count Node): Invalid audio data format. Returning 0 frames.")
                return (0,)

            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")

            if not isinstance(waveform, torch.Tensor) or not isinstance(sample_rate, int):
                print("Warning (Frame Count Node): Invalid audio data types. Returning 0 frames.")
                return (0,)

            if waveform.numel() == 0 or sample_rate == 0:
                print("Warning (Frame Count Node): Empty audio data or zero sample rate. Returning 0 frames.")
                return (0,)

            num_samples = waveform.shape[-1]
            duration_seconds = num_samples / sample_rate
            calculated_frames = int(duration_seconds * fps)

            # Ensure at least 1 frame if there's any audio duration
            if calculated_frames == 0 and duration_seconds > 0:
                calculated_frames = 1

            frame_count = calculated_frames

        # Step 2: If quantization is enabled, adjust the frame count
        if quantize_for_wan and frame_count > 0:
            # The target format is y = 4n + 1.
            # This logic finds the smallest valid number that is >= the current frame_count.

            # Subtract 1, find the remainder when divided by 4
            remainder = (frame_count - 1) % 4

            if remainder != 0:
                # If there's a remainder, add the difference to get to the next multiple of 4
                frame_count += 4 - remainder

            # WAN models often expect a minimum of 5 frames (where n=1).
            # If the calculation results in 1, bump it to 5.
            if frame_count < 5:
                frame_count = 5

        return (frame_count,)

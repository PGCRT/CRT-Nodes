# /custom_nodes/AudioCompressor.py

import torch
import numpy as np

# Try to import pedalboard, and gracefully handle if it fails
PEDALBOARD_AVAILABLE = False
pedalboard = None
PEDALBOARD_ERROR = None

try:
    import pedalboard

    PEDALBOARD_AVAILABLE = True
except ImportError as e:
    PEDALBOARD_ERROR = "pedalboard library not installed. Install with: pip install pedalboard"
except Exception as e:
    # Catch AVX2 and other runtime errors
    PEDALBOARD_ERROR = f"pedalboard cannot be loaded (CPU may lack AVX2 support): {str(e)}"

if not PEDALBOARD_AVAILABLE:
    print("------------------------------------------------------------------")
    print("AudioCompressor node disabled:")
    print(PEDALBOARD_ERROR)
    print("------------------------------------------------------------------")


class AudioCompressor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        if not PEDALBOARD_AVAILABLE:
            return {
                "required": {
                    "error": (
                        "STRING",
                        {"default": f"AudioCompressor unavailable:\n{PEDALBOARD_ERROR}", "multiline": True},
                    )
                }
            }

        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0, "step": 0.1}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 200.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 2000.0, "step": 1.0}),
                "warmth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix_wet": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "match_input_peak": (["OFF", "ON"], {"default": "OFF"}),
                "soft_clipper_toggle": (["OFF", "ON"], {"default": "OFF"}),
                "clipper_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "gr_meter_data")
    FUNCTION = "process"
    CATEGORY = "CRT/Audio"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Force re-evaluation if pedalboard is not available
        if not PEDALBOARD_AVAILABLE:
            return float("nan")
        return ""

    def measure_latency(self, board, sample_rate, num_channels):
        impulse = np.zeros((num_channels, sample_rate), dtype=np.float32)
        impulse[0, 0] = 1.0
        if num_channels > 1:
            impulse[1, 0] = 1.0

        processed_impulse = board(impulse, sample_rate)
        return np.argmax(np.abs(processed_impulse[0]))

    def process(
        self,
        audio=None,
        threshold_db=-20.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=100.0,
        warmth=0.1,
        mix_wet=1.0,
        makeup_gain_db=0.0,
        match_input_peak="OFF",
        soft_clipper_toggle="OFF",
        clipper_threshold=1.0,
        prompt=None,
        extra_pnginfo=None,
        **kwargs,
    ):

        if not PEDALBOARD_AVAILABLE:
            error_msg = f"AudioCompressor is unavailable: {PEDALBOARD_ERROR}"
            print(f"[AudioCompressor] {error_msg}")
            # Return a dummy output to prevent workflow errors
            if audio is not None:
                return (audio, "{}")
            else:
                # Create silent audio as fallback
                silent = torch.zeros((1, 2, 44100), dtype=torch.float32)
                return ({"waveform": silent, "sample_rate": 44100}, "{}")

        audio_tensor, sample_rate = audio['waveform'], audio['sample_rate']

        audio_np = audio_tensor.squeeze(0).cpu().numpy()
        num_channels = audio_np.shape[0] if audio_np.ndim > 1 else 1

        mono_for_peak_detection = np.mean(audio_np, axis=0) if num_channels > 1 else audio_np
        original_peak = np.max(np.abs(mono_for_peak_detection))
        if original_peak == 0:
            return (audio, "{}")

        normalized_audio = audio_np / original_peak

        wet_board = pedalboard.Pedalboard()
        if warmth > 0:
            wet_board.append(pedalboard.Distortion(drive_db=warmth * 12.0))
        compressor = pedalboard.Compressor(
            threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms
        )
        wet_board.append(compressor)

        gr_meter_data = self.get_gain_reduction_envelope(mono_for_peak_detection, sample_rate, compressor)
        gr_data_json = str(gr_meter_data.tolist())

        latency_samples = self.measure_latency(wet_board, sample_rate, num_channels)

        wet_signal = wet_board(normalized_audio, sample_rate)

        aligned_wet = wet_signal[:, latency_samples:]
        aligned_dry = normalized_audio[:, :-latency_samples] if latency_samples > 0 else normalized_audio

        mixed_signal = (aligned_wet * mix_wet) + (aligned_dry * (1.0 - mix_wet))

        if match_input_peak == "ON":
            current_peak = np.max(np.abs(mixed_signal))
            if current_peak > 1e-9:
                mixed_signal *= 1.0 / current_peak
        else:
            rms_in = np.sqrt(np.mean(aligned_dry**2))
            rms_out = np.sqrt(np.mean(mixed_signal**2))
            if rms_out > 1e-9:
                mixed_signal *= rms_in / rms_out

        mixed_signal *= 10 ** (makeup_gain_db / 20.0)

        if soft_clipper_toggle == "ON":
            clipper_db = -12.0 * (1.0 - clipper_threshold) / 0.9
            mixed_signal = pedalboard.Clipping(threshold_db=clipper_db)(mixed_signal, sample_rate)

        final_signal = np.clip(mixed_signal, -(10 ** (-1.0 / 20.0)), (10 ** (-1.0 / 20.0)))

        padding = np.zeros((num_channels, latency_samples))
        final_signal_padded = np.concatenate((final_signal, padding), axis=1)

        output_torch = torch.from_numpy(final_signal_padded.astype(np.float32))
        final_output_tensor = output_torch.unsqueeze(0).to(audio_tensor.device)

        return ({"waveform": final_output_tensor, "sample_rate": sample_rate}, gr_data_json)

    def get_gain_reduction_envelope(self, signal, sr, compressor):
        downsample_factor = 512
        num_blocks = len(signal) // downsample_factor
        if num_blocks == 0:
            return np.array([0.0])
        envelope_signal = np.max(
            np.abs(signal[: num_blocks * downsample_factor].reshape(num_blocks, downsample_factor)), axis=1
        )
        gr_envelope = np.zeros_like(envelope_signal)
        for i, sample_peak in enumerate(envelope_signal):
            level_db = 20 * np.log10(sample_peak + 1e-9)
            over_threshold = level_db - compressor.threshold_db
            gr_db = 0.0
            if over_threshold > 0:
                gr_db = (1.0 / compressor.ratio - 1.0) * over_threshold
            gr_envelope[i] = min(abs(gr_db) / 30.0, 1.0)
        return gr_envelope


NODE_CLASS_MAPPINGS = {"AudioCompressor": AudioCompressor}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioCompressor": "Tube Compressor"}

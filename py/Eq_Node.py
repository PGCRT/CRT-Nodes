import torch
import numpy as np
import math
from scipy.signal import lfilter  # Import the fast filter function


# --- The EQFilterDesigner class can remain exactly as you wrote it ---
class EQFilterDesigner:
    @staticmethod
    def calculate_peaking_eq(f, g, q, sr):
        A = 10 ** (g / 40)
        w = 2 * math.pi * f / sr
        a = math.sin(w) / (2 * q)
        c = math.cos(w)
        b0 = 1 + a * A
        b1 = -2 * c
        b2 = 1 - a * A
        a0 = 1 + a / A
        a1 = -2 * c
        a2 = 1 - a / A
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_low_pass(f, q, sr):
        w = 2 * math.pi * f / sr
        a = math.sin(w) / (2 * q)
        c = math.cos(w)
        b0 = (1 - c) / 2
        b1 = 1 - c
        b2 = b0
        a0 = 1 + a
        a1 = -2 * c
        a2 = 1 - a
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_high_pass(f, q, sr):
        w = 2 * math.pi * f / sr
        a = math.sin(w) / (2 * q)
        c = math.cos(w)
        b0 = (1 + c) / 2
        b1 = -(1 + c)
        b2 = b0
        a0 = 1 + a
        a1 = -2 * c
        a2 = 1 - a
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_low_shelf(f, g, q, sr):
        w = 2 * math.pi * f / sr
        A = 10 ** (g / 40)
        b = math.sqrt(A) / q
        c = math.cos(w)
        s = math.sin(w)
        b0 = A * ((A + 1) - (A - 1) * c + b * s)
        b1 = 2 * A * ((A - 1) - (A + 1) * c)
        b2 = A * ((A + 1) - (A - 1) * c - b * s)
        a0 = (A + 1) + (A - 1) * c + b * s
        a1 = -2 * ((A - 1) + (A + 1) * c)
        a2 = (A + 1) + (A - 1) * c - b * s
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_high_shelf(f, g, q, sr):
        w = 2 * math.pi * f / sr
        A = 10 ** (g / 40)
        b = math.sqrt(A) / q
        c = math.cos(w)
        s = math.sin(w)
        b0 = A * ((A + 1) + (A - 1) * c + b * s)
        b1 = -2 * A * ((A - 1) + (A + 1) * c)
        b2 = A * ((A + 1) + (A - 1) * c - b * s)
        a0 = (A + 1) - (A - 1) * c + b * s
        a1 = 2 * ((A - 1) - (A + 1) * c)
        a2 = (A + 1) - (A - 1) * c - b * s
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_band_pass(f, q, sr):
        w = 2 * math.pi * f / sr
        a = math.sin(w) / (2 * q)
        c = math.cos(w)
        b0 = a
        b1 = 0
        b2 = -a
        a0 = 1 + a
        a1 = -2 * c
        a2 = 1 - a
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def calculate_notch(f, q, sr):
        w = 2 * math.pi * f / sr
        a = math.sin(w) / (2 * q)
        c = math.cos(w)
        b0 = 1
        b1 = -2 * c
        b2 = 1
        a0 = 1 + a
        a1 = -2 * c
        a2 = 1 - a
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


class EQBand:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.enabled = False
        self.coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)  # b0, b1, b2, a1, a2

    def set_parameters(self, enabled, filter_type, frequency, gain_db, q_factor):
        self.enabled = enabled
        if not self.enabled:
            self.coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)
            return

        frequency = max(1.0, min(frequency, self.sample_rate / 2 - 1.0))
        q_factor = max(0.01, q_factor)

        try:
            if filter_type == "bell":
                self.coeffs = EQFilterDesigner.calculate_peaking_eq(frequency, gain_db, q_factor, self.sample_rate)
            elif filter_type == "low_pass":
                self.coeffs = EQFilterDesigner.calculate_low_pass(frequency, q_factor, self.sample_rate)
            elif filter_type == "high_pass":
                self.coeffs = EQFilterDesigner.calculate_high_pass(frequency, q_factor, self.sample_rate)
            elif filter_type == "low_shelf":
                self.coeffs = EQFilterDesigner.calculate_low_shelf(frequency, gain_db, q_factor, self.sample_rate)
            elif filter_type == "high_shelf":
                self.coeffs = EQFilterDesigner.calculate_high_shelf(frequency, gain_db, q_factor, self.sample_rate)
            elif filter_type == "band_pass":
                self.coeffs = EQFilterDesigner.calculate_band_pass(frequency, q_factor, self.sample_rate)
            elif filter_type == "notch":
                self.coeffs = EQFilterDesigner.calculate_notch(frequency, q_factor, self.sample_rate)
            else:
                self.coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)
        except (ValueError, ZeroDivisionError):
            self.coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    def process(self, audio_channel):
        b = np.array([self.coeffs[0], self.coeffs[1], self.coeffs[2]])
        a = np.array([1, self.coeffs[3], self.coeffs[4]])
        return lfilter(b, a, audio_channel)


class ParametricEQNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
                "output_gain": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.1}),
                "bypass": ("BOOLEAN", {"default": False}),
                **{
                    f"band_{i}_{p}": v
                    for i in range(1, 9)
                    for p, v in {
                        "enabled": ("BOOLEAN", {"default": True}),
                        "type": (["bell", "low_pass", "high_pass", "low_shel", "high_shel", "band_pass", "notch"],),
                        "frequency": ("FLOAT", {"default": 1000.0, "min": 20.0, "max": 20000.0}),
                        "gain": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0}),
                        "q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0}),
                    }.items()
                },
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process_parametric_eq"
    CATEGORY = "CRT/Audio"

    def __init__(self):
        self.eq_bands = []
        self.current_sample_rate = 0

    def _initialize_bands(self, sample_rate):
        if self.current_sample_rate != sample_rate or not self.eq_bands:
            self.eq_bands = [EQBand(sample_rate) for _ in range(8)]
            self.current_sample_rate = sample_rate

    def process_parametric_eq(self, audio, sample_rate, output_gain, bypass, **kwargs):
        if "waveform" not in audio or "sample_rate" not in audio:
            return (audio,)

        waveform = audio["waveform"].clone()
        sr = audio["sample_rate"]

        if bypass:
            return ({"waveform": waveform, "sample_rate": sr},)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[-1] == 0:
            return (audio,)

        self._initialize_bands(sr)
        for i, band in enumerate(self.eq_bands):
            band.set_parameters(
                kwargs.get(f"band_{i+1}_enabled", False),
                kwargs.get(f"band_{i+1}_type", "bell"),
                kwargs.get(f"band_{i+1}_frequency", 1000.0),
                kwargs.get(f"band_{i+1}_gain", 0.0),
                kwargs.get(f"band_{i+1}_q", 1.0),
            )

        processed_waveform = waveform.cpu().numpy().astype(np.float64)
        num_channels = processed_waveform.shape[0]

        for i in range(num_channels):
            channel_data = processed_waveform[i, :]
            for band in self.eq_bands:
                if band.enabled:
                    channel_data = band.process(channel_data)
            processed_waveform[i, :] = channel_data

        if abs(output_gain) > 1e-6:
            linear_gain = 10.0 ** (output_gain / 20.0)
            processed_waveform *= linear_gain

        final_waveform = torch.from_numpy(processed_waveform.astype(np.float32))

        if final_waveform.dim() == 1:
            final_waveform = final_waveform.unsqueeze(0)

        return ({"waveform": final_waveform, "sample_rate": sr},)


# --- FIX ---
# Explicitly register the node using its Python class name as the key.
# This makes the node load correctly while keeping the ID the JavaScript expects.
NODE_CLASS_MAPPINGS = {"ParametricEQNode": ParametricEQNode}

# Also map the display name for a clean title in the UI.
NODE_DISPLAY_NAME_MAPPINGS = {"ParametricEQNode": "Parametric EQ (CRT)"}

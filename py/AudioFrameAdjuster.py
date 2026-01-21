import torch
import torchaudio
import numpy as np

class AudioFrameAdjuster:
    """
    Extends or trims audio to match a specific frame count and FPS.
    Supports looping for extension and precise trimming.
    Now also outputs the final duration in seconds.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_count": ("INT", {
                    "default": 120,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "extension_mode": (["loop", "silence", "fade_loop"],),
                "fade_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Fade duration in seconds for fade_loop mode"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration_seconds")
    FUNCTION = "adjust_audio"
    CATEGORY = "CRT/Audio"

    def adjust_audio(self, audio, frame_count, fps, extension_mode, fade_duration):
        """
        Adjust audio duration to match frame_count / fps
        and return both the adjusted audio and its final duration in seconds.
        """
        waveform = audio['waveform']  # Shape: [batch, channels, samples]
        sample_rate = audio['sample_rate']
        
        # Calculate target duration in seconds
        target_duration = frame_count / fps
        target_samples = int(target_duration * sample_rate)
        
        # Get current number of samples
        current_samples = waveform.shape[2]
        
        print(f"[AudioFrameAdjuster] Current duration: {current_samples/sample_rate:.2f}s ({current_samples} samples)")
        print(f"[AudioFrameAdjuster] Target duration: {target_duration:.2f}s ({target_samples} samples)")
        print(f"[AudioFrameAdjuster] Frame count: {frame_count}, FPS: {fps}")
        
        if current_samples == target_samples:
            print(f"[AudioFrameAdjuster] Audio duration matches target, no adjustment needed")
            adjusted_waveform = waveform
        elif current_samples > target_samples:
            # Trim audio
            print(f"[AudioFrameAdjuster] Trimming audio from {current_samples} to {target_samples} samples")
            adjusted_waveform = waveform[:, :, :target_samples]
        else:
            # Extend audio
            samples_needed = target_samples - current_samples
            print(f"[AudioFrameAdjuster] Extending audio by {samples_needed} samples using '{extension_mode}' mode")
            
            if extension_mode == "silence":
                silence = torch.zeros(
                    waveform.shape[0],
                    waveform.shape[1],
                    samples_needed,
                    dtype=waveform.dtype,
                    device=waveform.device
                )
                adjusted_waveform = torch.cat([waveform, silence], dim=2)
            
            elif extension_mode == "loop":
                num_loops = (samples_needed // current_samples) + 1
                looped = waveform.repeat(1, 1, num_loops + 1)
                adjusted_waveform = looped[:, :, :target_samples]
            
            elif extension_mode == "fade_loop":
                fade_samples = int(fade_duration * sample_rate)
                fade_samples = min(fade_samples, current_samples // 2)
                
                fade_out = torch.linspace(1, 0, fade_samples, device=waveform.device)
                fade_in  = torch.linspace(0, 1, fade_samples, device=waveform.device)
                
                segments = [waveform]
                remaining = samples_needed
                
                while remaining > 0:
                    if remaining >= current_samples:
                        segment = waveform.clone()
                        
                        if len(segments) > 0:
                            segment[:, :, :fade_samples] *= fade_in
                            segments[-1][:, :, -fade_samples:] *= fade_out
                            overlap = segments[-1][:, :, -fade_samples:] + segment[:, :, :fade_samples]
                            segments[-1] = torch.cat([
                                segments[-1][:, :, :-fade_samples],
                                overlap
                            ], dim=2)
                            segment = segment[:, :, fade_samples:]
                        
                        segments.append(segment)
                        remaining -= current_samples
                    else:
                        segment = waveform[:, :, :remaining].clone()
                        
                        if len(segments) > 0 and remaining > fade_samples:
                            segment[:, :, :fade_samples] *= fade_in
                            segments[-1][:, :, -fade_samples:] *= fade_out
                            overlap = segments[-1][:, :, -fade_samples:] + segment[:, :, :fade_samples]
                            segments[-1] = torch.cat([
                                segments[-1][:, :, :-fade_samples],
                                overlap
                            ], dim=2)
                            segment = segment[:, :, fade_samples:]
                        
                        segments.append(segment)
                        remaining = 0
                
                adjusted_waveform = torch.cat(segments, dim=2)[:, :, :target_samples]

        # Final result audio
        result_audio = {
            'waveform': adjusted_waveform,
            'sample_rate': sample_rate
        }
        
        # Calculate and return final duration in seconds
        final_samples = adjusted_waveform.shape[2]
        final_duration = final_samples / sample_rate
        
        print(f"[AudioFrameAdjuster] Final duration: {final_duration:.3f}s ({final_samples} samples)")
        
        return (result_audio, final_duration)


NODE_CLASS_MAPPINGS = {
    "AudioFrameAdjuster": AudioFrameAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameAdjuster": "Audio Frame Adjuster (CRT)",
}
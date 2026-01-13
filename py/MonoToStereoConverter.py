"""
Mono to Stereo Converter Node for ComfyUI
Converts mono audio to stereo by duplicating the mono channel
"""

import torch
import numpy as np

class MonoToStereoConverter:
    """
    Converts mono audio to stereo by duplicating the mono channel to both left and right channels.
    If input is already stereo or multi-channel, passes through unchanged.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = "CRT/Audio"
    
    def convert(self, audio):
        """
        Convert mono audio to stereo
        
        Args:
            audio: Dictionary containing 'waveform' (tensor) and 'sample_rate' (int)
                  waveform shape: (batch, channels, samples)
        
        Returns:
            Dictionary with stereo audio
        """
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Check the shape: (batch, channels, samples)
        batch_size, channels, samples = waveform.shape
        
        # If already stereo or multi-channel, return as-is
        if channels >= 2:
            return (audio,)
        
        # If mono (1 channel), duplicate to create stereo
        if channels == 1:
            # Duplicate the mono channel to create stereo by concatenating along channel dimension
            stereo_waveform = torch.cat([waveform, waveform], dim=1)
            
            result = {
                'waveform': stereo_waveform,
                'sample_rate': sample_rate
            }
            
            print(f"[MonoToStereoConverter] Converted from {channels} channel(s) to {stereo_waveform.shape[1]} channels")
            
            return (result,)
        
        # This shouldn't happen, but handle edge case of 0 channels
        return (audio,)
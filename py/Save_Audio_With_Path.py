import os
import torch
import numpy as np
import folder_paths
import wave
from scipy.io.wavfile import write as write_wav


CLIP_CEILING_DBFS = -1.0
CLIP_CEILING = 10 ** (CLIP_CEILING_DBFS / 20.0)


def _write_pcm24_wav(path, sample_rate, audio_data):
    if audio_data.ndim == 1:
        audio_data = audio_data[:, None]

    audio_i24 = np.round(audio_data * 8388607.0).astype("<i4")
    audio_i24 = np.clip(audio_i24, -8388608, 8388607)
    packed = audio_i24.reshape(-1, 1).view(np.uint8).reshape(-1, 4)[:, :3]

    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio_data.shape[1])
        wav.setsampwidth(3)
        wav.setframerate(int(sample_rate))
        wav.writeframes(packed.tobytes())


class SaveAudioWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "audio": ("AUDIO",),
                "folder_path": (
                    "STRING",
                    {
                        "default": output_dir,
                        "tooltip": "Base folder (or a full file path to get the folder from). Defaults to ComfyUI's output folder.",
                    },
                ),
                "subfolder_name": (
                    "STRING",
                    {"default": "audio", "tooltip": "Subfolder to create within the base folder."},
                ),
                "filename": (
                    "STRING",
                    {"default": "output", "tooltip": "File name for the audio file (without extension)."},
                ),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix to append to the filename."}),
                "sample_rate": (
                    ["44100", "48000", "88200", "96000", "192000"],
                    {
                        "default": "44100",
                        "tooltip": "Fallback audio sample rate in Hz if not in audio data.",
                    },
                ),
                "bit_depth": (
                    ["24-bit PCM", "32-bit float"],
                    {
                        "default": "24-bit PCM",
                        "tooltip": "WAV encoding. 24-bit PCM is the default; 32-bit float can preserve over-full-scale samples when normalization is disabled.",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, existing files will be overwritten. If disabled, a numbered suffix is added.",
                    },
                ),
                "normalize_clipping": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, peak-normalizes clipped audio. If disabled, 24-bit PCM hard-clips to -1 dBFS while 32-bit float saves over-full-scale samples.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    CATEGORY = "CRT/Save"
    DESCRIPTION = "Saves audio to a specified folder with a subfolder as an uncompressed WAV file."

    def save_audio(
        self,
        audio,
        folder_path,
        subfolder_name,
        filename,
        suffix,
        sample_rate,
        bit_depth,
        overwrite,
        normalize_clipping=True,
    ):
        if audio is None:
            print("❌ ERROR: No input audio provided to SaveAudioWithPath.")
            return ()

        try:
            # --- NEW: Automatically handle if folder_path is actually a file path ---
            if os.path.isfile(folder_path):
                # If the input is a file, use its parent directory
                base_path = os.path.dirname(folder_path)
                print(f"💡 Input 'folder_path' was a file. Using its parent directory: {base_path}")
            else:
                # Otherwise, use the input path as is
                base_path = folder_path
            # --- END NEW ---

            # Clean up folder, file, and suffix names
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')
            suffix_clean = suffix.strip()

            if not subfolder_clean or not filename_clean:
                raise ValueError("Subfolder and Filename fields cannot be empty.")

            # Combine filename and the custom suffix
            filename_with_suffix = f"{filename_clean}{suffix_clean}"

            # Construct the full directory path using the corrected base_path
            final_dir = os.path.join(base_path, subfolder_clean)
            os.makedirs(final_dir, exist_ok=True)

            # Handle file naming and overwriting
            base_filepath = os.path.join(final_dir, f"{filename_with_suffix}.wav")
            final_filepath = base_filepath

            if not overwrite and os.path.exists(base_filepath):
                counter = 1
                while os.path.exists(final_filepath):
                    final_filepath = os.path.join(final_dir, f"{filename_with_suffix}_{counter}.wav")
                    counter += 1

            # Correctly handle the AUDIO data type
            if not isinstance(audio, dict) or 'waveform' not in audio:
                raise TypeError(
                    "Input 'audio' is not in the expected format. It must be a dictionary with a 'waveform' key."
                )

            waveform_tensor = audio['waveform']
            sr_to_use = int(audio.get('sample_rate', sample_rate))

            if waveform_tensor is None or waveform_tensor.nelement() == 0:
                print("❌ ERROR: The 'waveform' tensor in the audio input is empty.")
                return ()

            audio_data = waveform_tensor[0].cpu().numpy()
            audio_data = audio_data.T

            peak = np.max(np.abs(audio_data))
            if peak > 1.0:
                if normalize_clipping:
                    print("⚠️ Warning: Audio data is clipping. It will be normalized.")
                    audio_data /= peak
                elif bit_depth == "32-bit float":
                    print("⚠️ Warning: Audio data is clipping. Saving over-full-scale samples as 32-bit float.")
                else:
                    print("⚠️ Warning: Audio data is clipping. Hard-clipping peaks to -1 dBFS.")
                    audio_data = np.clip(audio_data, -CLIP_CEILING, CLIP_CEILING)

            if bit_depth == "32-bit float":
                write_wav(final_filepath, sr_to_use, audio_data.astype(np.float32))
            else:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                _write_pcm24_wav(final_filepath, sr_to_use, audio_data)
            print(f"✅ Saved audio successfully to: {final_filepath}")
            return ()

        except Exception as e:
            print(f"❌ ERROR in SaveAudioWithPath: {str(e)}")
            raise e


# Dictionary mappings for ComfyUI
NODE_CLASS_MAPPINGS = {"SaveAudioWithPath": SaveAudioWithPath}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveAudioWithPath": "Save Audio With Path (CRT)"}

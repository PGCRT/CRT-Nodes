import os
import torch
import numpy as np
import folder_paths
from scipy.io.wavfile import write as write_wav


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
                    "INT",
                    {
                        "default": 44100,
                        "min": 1,
                        "max": 192000,
                        "tooltip": "Fallback audio sample rate in Hz if not in audio data.",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, existing files will be overwritten. If disabled, a numbered suffix is added.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    CATEGORY = "CRT/Save"
    DESCRIPTION = "Saves audio to a specified folder with a subfolder as an uncompressed WAV file."

    def save_audio(self, audio, folder_path, subfolder_name, filename, suffix, sample_rate, overwrite):
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
            sr_to_use = audio.get('sample_rate', sample_rate)

            if waveform_tensor is None or waveform_tensor.nelement() == 0:
                print("❌ ERROR: The 'waveform' tensor in the audio input is empty.")
                return ()

            audio_data = waveform_tensor[0].cpu().numpy()
            audio_data = audio_data.T

            if np.any(np.abs(audio_data) > 1.0):
                print("⚠️ Warning: Audio data is clipping. It will be normalized.")
                audio_data /= np.max(np.abs(audio_data))

            audio_pcm = (audio_data * 32767).astype(np.int16)

            write_wav(final_filepath, sr_to_use, audio_pcm)
            print(f"✅ Saved audio successfully to: {final_filepath}")
            return ()

        except Exception as e:
            print(f"❌ ERROR in SaveAudioWithPath: {str(e)}")
            raise e


# Dictionary mappings for ComfyUI
NODE_CLASS_MAPPINGS = {"SaveAudioWithPath": SaveAudioWithPath}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveAudioWithPath": "Save Audio With Path (CRT)"}

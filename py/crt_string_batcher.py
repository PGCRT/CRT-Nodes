import random
import os
import folder_paths


class CRT_StringBatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # --- Batching Controls ---
                "consolidate_lines": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "If True, removes line breaks within each individual string input"},
                ),
                "join_with": (
                    ["Newline", "Space"],
                    {"tooltip": "'Newline' for 'return to line' behavior between strings"},
                ),
                "suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Character(s) to add to the end of each string before joining",
                    },
                ),
                # --- Randomization ---
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # --- File Saving Options ---
                "save_to_file": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable to save each processed string as a .txt file"},
                ),
                # This path is relative to the ComfyUI output directory.
                # The script correctly handles creating this subfolder.
                "file_path": (
                    "STRING",
                    {
                        "default": "SavedPrompts",
                        "tooltip": "The directory within the main output folder to save the .txt files to",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "prompt", "tooltip": "The prefix for the saved filenames"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_output",)
    FUNCTION = "batch_strings"
    CATEGORY = "CRT/Text"

    def batch_strings(
        self, consolidate_lines, join_with, suffix, randomize, seed, save_to_file, file_path, filename_prefix, **kwargs
    ):
        # Collect and sort all incoming string inputs to ensure correct order
        string_keys = sorted([key for key in kwargs if key.startswith('string_')], key=lambda x: int(x.split('_')[1]))

        processed_strings = []
        for key in string_keys:
            s = kwargs[key]
            # Skip empty or whitespace-only strings
            if not s or not s.strip():
                continue

            # 1. Consolidate lines within the string if toggled ON
            processed_s = " ".join(s.split()) if consolidate_lines else s

            # 2. Add the user-defined suffix
            if suffix:
                processed_s += suffix

            # Save the individual string to a file if toggled ON
            if save_to_file:
                # Get the absolute path by joining the main output directory with the specified file_path
                output_dir = os.path.join(folder_paths.get_output_directory(), file_path)

                # Create the directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Get the index from the input key for a consistent filename
                string_index = int(key.split('_')[1])
                file_name = f"{filename_prefix}_{string_index:03d}.txt"
                full_path = os.path.join(output_dir, file_name)

                # Write the processed string to the file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(processed_s)

            processed_strings.append(processed_s)

        # 3. Randomize the list of processed strings if enabled
        if randomize:
            rng = random.Random(seed)
            rng.shuffle(processed_strings)

        # 4. Join the final list of strings using the selected separator
        separator = "\n" if join_with == "Newline" else " "
        final_string = separator.join(processed_strings)

        return (final_string,)

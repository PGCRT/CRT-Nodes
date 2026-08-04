import os
import secrets

import folder_paths
import numpy as np
from PIL import Image


class SaveImageWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "image": ("IMAGE",),
                "folder_path": (
                    "STRING",
                    {
                        "default": output_dir,
                        "tooltip": "Base folder path. Defaults to ComfyUI's output folder.",
                    },
                ),
                "subfolder_name": (
                    "STRING",
                    {
                        "default": "images",
                        "tooltip": "Optional subfolder within the base folder. Leave empty to save directly into the base folder.",
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": "Base file name without extension. Leave empty to generate a random name. Existing files are never overwritten.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional suffix appended to filename.",
                    },
                ),
                "extension": (
                    ["png", "jpg"],
                    {"default": "png", "tooltip": "Image file extension."},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CATEGORY = "CRT/Save"
    DESCRIPTION = (
        "Saves every image in a batch to a specified folder without overwriting "
        "existing files. An empty filename generates a random name."
    )

    @staticmethod
    def _reserve_unique_file(directory, base_filename, extension):
        """Atomically reserve a new path, adding a counter after collisions."""
        counter = 0
        while True:
            numbered_suffix = "" if counter == 0 else f"_{counter}"
            final_path = os.path.join(
                directory,
                f"{base_filename}{numbered_suffix}.{extension}",
            )
            try:
                return final_path, open(final_path, "xb")
            except FileExistsError:
                counter += 1

    def save_images(
        self,
        image,
        folder_path,
        subfolder_name,
        filename,
        suffix,
        extension,
    ):
        if image is None:
            return ()

        try:
            subfolder_clean = subfolder_name.strip().lstrip("/\\")
            filename_clean = filename.strip().lstrip("/\\")
            suffix_clean = suffix.strip()
            extension_clean = extension.lower()

            final_dir = (
                os.path.join(folder_path, subfolder_clean)
                if subfolder_clean
                else folder_path
            )
            os.makedirs(final_dir, exist_ok=True)

            random_filename = not filename_clean
            filename_root = (
                f"image_{secrets.token_hex(16)}"
                if random_filename
                else filename_clean
            )
            filename_root = f"{filename_root}{suffix_clean}"
            batch_size = image.shape[0]

            for index in range(batch_size):
                base_filename = (
                    f"{filename_root}_{index + 1}"
                    if batch_size > 1
                    else filename_root
                )
                image_array = np.clip(
                    image[index].detach().cpu().numpy() * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
                pil_image = Image.fromarray(image_array)

                final_filepath = None
                file_handle = None
                try:
                    final_filepath, file_handle = self._reserve_unique_file(
                        final_dir,
                        base_filename,
                        extension_clean,
                    )
                    with file_handle:
                        if extension_clean == "jpg":
                            pil_image.save(
                                file_handle,
                                format="JPEG",
                                quality=98,
                                subsampling="4:4:4",
                            )
                        else:
                            pil_image.save(file_handle, format="PNG")
                except Exception:
                    if file_handle is not None and not file_handle.closed:
                        file_handle.close()
                    if final_filepath is not None:
                        try:
                            os.remove(final_filepath)
                        except OSError:
                            pass
                    raise

                name_mode = "random name" if random_filename else "unique path"
                print(
                    f"[CRT Save Image With Path][OK] Saved using {name_mode}: "
                    f"{final_filepath}"
                )

            return ()

        except Exception as error:
            print(f"[CRT Save Image With Path][ERROR] {error}")
            raise


NODE_CLASS_MAPPINGS = {"SaveImageWithPath": SaveImageWithPath}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithPath": "Save Image With Path (CRT)"
}

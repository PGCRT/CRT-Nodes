from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import comfy.sd
import comfy.utils
import folder_paths


MAX_SEED = 0xFFFFFFFFFFFFFFFF
LORA_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth"}


def _clean_folder_value(value: str) -> str:
    return os.path.expandvars(os.path.expanduser((value or "").strip().strip("\"'")))


def _relative_folder(value: str) -> str:
    value = value.replace("\\", "/").strip("/")
    if value.casefold() == "loras":
        return ""
    if value.casefold().startswith("loras/"):
        return value[6:].strip("/")
    return value


def list_lora_candidates(folder_value: str) -> list[tuple[str, Path]]:
    """
    Return stable (display name, absolute path) pairs from one folder.

    The folder may be an absolute filesystem path or a path relative to
    ComfyUI's models/loras folder. Only files directly in that folder are used.
    """
    cleaned = _clean_folder_value(folder_value)
    if not cleaned:
        raise ValueError("lora_folder cannot be empty.")

    requested_path = Path(cleaned)
    candidates: list[tuple[str, Path]] = []

    if requested_path.is_absolute():
        if not requested_path.is_dir():
            raise FileNotFoundError(f"LoRA folder does not exist: {requested_path}")

        for path in requested_path.iterdir():
            if path.is_file() and path.suffix.casefold() in LORA_EXTENSIONS:
                candidates.append((path.name, path.resolve()))
    else:
        requested_relative = _relative_folder(cleaned)
        requested_key = requested_relative.casefold()

        for relative_name in folder_paths.get_filename_list("loras"):
            normalized_name = relative_name.replace("\\", "/")
            suffix = PurePosixPath(normalized_name).suffix.casefold()
            parent = PurePosixPath(normalized_name).parent.as_posix()
            if parent == ".":
                parent = ""

            if suffix not in LORA_EXTENSIONS or parent.casefold() != requested_key:
                continue

            full_path = folder_paths.get_full_path("loras", relative_name)
            if full_path and Path(full_path).is_file():
                candidates.append((relative_name, Path(full_path).resolve()))

    candidates.sort(key=lambda item: item[0].casefold())
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA files were found directly in folder: {folder_value}"
        )
    return candidates


def select_lora(folder_value: str, seed: int) -> tuple[int, int, str, Path]:
    candidates = list_lora_candidates(folder_value)
    index = int(seed) % len(candidates)
    display_name, full_path = candidates[index]
    return index, len(candidates), display_name, full_path


def load_persona_text(lora_path: Path) -> tuple[str, Path]:
    persona_path = lora_path.with_suffix(".md")
    if not persona_path.is_file():
        raise FileNotFoundError(
            "Matching persona markdown was not found. Expected: "
            f"{persona_path}"
        )
    return persona_path.read_text(encoding="utf-8-sig"), persona_path


class SeededPersonaLoraLoader:
    """
    Deterministically select a LoRA from a folder, apply it to MODEL only,
    and return the text of the same-stem .md persona file.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model to patch with the selected LoRA."
                    },
                ),
                "lora_folder": (
                    "STRING",
                    {
                        "default": r"KREA2\Turbo\PERSONA",
                        "multiline": False,
                        "tooltip": (
                            "Folder relative to models/loras, or an absolute folder. "
                            "Only LoRA files directly inside it are considered."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": (
                            "Selects the LoRA by seed modulo sorted LoRA count. "
                            "Use ComfyUI's fixed, increment, decrement, or randomize control."
                        ),
                    },
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength applied to the diffusion model.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "persona")
    OUTPUT_TOOLTIPS = (
        "The model with the selected LoRA applied.",
        "Contents of the .md file whose stem matches the selected LoRA.",
    )
    FUNCTION = "load_seeded_persona_lora"
    CATEGORY = "CRT/LoRA"
    DESCRIPTION = (
        "Selects a LoRA deterministically from a folder using a native ComfyUI "
        "seed control, applies it to MODEL, and outputs its same-stem markdown "
        "persona as a string. Works with any model architecture."
    )
    SEARCH_ALIASES = [
        "seeded lora",
        "random lora",
        "persona lora",
        "lora folder",
    ]

    @classmethod
    def VALIDATE_INPUTS(cls, lora_folder, **kwargs):
        if not (lora_folder or "").strip():
            return "lora_folder cannot be empty."
        return True

    def _load_lora_state(self, lora_path: Path):
        stat = lora_path.stat()
        cache_key = (str(lora_path), stat.st_mtime_ns, stat.st_size)

        if self.loaded_lora is not None and self.loaded_lora[0] == cache_key:
            return self.loaded_lora[1], self.loaded_lora[2]

        lora, metadata = comfy.utils.load_torch_file(
            str(lora_path),
            safe_load=True,
            return_metadata=True,
        )
        self.loaded_lora = (cache_key, lora, metadata)
        return lora, metadata

    def load_seeded_persona_lora(
        self,
        model,
        lora_folder,
        seed,
        strength_model,
    ):
        index, count, display_name, lora_path = select_lora(lora_folder, seed)
        persona_text, persona_path = load_persona_text(lora_path)

        output_model = model

        if float(strength_model) != 0.0:
            lora, metadata = self._load_lora_state(lora_path)
            output_model, _ = comfy.sd.load_lora_for_models(
                model,
                None,
                lora,
                float(strength_model),
                0.0,
                lora_metadata=metadata,
            )

        selected_number = index + 1
        print(
            "[Seeded Persona LoRA] "
            f"seed={seed} selected={selected_number}/{count} "
            f"lora='{display_name}' persona='{persona_path.name}' "
            f"strength={float(strength_model):g}"
        )

        return output_model, persona_text

import random
import re
from pathlib import Path

import torch


class CRT_FileBatchPromptScheduler:
    @staticmethod
    def natural_sort_key(path):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", path.name)
        ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "folder_path": ("STRING", {"default": ""}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 64}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "file_extension": ("STRING", {"default": ".txt"}),
                "max_words": ("INT", {"default": 0, "min": 0}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "print_index": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Batch Randomize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Shuffle without repeats across incrementing seeds. Every file is presented once before a new shuffled cycle begins. No cache is stored. Disabled keeps consecutive selection.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "batch_count", "prompts_text")
    FUNCTION = "schedule_from_files"
    CATEGORY = "CRT/Conditioning"

    @staticmethod
    def limit_words(text, max_words):
        if max_words <= 0:
            return text.strip()
        return " ".join(text.split()[:max_words])

    @staticmethod
    def select_files(files, batch_count, seed, batch_randomize=False):
        files = list(files)
        file_count = len(files)
        requested = max(1, int(batch_count))
        if file_count == 0:
            return []

        if not batch_randomize:
            start = (int(seed) * requested) % file_count
            return [files[(start + index) % file_count] for index in range(requested)]

        # Build a deterministic stream of shuffled cycles and use the seed as
        # the batch/run index into that stream. This provides no-repeat random
        # selection across incrementing seeds without retaining RAM or disk
        # state between executions.
        stream_start = int(seed) * requested
        selected = []
        cycles = {}
        for position in range(stream_start, stream_start + requested):
            cycle_index, cycle_offset = divmod(position, file_count)
            cycle = cycles.get(cycle_index)
            if cycle is None:
                cycle = list(range(file_count))
                random.Random(cycle_index).shuffle(cycle)
                cycles[cycle_index] = cycle
            selected.append(files[cycle[cycle_offset]])
        return selected

    def schedule_from_files(
        self,
        clip,
        folder_path,
        batch_count,
        seed,
        file_extension,
        max_words,
        crawl_subfolders,
        print_index,
        **kwargs,
    ):
        batch_randomize = bool(
            kwargs.get("Batch Randomize", kwargs.get("batch_randomize", False))
        )
        prompts = [""]

        if folder_path and Path(folder_path).is_dir():
            try:
                folder = Path(folder_path)
                extension = f".{file_extension.strip().lstrip('.').lower()}"
                path_iterator = (
                    folder.rglob(f"*{extension}")
                    if crawl_subfolders
                    else folder.glob(f"*{extension}")
                )
                files = sorted(
                    (path for path in path_iterator if path.is_file()),
                    key=self.natural_sort_key,
                )

                if files:
                    selected = self.select_files(
                        files,
                        batch_count,
                        seed,
                        batch_randomize=batch_randomize,
                    )
                    mode = "random" if batch_randomize else "consecutive"
                    print(
                        f"[CRT File Batch Prompt Scheduler] Selected "
                        f"{len(selected)} file(s) in {mode} mode using seed {int(seed)}."
                    )
                    prompts = []
                    for path in selected:
                        try:
                            text = path.read_text(
                                encoding="utf-8", errors="ignore"
                            ).strip()
                            prompts.append(self.limit_words(text, max_words))
                        except Exception as error:
                            print(
                                f"[CRT File Batch Prompt Scheduler] "
                                f"Could not read '{path}': {error}"
                            )
                            prompts.append("")
                    prompts = [prompt for prompt in prompts if prompt] or [""]
            except Exception as error:
                print(f"[CRT File Batch Prompt Scheduler] File loading error: {error}")

        lines = [
            f"Prompt {index + 1} : {prompt}" if print_index else prompt
            for index, prompt in enumerate(prompts)
        ]
        final_text = "\n\n".join(lines)

        cond_list = []
        pooled_list = []

        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)

            if pooled is None:
                hidden_size = cond.shape[-1]
                pooled = torch.zeros(
                    cond.shape[0],
                    hidden_size,
                    device=cond.device,
                    dtype=cond.dtype,
                )
            pooled_list.append(pooled)

        if cond_list:
            max_length = max(conditioning.shape[1] for conditioning in cond_list)
            for index, conditioning in enumerate(cond_list):
                if conditioning.shape[1] < max_length:
                    padding = torch.zeros(
                        conditioning.shape[0],
                        max_length - conditioning.shape[1],
                        conditioning.shape[2],
                        device=conditioning.device,
                        dtype=conditioning.dtype,
                    )
                    cond_list[index] = torch.cat([conditioning, padding], dim=1)

        final_cond = torch.cat(cond_list, dim=0)
        final_pooled = torch.cat(pooled_list, dim=0)
        conditioning = [[final_cond, {"pooled_output": final_pooled}]]

        return (conditioning, len(prompts), final_text)

import re
from pathlib import Path
import torch


class CRT_FileBatchPromptScheduler:
    @staticmethod
    def natural_sort_key(s):
        s = s.name
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "folder_path": ("STRING", {"default": ""}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "file_extension": ("STRING", {"default": ".txt"}),
                "max_words": ("INT", {"default": 0, "min": 0}),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "print_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "batch_count", "prompts_text")
    FUNCTION = "schedule_from_files"
    CATEGORY = "CRT/Conditioning"

    def limit_words(self, text, max_words):
        if max_words <= 0:
            return text.strip()
        return ' '.join(text.split()[:max_words])

    def schedule_from_files(
        self, clip, folder_path, batch_count, seed, file_extension, max_words, crawl_subfolders, print_index
    ):
        prompts = [""]

        if folder_path and Path(folder_path).is_dir():
            try:
                folder = Path(folder_path)
                ext = f".{file_extension.strip().lstrip('.').lower()}"
                files = list(folder.rglob(f'*{ext}') if crawl_subfolders else folder.glob(f'*{ext}'))
                files = [f for f in files if f.is_file()]
                files = sorted(files, key=self.natural_sort_key)

                if files:
                    start = (seed * batch_count) % len(files)
                    selected = [files[(start + i) % len(files)] for i in range(batch_count)]
                    prompts = []
                    for f in selected:
                        try:
                            text = f.read_text(encoding="utf-8", errors="ignore").strip()
                            prompts.append(self.limit_words(text, max_words))
                        except Exception:
                            prompts.append("")
                    prompts = [p for p in prompts if p] or [""]
            except Exception as e:
                print(f"[CRT] File loading error: {e}")

        # ── Text output ─────────────────────────────────────
        lines = [f"Prompt {i+1} : {p}" if print_index else p for i, p in enumerate(prompts)]
        final_text = "\n\n".join(lines)

        # ── Conditioning (always with valid pooled_output) ─────────────────────────────────────
        cond_list = []
        pooled_list = []

        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)

            # ←←← ALWAYS provide pooled_output (zero tensor if None)
            if pooled is None:
                hidden_size = cond.shape[-1]
                pooled = torch.zeros(cond.shape[0], hidden_size, device=cond.device, dtype=cond.dtype)
            pooled_list.append(pooled)

        # Pad to same length
        if cond_list:
            max_len = max(c.shape[1] for c in cond_list)
            for i, c in enumerate(cond_list):
                if c.shape[1] < max_len:
                    pad = torch.zeros(c.shape[0], max_len - c.shape[1], c.shape[2], device=c.device, dtype=c.dtype)
                    cond_list[i] = torch.cat([c, pad], dim=1)

        final_cond = torch.cat(cond_list, dim=0)
        final_pooled = torch.cat(pooled_list, dim=0)

        conditioning = [[final_cond, {"pooled_output": final_pooled}]]

        return (conditioning, len(prompts), final_text)

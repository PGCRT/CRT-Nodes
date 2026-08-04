"""File-backed batch conditioning with KREA2 per-token prompt weighting.

The KREA2 attention-weighting technique is adapted from Krea2PromptWeight in
ComfyUI-KJNodes by Kijai (GPL-3.0), source revision
e27a505b3ba6ce42687fe00500deda103d9d6071. The CRT integration adds file
scheduling, shared batch weighting controls, batch-aware token positions,
conditioning padding, and CRT logging.
"""

import logging
import re
import types
from pathlib import Path

import torch
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention

from .File_Batch_Prompt_Scheduler import CRT_FileBatchPromptScheduler


TAG = "[CRT File Batch Prompt Scheduler KREA2]"
_QWEN_IM_START = 151644
_QWEN_USER = 872
_QWEN_NEWLINE = 198
_QWEN_IM_END = 151645
_WEIGHT_PATTERN = re.compile(r"\(([^():]+):(-?(?:\d+(?:\.\d*)?|\.\d+))\)")


def _krea2_user_content_span(token_ids):
    """Return the user-prompt span inside Qwen's chat template."""
    for index in range(len(token_ids) - 2):
        if (
            token_ids[index] == _QWEN_IM_START
            and token_ids[index + 1] == _QWEN_USER
            and token_ids[index + 2] == _QWEN_NEWLINE
        ):
            start = index + 3
            end = start
            while end < len(token_ids) and token_ids[end] != _QWEN_IM_END:
                end += 1
            return start, end
    return None, None


def _token_ids(tokens):
    key = next(iter(tokens))
    return [token[0] for token in tokens[key][0]]


def _krea2_phrase_token_ids(clip, text):
    ids = _token_ids(clip.tokenize(text))
    start, end = _krea2_user_content_span(ids)
    if start is None:
        return []
    return ids[start:end]


def _find_subsequence(sequence, subsequence, start, end):
    if not subsequence:
        return []
    length = len(subsequence)
    return [
        index
        for index in range(start, end - length + 1)
        if sequence[index : index + length] == subsequence
    ]


def _normalize_weight_sets(raw_weights):
    if not raw_weights:
        return []
    first = raw_weights[0]
    if isinstance(first, tuple) and len(first) == 3:
        return [raw_weights]
    return raw_weights


def krea2_attn_forward_weight(
    self,
    x,
    freqs=None,
    mask=None,
    transformer_options=None,
):
    """KREA2 self-attention with per-token value scaling and key bias."""
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope

    transformer_options = transformer_options or {}
    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

    weight_sets = _normalize_weight_sets(
        transformer_options.get("krea2_token_weights")
    )
    if weight_sets:
        v = v.clone()
        for batch_index in range(v.shape[0]):
            weights = weight_sets[batch_index % len(weight_sets)]
            for position, value_factor, _ in weights:
                if value_factor != 1.0 and 0 <= position < v.shape[2]:
                    v[batch_index, :, position] *= value_factor

    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        repeat = self.heads // self.kvheads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    bias = None
    if weight_sets and any(
        key_bias != 0.0
        for weights in weight_sets
        for _, _, key_bias in weights
    ):
        bias = q.new_zeros((q.shape[0], 1, k.shape[2]))
        for batch_index in range(q.shape[0]):
            weights = weight_sets[batch_index % len(weight_sets)]
            for position, _, key_bias in weights:
                if key_bias != 0.0 and 0 <= position < bias.shape[2]:
                    bias[batch_index, 0, position] = key_bias

    if bias is not None:
        output = attention_pytorch(
            q,
            k,
            v,
            self.heads,
            mask=bias,
            skip_reshape=True,
        )
    else:
        output = optimized_attention(
            q,
            k,
            v,
            self.heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
    return self.wo(output * torch.sigmoid(gate))


class CRT_FileBatchPromptSchedulerKREA2(CRT_FileBatchPromptScheduler):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "folder_path": (
                    "STRING",
                    {"default": "", "tooltip": "Folder containing prompt text files."},
                ),
                "batch_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "file_extension": ("STRING", {"default": ".txt"}),
                "max_words": ("INT", {"default": 0, "min": 0}),
                "weighted_phrases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Shared KREA2 weights applied to every selected prompt. "
                            "Use (phrase:-1) to suppress or (phrase:1.5) to emphasize. "
                            "The phrase must occur in the prompt."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Global multiplier for the shared weighting effect. "
                            "Use sampler CFG 1.0 with KREA2."
                        ),
                    },
                ),
                "crawl_subfolders": ("BOOLEAN", {"default": False}),
                "print_index": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Batch Randomize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Shuffle without repeats across incrementing seeds. "
                            "Every file is presented before a new cycle begins."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("model", "conditioning", "batch_count", "prompts_text")
    FUNCTION = "schedule_krea2"
    CATEGORY = "CRT/Conditioning"
    DESCRIPTION = (
        "Loads a scheduled batch of prompt files, applies one shared set of "
        "KREA2 phrase weights to the whole batch, and returns the patched model. "
        "Use CFG 1.0."
    )
    EXPERIMENTAL = True

    @staticmethod
    def parse_weighted_phrases(weighted_phrases):
        return [
            (match.group(1).strip(), float(match.group(2)))
            for match in _WEIGHT_PATTERN.finditer(weighted_phrases)
            if match.group(1).strip()
        ]

    @staticmethod
    def _weights_for_prompt(clip, tokens, cond_length, terms, strength):
        ids = _token_ids(tokens)
        visible_start = len(ids) - cond_length
        start, end = _krea2_user_content_span(ids)
        if start is None:
            start, end = visible_start, len(ids)

        weight_pairs = []
        missing = []
        for phrase, weight in terms:
            if weight > 1.0:
                value_factor = 1.0
                key_bias = strength * (weight - 1.0) * 2.0
            else:
                value_factor = 1.0 + strength * (weight - 1.0)
                key_bias = 0.0

            positions = []
            for variant in (" " + phrase, phrase):
                phrase_ids = _krea2_phrase_token_ids(clip, variant)
                matches = _find_subsequence(ids, phrase_ids, start, end)
                if matches:
                    for match in matches:
                        positions.extend(
                            match + offset - visible_start
                            for offset in range(len(phrase_ids))
                        )
                    break

            valid_positions = [
                position
                for position in positions
                if 0 <= position < cond_length
            ]
            if not valid_positions:
                missing.append(phrase)
                continue
            weight_pairs.extend(
                (position, value_factor, key_bias)
                for position in valid_positions
            )

        return weight_pairs, missing

    @staticmethod
    def _merge_conditioning(entries):
        max_length = max(conditioning.shape[1] for conditioning, _ in entries)
        padded_conditioning = []
        attention_masks = []
        need_attention_mask = any(
            "attention_mask" in metadata
            for _, metadata in entries
        ) or any(conditioning.shape[1] != max_length for conditioning, _ in entries)

        for conditioning, metadata in entries:
            length = conditioning.shape[1]
            if length < max_length:
                padding = conditioning.new_zeros(
                    conditioning.shape[0],
                    max_length - length,
                    conditioning.shape[2],
                )
                conditioning = torch.cat((conditioning, padding), dim=1)
            padded_conditioning.append(conditioning)

            if need_attention_mask:
                mask = metadata.get("attention_mask")
                if mask is None:
                    mask = torch.ones(
                        conditioning.shape[0],
                        length,
                        device=conditioning.device,
                        dtype=torch.long,
                    )
                if mask.shape[1] < max_length:
                    mask = torch.cat(
                        (
                            mask,
                            mask.new_zeros(
                                mask.shape[0],
                                max_length - mask.shape[1],
                            ),
                        ),
                        dim=1,
                    )
                attention_masks.append(mask)

        merged_metadata = dict(entries[0][1])
        if need_attention_mask:
            merged_metadata["attention_mask"] = torch.cat(
                attention_masks,
                dim=0,
            )

        pooled_outputs = [
            metadata.get("pooled_output")
            for _, metadata in entries
        ]
        if pooled_outputs and all(
            torch.is_tensor(pooled)
            for pooled in pooled_outputs
        ):
            merged_metadata["pooled_output"] = torch.cat(
                pooled_outputs,
                dim=0,
            )

        return [[torch.cat(padded_conditioning, dim=0), merged_metadata]]

    @staticmethod
    def _patch_model(model, weight_sets):
        if not any(weight_sets):
            return model

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None or not all(hasattr(block, "attn") for block in blocks):
            raise TypeError(
                "The connected MODEL is not a compatible KREA2 diffusion model."
            )

        transformer_options = model_clone.model_options.get(
            "transformer_options",
            {},
        ).copy()
        transformer_options["krea2_token_weights"] = weight_sets
        model_clone.model_options["transformer_options"] = transformer_options

        for index, block in enumerate(blocks):
            patched_attention = types.MethodType(
                krea2_attn_forward_weight,
                block.attn,
            )
            model_clone.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                patched_attention,
            )
        return model_clone

    def schedule_krea2(
        self,
        model,
        clip,
        folder_path,
        batch_count,
        seed,
        file_extension,
        max_words,
        weighted_phrases,
        strength,
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
                    mode = "random no-repeat" if batch_randomize else "consecutive"
                    print(
                        f"{TAG} Selected {len(selected)} file(s) in {mode} "
                        f"mode using seed {int(seed)}."
                    )
                    prompts = []
                    for path in selected:
                        try:
                            prompt = path.read_text(
                                encoding="utf-8",
                                errors="ignore",
                            ).strip()
                            prompts.append(self.limit_words(prompt, max_words))
                        except Exception as error:
                            print(f"{TAG} Could not read '{path}': {error}")
                            prompts.append("")
                    prompts = [prompt for prompt in prompts if prompt] or [""]
            except Exception as error:
                print(f"{TAG} File loading error: {error}")

        terms = self.parse_weighted_phrases(weighted_phrases)
        if weighted_phrases.strip() and not terms:
            print(
                f"{TAG} WARNING: no valid weights found. Use syntax such as "
                "(phrase:-1) or (phrase:1.5)."
            )

        conditioning_entries = []
        weight_sets = []
        missing_counts = {phrase: 0 for phrase, _ in terms}
        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            if len(conditioning) != 1:
                raise ValueError(
                    "KREA2 file batching does not support active CLIP hook schedules."
                )
            cond_tensor, metadata = conditioning[0]
            conditioning_entries.append((cond_tensor, dict(metadata)))

            weights, missing = self._weights_for_prompt(
                clip,
                tokens,
                cond_tensor.shape[1],
                terms,
                float(strength),
            )
            weight_sets.append(weights)
            for phrase in missing:
                missing_counts[phrase] += 1

        for phrase, missing_count in missing_counts.items():
            if missing_count:
                logging.warning(
                    "%s Shared phrase '%s' was absent from %d/%d prompt(s).",
                    TAG,
                    phrase,
                    missing_count,
                    len(prompts),
                )

        weighted_tokens = sum(len(weights) for weights in weight_sets)
        print(
            f"{TAG} Shared weights: {len(terms)} phrase(s), "
            f"{weighted_tokens} matched token(s), strength={float(strength):g}."
        )
        patched_model = self._patch_model(model, weight_sets)
        conditioning = self._merge_conditioning(conditioning_entries)

        lines = [
            f"Prompt {index + 1} : {prompt}" if print_index else prompt
            for index, prompt in enumerate(prompts)
        ]
        return (
            patched_model,
            conditioning,
            len(prompts),
            "\n\n".join(lines),
        )


NODE_CLASS_MAPPINGS = {
    "CRT_FileBatchPromptSchedulerKREA2": CRT_FileBatchPromptSchedulerKREA2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_FileBatchPromptSchedulerKREA2": (
        "File Batch Prompt Scheduler KREA2 (CRT)"
    ),
}

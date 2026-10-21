"""OpenCode Go bridge for ComfyUI.

Sends a text prompt and/or image batch to one of the models served by the
OpenCode Go plan and returns the answer as a string. Reasoning tokens are
streamed live to the companion Thinking Display node over websocket.
"""

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

GO_ROOT = "https://opencode.ai/inference/go"
DEFAULT_MODEL = "glm-5.3-flash"
THINKING_EVENT = "crt_opencode_go_thinking"
USER_AGENT = "CRT-Nodes/2.24 (ComfyUI; opencode-go-bridge)"
STREAM_BROADCAST_MIN_INTERVAL = 0.08
DEFAULT_TIMEOUT = 600
MAX_IMAGES = 8

EFFORTS = ["auto", "minimal", "low", "medium", "high"]

THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
}

GOOGLE_THINKING_BUDGETS = {
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}

GO_MODELS = [
    "glm-5.3-flash",
    "glm-5.3",
    "glm-5.2",
    "glm-5.1",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
    "minimax-m3",
    "minimax-m3.1-flash-free",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "longcat-2.0",
    "longcat-2.5-preview-free",
    "mimo-v2.6-flash",
    "mimo-v2.6-pro",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "hy4-preview",
    "hy3",
    "step-5-preview",
    "beam-free",
    "gpt-6-luna",
    "gpt-5.6-luna",
    "claude-haiku-5-5",
    "grok-4.7",
    "grok-4.6",
    "gemini-4-argon-contributor",
    "muse-spark-1.3-contributor",
    "muse-spark-1.2-contributor",
    "omen-alpha",
]

RESPONSES_MODELS = {
    "grok-4.7",
    "gpt-6-luna",
    "gpt-5.6-luna",
    "muse-spark-1.3-contributor",
    "muse-spark-1.2-contributor",
}

ANTHROPIC_MODELS = {
    "claude-haiku-5-5",
    "minimax-m3",
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-plus",
}

GOOGLE_MODELS = {"gemini-4-argon-contributor"}

# Pay-as-you-go equivalent prices per 1M tokens for OpenCode Go models
# (threshold, input, output, cached read, cached write). The threshold is the
# prompt-token limit for that tier; None means no limit.
GO_PRICING = {
    "glm-5.3-flash": [(None, 0.15, 0.50, 0.03, None)],
    "glm-5.3": [(None, 1.40, 4.40, 0.26, None)],
    "glm-5.2": [(None, 1.40, 4.40, 0.26, None)],
    "glm-5.1": [(None, 1.40, 4.40, 0.26, None)],
    "kimi-k3": [(None, 3.00, 15.00, 0.30, None)],
    "kimi-k2.7-code": [(None, 0.95, 4.00, 0.19, None)],
    "kimi-k2.6": [(None, 0.95, 4.00, 0.16, None)],
    "longcat-2.0": [(None, 0.30, 1.20, 0.006, None)],
    "longcat-2.5-preview-free": [(None, 0.0, 0.0, 0.0, None)],
    "step-5-preview": [(None, 1.00, 2.70, 0.10, None)],
    "beam-free": [(None, 0.0, 0.0, 0.0, None)],
    "mimo-v2.6-flash": [(None, 0.14, 0.28, 0.0028, None)],
    "mimo-v2.6-pro": [(None, 0.435, 0.87, 0.003625, None)],
    "mimo-v2.5": [(None, 0.14, 0.28, 0.0028, None)],
    "mimo-v2.5-pro": [(None, 0.435, 0.87, 0.003625, None)],
    "minimax-m3.1-flash-free": [(None, 0.0, 0.0, 0.0, None)],
    "minimax-m3": [(None, 0.30, 1.20, 0.06, None)],
    "minimax-m2.7": [(None, 0.30, 1.20, 0.06, None)],
    "minimax-m2.5": [(None, 0.30, 1.20, 0.06, None)],
    "muse-spark-1.3-contributor": [(None, 0.10, 0.20, 0.002, None)],
    "muse-spark-1.2-contributor": [(None, 0.10, 0.20, 0.002, None)],
    "qwen3.8-max": [(None, 2.00, 6.00, 0.25, 2.50)],
    "qwen3.8-flash": [(None, 0.15, 0.47, 0.016, 0.20)],
    "qwen3.7-max": [(None, 2.50, 7.50, 0.50, 3.125)],
    "qwen3.6-plus": [(None, 0.50, 3.00, 0.05, 0.625)],
    "qwen3.7-plus": [
        (256000, 0.40, 1.60, 0.04, 0.50),
        (None, 1.20, 4.80, 0.12, 1.50),
    ],
    "deepseek-v4.1-flash": {
        "off_peak": [(None, 0.15, 0.60, 0.003, None)],
        "peak": [(None, 0.30, 1.20, 0.006, None)],
    },
    "deepseek-v4-pro": {
        "off_peak": [(None, 0.66, 1.98, 0.022, None)],
        "peak": [(None, 1.32, 3.96, 0.044, None)],
    },
    "hy4-preview": [(None, 0.834, 2.501, 0.042, None)],
    "hy3": [(None, 0.14, 0.58, 0.035, None)],
    "grok-4.7": [
        (200000, 2.00, 6.00, 0.50, None),
        (None, 4.00, 12.00, 1.00, None),
    ],
    "grok-4.6": [
        (200000, 2.00, 6.00, 0.50, None),
        (None, 4.00, 12.00, 1.00, None),
    ],
    "gpt-6-luna": [
        (272000, 0.10, 0.50, 0.01, 0.125),
        (None, 0.20, 0.75, 0.02, 0.25),
    ],
    "gpt-5.6-luna": [
        (272000, 0.20, 1.20, 0.02, 0.25),
        (None, 0.40, 1.80, 0.04, 0.50),
    ],
    "claude-haiku-5-5": [
        (100000, 0.10, 0.50, 0.01, 0.125),
        (None, 0.50, 2.50, 0.05, 0.625),
    ],
    "gemini-4-argon-contributor": [(None, 4.00, 20.00, 0.20, None)],
}


def _log(message):
    print(f"[OpenCode Go] {message}", file=sys.stderr)


def _resolve_family(model):
    model = (model or "").strip().lower()
    if model in RESPONSES_MODELS:
        return "responses"
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in GOOGLE_MODELS:
        return "google"
    if model.startswith(("claude", "minimax", "qwen")):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    if model.startswith(("gpt", "grok", "muse")):
        return "responses"
    return "chat"


def _endpoint(family, model):
    if family == "responses":
        return f"{GO_ROOT}/openai/v1/responses"
    if family == "anthropic":
        return f"{GO_ROOT}/anthropic/v1/messages"
    if family == "google":
        return (
            f"{GO_ROOT}/google/v1beta/models/{model}:streamGenerateContent?alt=sse"
        )
    return f"{GO_ROOT}/openai/v1/chat/completions"


def _resolve_api_key(explicit):
    key = str(explicit or "").strip()
    if key:
        return key
    for name in ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_GO_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _session_id(unique_id):
    seed = str(unique_id if unique_id is not None else "crt-opencode-go")
    return f"crt-comfy-{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"


def _headers(family, api_key, session_id, stream=True):
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {api_key}",
        "x-opencode-session": session_id,
    }
    if family == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        headers["x-api-key"] = api_key
    if family == "google":
        headers["x-goog-api-key"] = api_key
    return headers


_SKILLS_CACHE = {}


def _collect_skill_files(raw):
    if os.path.isfile(raw):
        return [raw], os.path.dirname(raw) or "."
    if os.path.isdir(raw):
        files = []
        for base, _dirs, names in os.walk(raw):
            for name in names:
                if name.lower().endswith(".md"):
                    files.append(os.path.join(base, name))
        files.sort()
        return files, raw
    raise RuntimeError(f"Skills path does not exist: {raw}")


def _skills_signature(files):
    signature = []
    for path in files:
        try:
            stat = os.stat(path)
        except OSError:
            signature.append((path, None, None))
            continue
        signature.append((path, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _skills_text(skills_path):
    """Load a single .md file or every .md file under a directory.

    The built block is cached per path and revalidated by file mtime/size, so
    unchanged skills are not re-read, while a changed or replaced skill set
    overrides the previous cached block on the next run.
    """
    raw = str(skills_path or "").strip()
    if not raw:
        return "", 0
    raw = os.path.expanduser(raw)
    files, root = _collect_skill_files(raw)
    cache_key = os.path.normcase(os.path.abspath(raw))
    signature = _skills_signature(files)
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached["signature"] == signature:
        return cached["text"], cached["count"]

    parts = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read().strip()
        except OSError as exc:
            _log(f"WARNING: could not read skill file {path}: {exc}")
            continue
        if not content:
            continue
        rel = (
            os.path.relpath(path, root)
            if os.path.isdir(root)
            else os.path.basename(path)
        )
        parts.append(f"## Skill: {rel}\n\n{content}")
    text = "# Skills\n\n" + "\n\n".join(parts) if parts else ""
    _SKILLS_CACHE[cache_key] = {
        "signature": signature,
        "text": text,
        "count": len(parts),
    }
    return text, len(parts)


def _pick_even_indices(batch_size, count):
    """Evenly spaced indices including the first and last frame."""
    if batch_size <= 1 or count >= batch_size:
        return list(range(batch_size))
    if count <= 1:
        return [0]
    return [int((i * (batch_size - 1)) / (count - 1)) for i in range(count)]


def _image_payloads(image):
    """Convert a ComfyUI IMAGE tensor into base64 PNG parts (max 8).

    Batches larger than 8 frames (for example a short video clip) are evenly
    sampled down to 8 frames, keeping the first and last frame.
    """
    if image is None:
        return []
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow and numpy are required to encode image input.") from exc

    tensor = image.detach().cpu().clamp(0.0, 1.0) if hasattr(image, "detach") else image
    tensor = np.asarray(tensor)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected IMAGE tensor [B,H,W,C] or [H,W,C], got {tensor.shape}.")
    if tensor.shape[0] > MAX_IMAGES:
        indices = _pick_even_indices(tensor.shape[0], MAX_IMAGES)
        _log(
            f"WARNING: {tensor.shape[0]} images supplied, evenly sampled "
            f"{len(indices)} frames."
        )
        tensor = tensor[indices]

    images = []
    for frame in tensor:
        frame = (frame * 255.0).round().clip(0, 255).astype("uint8")
        if frame.shape[-1] == 1:
            pil = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
        elif frame.shape[-1] == 4:
            rgba = Image.fromarray(frame, mode="RGBA")
            pil = Image.new("RGB", rgba.size, (255, 255, 255))
            pil.paste(rgba, mask=rgba.getchannel("A"))
        else:
            pil = Image.fromarray(frame, mode="RGB")
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        raw = buffer.getvalue()
        images.append(
            {
                "mime": "image/png",
                "b64": base64.b64encode(raw).decode("ascii"),
            }
        )
        images[-1]["data_url"] = f"data:image/png;base64,{images[-1]['b64']}"
    return images


def _build_chat_body(model, system_text, prompt, images, effort):
    if images:
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        for item in images:
            user_content.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
    else:
        user_content = prompt
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_content})
    body = {"model": model, "messages": messages, "stream": True}
    if effort and effort != "auto":
        body["reasoning_effort"] = effort
    return body


def _build_responses_body(model, system_text, prompt, images, effort):
    content = []
    if prompt:
        content.append({"type": "input_text", "text": prompt})
    for item in images:
        content.append({"type": "input_image", "image_url": item["data_url"]})
    input_items = []
    if system_text:
        input_items.append(
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]}
        )
    input_items.append({"role": "user", "content": content})
    body = {"model": model, "input": input_items, "stream": True}
    if effort and effort != "auto":
        body["reasoning"] = {"effort": effort, "summary": "auto"}
    return body


def _build_anthropic_body(model, system_text, prompt, images, effort, use_cache=True):
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for item in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item["mime"],
                    "data": item["b64"],
                },
            }
        )
    limit = 8192
    body = {
        "model": model,
        "max_tokens": limit,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }
    if system_text:
        if use_cache:
            body["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            body["system"] = system_text
    budget = THINKING_BUDGETS.get(effort) if effort and effort != "auto" else None
    if budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["max_tokens"] = max(limit, budget + 1024)
    return body


def _build_google_body(model, system_text, prompt, images, effort):
    parts = []
    if prompt:
        parts.append({"text": prompt})
    for item in images:
        parts.append({"inlineData": {"mimeType": item["mime"], "data": item["b64"]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    config = {}
    budget = GOOGLE_THINKING_BUDGETS.get(effort) if effort and effort != "auto" else None
    if budget:
        config["thinkingConfig"] = {"thinkingBudget": budget}
    if config:
        body["generationConfig"] = config
    return body


def _build_body(family, model, system_text, prompt, images, effort, use_cache=True):
    if family == "responses":
        return _build_responses_body(model, system_text, prompt, images, effort)
    if family == "anthropic":
        return _build_anthropic_body(
            model, system_text, prompt, images, effort, use_cache
        )
    if family == "google":
        return _build_google_body(model, system_text, prompt, images, effort)
    return _build_chat_body(model, system_text, prompt, images, effort)


def _compact(text, limit=600):
    text = " ".join(str(text or "").split())
    return text[:limit]


def _open(url, body, headers, timeout):
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenCode Go HTTP {exc.code}: {_compact(detail)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenCode Go ({url}): {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"OpenCode Go request timed out after {timeout}s") from exc


def _iter_sse(response):
    buffer = ""
    while True:
        try:
            chunk = response.read(8192)
        except TimeoutError as exc:
            raise RuntimeError("OpenCode Go stream timed out") from exc
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            yield block
    if buffer.strip():
        yield buffer


def _parse_block(block):
    event = None
    data_lines = []
    for line in block.split("\n"):
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    return event, "\n".join(data_lines)


def _merge_usage(state, usage):
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, (int, float, dict)):
            state["usage"][key] = value


def _consume_openai_chat(obj, state):
    if isinstance(obj.get("usage"), dict):
        _merge_usage(state, obj["usage"])
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = choice.get("message")
    if not isinstance(delta, dict):
        delta = {}
    reasoning = delta.get("reasoning_content", delta.get("reasoning", ""))
    content = delta.get("content", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    if not isinstance(content, str):
        content = ""
    if not content and not reasoning and isinstance(choice.get("text"), str):
        content = choice["text"]
    return reasoning, content


def _consume_responses(obj, state):
    kind = obj.get("type") or ""
    if kind == "response.output_text.delta":
        delta = obj.get("delta")
        return "", delta if isinstance(delta, str) else ""
    if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        delta = obj.get("delta")
        return (delta if isinstance(delta, str) else ""), ""
    if kind == "response.completed":
        _merge_usage(state, (obj.get("response") or {}).get("usage"))
        state["done"] = True
        return "", ""
    if kind in ("error", "response.failed"):
        message = obj.get("message") or (obj.get("response") or {}).get("error")
        raise RuntimeError(f"OpenCode Go stream error: {_compact(message)}")
    return "", ""


def _consume_anthropic(obj, event, state):
    kind = event or obj.get("type") or ""
    if kind == "content_block_delta":
        delta = obj.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            return "", delta.get("text", "") or ""
        if dtype == "thinking_delta":
            return delta.get("thinking", "") or "", ""
        return "", ""
    if kind == "message_start":
        _merge_usage(state, (obj.get("message") or {}).get("usage"))
        return "", ""
    if kind == "message_delta":
        _merge_usage(state, obj.get("usage"))
        return "", ""
    if kind == "error":
        raise RuntimeError(f"OpenCode Go stream error: {_compact(obj.get('error'))}")
    return "", ""


def _consume_google(obj, state):
    _merge_usage(state, obj.get("usageMetadata"))
    reasoning_parts = []
    content_parts = []
    for candidate in obj.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if not isinstance(text, str):
                continue
            if part.get("thought"):
                reasoning_parts.append(text)
            else:
                content_parts.append(text)
    return "".join(reasoning_parts), "".join(content_parts)


def _consume(family, obj, event, state):
    if family == "responses":
        return _consume_responses(obj, state)
    if family == "anthropic":
        return _consume_anthropic(obj, event, state)
    if family == "google":
        return _consume_google(obj, state)
    return _consume_openai_chat(obj, state)


def _usage_stats(state, elapsed):
    usage = state["usage"]
    family = state.get("family")
    cached_tokens = 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens") or 0
    if usage.get("cache_read_input_tokens") is not None:
        cached_tokens = usage.get("cache_read_input_tokens") or 0
    if usage.get("cachedContentTokenCount") is not None:
        cached_tokens = usage.get("cachedContentTokenCount") or 0
    cache_write_tokens = usage.get("cache_creation_input_tokens") or 0

    if family == "anthropic":
        uncached = usage.get("input_tokens") or 0
        prompt_tokens = uncached + cached_tokens + cache_write_tokens
    else:
        prompt_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )

    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidatesTokenCount")
        or 0
    )
    total_tokens = (
        usage.get("total_tokens")
        or usage.get("totalTokenCount")
        or (prompt_tokens + completion_tokens)
    )
    if not completion_tokens:
        completion_tokens = state.get("live_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
    tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens else 0.0
    return {
        "tps": tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "elapsed": elapsed,
        "live": False,
    }


def _deepseek_peak(now=None):
    now = now or time.gmtime()
    if now.tm_wday >= 5:
        return False
    return 1 <= now.tm_hour < 4 or 6 <= now.tm_hour < 10


def _price_tier(model, prompt_tokens):
    spec = GO_PRICING.get(model)
    if spec is None:
        return None
    if isinstance(spec, dict):
        spec = spec["peak" if _deepseek_peak() else "off_peak"]
    for threshold, price_in, price_out, price_cached, price_write in spec:
        if threshold is None or prompt_tokens <= threshold:
            return price_in, price_out, price_cached, price_write
    return None


def _estimate_cost(model, stats):
    prompt_tokens = stats.get("prompt_tokens", 0) or 0
    tier = _price_tier(model, prompt_tokens)
    if tier is None:
        return None
    price_in, price_out, price_cached, price_write = tier
    cached_tokens = min(stats.get("cached_tokens", 0) or 0, prompt_tokens)
    write_tokens = min(
        stats.get("cache_write_tokens", 0) or 0,
        max(prompt_tokens - cached_tokens, 0),
    )
    uncached_tokens = max(prompt_tokens - cached_tokens - write_tokens, 0)
    cached_price = price_in if price_cached is None else price_cached
    write_price = price_in if price_write is None else price_write
    completion_tokens = stats.get("completion_tokens", 0) or 0
    return (
        uncached_tokens * price_in
        + cached_tokens * cached_price
        + write_tokens * write_price
        + completion_tokens * price_out
    ) / 1_000_000.0


def _stream(url, body, headers, family, timeout, on_delta):
    state = {
        "reasoning": [],
        "content": [],
        "usage": {},
        "live_tokens": 0,
        "done": False,
        "family": family,
    }
    start = time.perf_counter()
    response = _open(url, body, headers, timeout)
    try:
        for block in _iter_sse(response):
            event, data = _parse_block(block)
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "error" in obj and not any(
                key in obj for key in ("choices", "candidates", "type", "delta", "usage")
            ):
                raise RuntimeError(
                    f"OpenCode Go stream error: {_compact(obj.get('error'))}"
                )
            reasoning, content = _consume(family, obj, event, state)
            if reasoning:
                state["reasoning"].append(reasoning)
            if content:
                state["content"].append(content)
            if (reasoning or content) and on_delta is not None:
                state["live_tokens"] += 1
                elapsed = time.perf_counter() - start
                on_delta(
                    "".join(state["reasoning"]),
                    "".join(state["content"]),
                    {
                        "tps": state["live_tokens"] / elapsed if elapsed > 0 else 0.0,
                        "completion_tokens": state["live_tokens"],
                        "elapsed": elapsed,
                        "live": True,
                    },
                )
    finally:
        try:
            response.close()
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    content = "".join(state["content"])
    reasoning = "".join(state["reasoning"])
    if not content and not reasoning:
        raise RuntimeError("OpenCode Go returned an empty response")
    stats = _usage_stats(state, elapsed)
    if on_delta is not None:
        try:
            on_delta(reasoning, content, stats)
        except Exception:
            pass
    return content, reasoning, stats


def _broadcast(source_node_id, thinking, answer, done, error=None, stats=None, model=None):
    try:
        from server import PromptServer

        server = PromptServer.instance
        if server is None:
            return
        try:
            sid = server.client_id
        except Exception:
            sid = None
        prompt_id = None
        try:
            from comfy_execution.utils import get_executing_context

            ctx = get_executing_context()
            if ctx is not None:
                prompt_id = ctx.prompt_id
        except Exception:
            prompt_id = None
        if prompt_id is None:
            prompt_id = getattr(server, "last_prompt_id", None)
        data = {
            "source_node": str(source_node_id) if source_node_id is not None else None,
            "prompt_id": prompt_id,
            "thinking": thinking or "",
            "answer": answer or "",
            "done": bool(done),
            "model": model,
        }
        if isinstance(stats, dict) and stats:
            clean = {}
            for key in (
                "tps",
                "completion_tokens",
                "prompt_tokens",
                "total_tokens",
                "elapsed",
                "live",
                "cost",
            ):
                if stats.get(key) is not None:
                    clean[key] = stats[key]
            if clean:
                data["stats"] = clean
        if error:
            data["error"] = str(error)
        server.send_sync(THINKING_EVENT, data, sid)
    except Exception:
        pass


class CRT_OpenCodeGoBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Text prompt sent to the selected OpenCode Go model.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "OpenCode Go API key. Leave empty to read OPENCODE_API_KEY "
                            "or OPENCODE_GO_API_KEY from the environment."
                        ),
                    },
                ),
                "model": (
                    GO_MODELS,
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "Model served by the OpenCode Go plan.",
                    },
                ),
                "effort": (
                    EFFORTS,
                    {
                        "default": "auto",
                        "tooltip": (
                            "Reasoning effort. auto lets the provider decide; higher "
                            "values give the model more thinking budget."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional image batch. Batches larger than 8 frames are "
                            "evenly sampled down to 8 (first and last included). The "
                            "chosen model must support vision input."
                        ),
                    },
                ),
                "skills_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Path to a .md skill file, or to a folder whose .md files "
                            "(recursively) are appended to the system prompt."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "CRT/LLM"
    DESCRIPTION = (
        "OpenCode Go bridge. Sends a prompt and optional image batch to a Go-plan "
        "model (chat, Responses, Anthropic, or Google endpoint picked from the model "
        "id), streams thinking to the companion display node, and returns the answer."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(
        self,
        prompt,
        api_key,
        model,
        effort,
        image=None,
        skills_path="",
        unique_id=None,
    ):
        key = _resolve_api_key(api_key)
        if not key:
            raise RuntimeError(
                "OpenCode Go API key is required. Paste it into the api_key widget or "
                "set OPENCODE_API_KEY."
            )

        chosen = str(model or "").strip()
        if not chosen:
            raise RuntimeError("No OpenCode Go model selected.")
        family = _resolve_family(chosen)
        url = _endpoint(family, chosen)

        prompt = "" if prompt is None else str(prompt)
        if not prompt.strip() and image is None:
            raise RuntimeError("Provide a prompt or connect an image.")

        skills_text, skills_count = _skills_text(skills_path)
        system_text = skills_text
        if skills_count:
            _log(f"Loaded {skills_count} skill .md file(s)")

        images = _image_payloads(image)
        if images:
            _log(f"Attached {len(images)} image(s)")

        effort = str(effort or "auto").strip().lower()
        body = _build_body(
            family,
            chosen,
            system_text,
            prompt.strip(),
            images,
            effort,
        )
        headers = _headers(family, key, _session_id(unique_id))
        _log(
            f"model={chosen}, family={family}, effort={effort}, "
            f"skills={skills_count}, images={len(images)}"
        )

        stream_state = {"last_emit": 0.0}

        def on_delta(reasoning_piece, content_piece, stats):
            now = time.monotonic()
            if stats.get("live") and (
                now - stream_state["last_emit"] < STREAM_BROADCAST_MIN_INTERVAL
            ):
                return
            stream_state["last_emit"] = now
            _broadcast(
                unique_id,
                reasoning_piece,
                content_piece,
                not stats.get("live", False),
                stats=stats,
                model=chosen,
            )

        _broadcast(unique_id, "", "", False, model=chosen)

        try:
            content, reasoning, stats = _stream(
                url, body, headers, family, DEFAULT_TIMEOUT, on_delta
            )
        except RuntimeError as exc:
            retryable = "HTTP 400" in str(exc) or "HTTP 422" in str(exc)
            if retryable and (
                effort != "auto" or (family == "anthropic" and system_text)
            ):
                _log(
                    f"WARNING: {exc}; retrying without optional request fields."
                )
                body = _build_body(
                    family,
                    chosen,
                    system_text,
                    prompt.strip(),
                    images,
                    "auto",
                    use_cache=False,
                )
                content, reasoning, stats = _stream(
                    url, body, headers, family, DEFAULT_TIMEOUT, on_delta
                )
            else:
                _broadcast(unique_id, "", "", True, error=exc, model=chosen)
                raise

        cost = _estimate_cost(chosen, stats)
        if cost is not None:
            stats["cost"] = cost
            _log(f"estimated cost: ${cost:.6f}")
        _broadcast(unique_id, reasoning, content, True, stats=stats, model=chosen)

        _log(
            f"response: {len(content)} chars, thinking: {len(reasoning)} chars, "
            f"tokens={stats.get('completion_tokens')}, {stats.get('elapsed', 0):.2f}s"
        )
        if not content and not reasoning:
            raise RuntimeError("OpenCode Go returned an empty response.")
        return (content if content else reasoning,)


NODE_CLASS_MAPPINGS = {
    "CRT_OpenCodeGoBridge": CRT_OpenCodeGoBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_OpenCodeGoBridge": "OpenCode Go Bridge (CRT)",
}

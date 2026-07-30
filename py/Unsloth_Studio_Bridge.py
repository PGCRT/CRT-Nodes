import os
import re
import json
import sys
import base64
import io
import urllib.request
import urllib.error
import urllib.parse
import glob


HTTP_TIMEOUT = 10
CONTEXT_SAFETY_TOKENS = 32
DEFAULT_UNSLOTH_STUDIO_URL = "http://127.0.0.1:8888"


def _request_json(url, body=None, timeout=HTTP_TIMEOUT, retries=1):
    data = None
    method = "GET"
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    last_exc = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"llama-server returned invalid JSON for {url}: {payload[:500]}"
                ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = error_body or exc.reason or "no response body"
            raise RuntimeError(
                f"llama-server HTTP {exc.code} for {url}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach llama-server at {url}: {exc.reason}") from exc
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            last_exc = exc
            if attempt < retries:
                import time
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"llama-server closed the connection for {url}. The server process may "
                "have crashed or restarted; reload the model and retry."
            ) from exc


def _find_latest_server_log():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    log_dir = os.path.join(unsloth_home, "logs", "server")
    if not os.path.isdir(log_dir):
        return None
    log_files = glob.glob(os.path.join(log_dir, "server-*.log"))
    if not log_files:
        return None
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def _find_llama_server_logs():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    log_dir = os.path.join(unsloth_home, "logs", "llama-server")
    if not os.path.isdir(log_dir):
        return []
    log_files = glob.glob(os.path.join(log_dir, "llama-*-port-*-try*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files


def _parse_ports_from_server_log(log_path):
    """Return list of ports in chronological order from server log."""
    if log_path is None:
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []
    ports = []
    for m in re.finditer(r'--port\s+(\d+)', content):
        ports.append(int(m.group(1)))
    for m in re.finditer(r'ready on port (\d+)', content):
        ports.append(int(m.group(1)))
    return ports


def _parse_port_from_llama_log(log_path):
    if log_path is None:
        return None
    name = os.path.basename(log_path)
    m = re.search(r'-port-(\d+)-', name)
    if m:
        return int(m.group(1))
    return None


def _parse_model_name_from_log(log_path):
    if log_path is None:
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None
    m = re.search(r'Detected local GGUF model:.*[\\/]([^\\/]+\.gguf)', content)
    if m:
        return m.group(1)
    m = re.search(r'llama-server ready on port \d+ for model [\'"]([^\'"]+)[\'"]', content)
    if m:
        return m.group(1)
    return None


def _normalize_server_url(value):
    raw = str(value or DEFAULT_UNSLOTH_STUDIO_URL).strip()
    if not raw:
        raw = DEFAULT_UNSLOTH_STUDIO_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "unsloth_server_url must be an HTTP(S) URL, for example "
            "http://127.0.0.1:8888."
        )
    return raw.rstrip("/")


def _is_server_alive(base_url, timeout=2):
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_port_alive(port, timeout=2):
    return _is_server_alive(f"http://127.0.0.1:{port}", timeout=timeout)


def _resolve_llama_server(configured_url):
    configured_base = _normalize_server_url(configured_url)
    default_base = _normalize_server_url(DEFAULT_UNSLOTH_STUDIO_URL)

    if configured_base != default_base and _is_server_alive(configured_base):
        return configured_base, "configured llama-server URL"

    llama_port = _detect_llama_port()
    if llama_port is None:
        if configured_base != default_base:
            raise RuntimeError(
                f"Could not reach the configured llama-server at {configured_base}, "
                "and no active port could be detected from Unsloth Studio logs."
            )
        raise RuntimeError(
            "Could not detect llama-server port from Unsloth Studio logs. "
            "Make sure Unsloth Studio is running and a model is loaded."
        )

    parsed = urllib.parse.urlparse(configured_base)
    hostname = parsed.hostname or "127.0.0.1"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    detected_base = f"{parsed.scheme}://{hostname}:{llama_port}"
    if not _is_server_alive(detected_base):
        raise RuntimeError(
            f"llama-server at {detected_base} is not responding. The model may "
            "have been unloaded or the port changed; reload it in Unsloth Studio."
        )
    return detected_base, f"Studio log discovery (port {llama_port})"


def _detect_llama_port():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")

    # Strategy 1: check ports from latest server log in reverse chronological order
    server_log = _find_latest_server_log()
    server_ports = _parse_ports_from_server_log(server_log)
    for port in reversed(server_ports):
        if _is_port_alive(port):
            return port

    # Strategy 2: check recent llama-server log files by modification time
    for log_path in _find_llama_server_logs():
        port = _parse_port_from_llama_log(log_path)
        if port and _is_port_alive(port):
            return port

    # Strategy 3: fallback to last port from server log even if not alive
    if server_ports:
        return server_ports[-1]
    return None


def _find_inference_defaults(log_path):
    if log_path:
        unsloth_home = os.path.dirname(os.path.dirname(os.path.dirname(log_path)))
        candidate = os.path.join(
            unsloth_home,
            "unsloth_studio", "Lib", "site-packages", "studio", "backend", "assets", "configs",
            "inference_defaults.json",
        )
        if os.path.isfile(candidate):
            return candidate
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    candidate = os.path.join(
        unsloth_home,
        "unsloth_studio", "Lib", "site-packages", "studio", "backend", "assets", "configs",
        "inference_defaults.json",
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_inference_defaults(log_path):
    path = _find_inference_defaults(log_path)
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _match_model_family(model_name, defaults):
    if not model_name:
        return None
    name_lower = model_name.lower()
    for pattern in defaults.get("patterns", []):
        if pattern.lower() in name_lower:
            return pattern
    return None


def _get_model_params(log_path):
    defaults = _load_inference_defaults(log_path)
    model_name = _parse_model_name_from_log(log_path)
    family = _match_model_family(model_name, defaults)
    params = defaults.get("families", {}).get(family, {}) if family else {}
    return {
        "model_name": model_name,
        "family": family,
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.95),
        "top_k": params.get("top_k", 20),
        "min_p": params.get("min_p", 0.0),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
        "presence_penalty": params.get("presence_penalty", 0.0),
    }


def _get_server_state(base_url):
    props = _request_json(f"{base_url}/props")
    generation = props.get("default_generation_settings", {})
    server_params = generation.get("params", {})
    context_length = generation.get("n_ctx")
    try:
        context_length = int(context_length)
    except (TypeError, ValueError):
        context_length = None
    return {
        "model_name": props.get("model_alias"),
        "context_length": context_length,
        "temperature": server_params.get("temperature"),
        "top_p": server_params.get("top_p"),
        "top_k": server_params.get("top_k"),
        "min_p": server_params.get("min_p"),
    }


def _merge_active_params(fallback, active):
    merged = dict(fallback)
    for key in ("temperature", "top_p", "top_k", "min_p"):
        value = active.get(key)
        if value is not None:
            merged[key] = value
    if active.get("model_name"):
        merged["model_name"] = active["model_name"]
    return merged


def _image_tensor_to_data_urls(image):
    if image is None:
        return []
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to encode image inputs for Unsloth LLM.") from exc

    if not hasattr(image, "detach"):
        raise RuntimeError("Expected ComfyUI IMAGE tensor input.")

    tensor = image.detach().cpu().clamp(0.0, 1.0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise RuntimeError(
            f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {tuple(tensor.shape)}."
        )

    data_urls = []
    for frame in tensor.numpy():
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise RuntimeError(
                f"Expected image frame with 1, 3, or 4 channels, got shape {frame.shape}."
            )
        frame = (frame * 255.0).round().clip(0, 255).astype("uint8")
        if frame.shape[-1] == 1:
            pil_image = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
        elif frame.shape[-1] == 4:
            rgba = Image.fromarray(frame, mode="RGBA")
            pil_image = Image.new("RGB", rgba.size, (255, 255, 255))
            pil_image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            pil_image = Image.fromarray(frame, mode="RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_urls.append(f"data:image/png;base64,{encoded}")
    return data_urls


def _build_user_content(prompt, image):
    text = prompt.strip() if isinstance(prompt, str) else ""
    image_urls = _image_tensor_to_data_urls(image)
    if not image_urls:
        return text, 0

    content = []
    if text:
        content.append({"type": "text", "text": text})
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content, len(image_urls)


def _messages_for_token_count(messages):
    text_only = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])
            message = dict(message)
            message["content"] = "\n".join(text_parts)
        text_only.append(message)
    return text_only


def _count_chat_tokens(base_url, messages, disable_thinking):
    template_body = {
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": not disable_thinking},
    }
    rendered = _request_json(
        f"{base_url}/apply-template",
        template_body,
    ).get("prompt")
    if not isinstance(rendered, str):
        raise RuntimeError("llama-server /apply-template did not return a prompt")
    tokenized = _request_json(
        f"{base_url}/tokenize",
        {
            "content": rendered,
            "add_special": True,
            "parse_special": True,
        },
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama-server /tokenize did not return a token list")
    return len(tokens)


def _validate_context_budget(
    context_length,
    prompt_tokens,
    safety_tokens=CONTEXT_SAFETY_TOKENS,
):
    if context_length is None or context_length <= 0:
        return None
    available = context_length - prompt_tokens - safety_tokens
    if available < 1:
        raise RuntimeError(
            f"Prompt uses {prompt_tokens} tokens, but the active llama-server context "
            f"is only {context_length} tokens ({safety_tokens} reserved). Reload the "
            "model with a larger context or shorten the prompt."
        )
    return available


def _chat_completion(base_url, messages, seed, params, include_reasoning, disable_thinking, clear_prompt_cache=True):
    import time
    body = {
        "model": "default",
        "messages": messages,
        "seed": seed,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "top_k": params.get("top_k"),
        "min_p": params.get("min_p"),
        "stream": False,
        "cache_prompt": False,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body = {k: v for k, v in body.items() if v is not None}
    start = time.perf_counter()
    result = _request_json(
        f"{base_url}/v1/chat/completions",
        body,
        timeout=300,
    )
    elapsed = time.perf_counter() - start
    choices = result.get("choices")
    if not choices:
        raise RuntimeError("No response from model")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[Unsloth Studio Bridge] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, "
        f"time={elapsed:.2f}s, speed={tps:.2f} tok/s",
        file=sys.stderr,
    )
    if clear_prompt_cache:
        try:
            erased = _request_json(
                f"{base_url}/slots/0?action=erase",
                timeout=HTTP_TIMEOUT,
            )
            print(f"[Unsloth Studio Bridge] Cleared slot cache: {erased}", file=sys.stderr)
        except Exception as exc:
            print(f"[Unsloth Studio Bridge] WARNING: could not clear slot cache: {exc}", file=sys.stderr)
    if include_reasoning and reasoning:
        if content:
            return content + "\n\n[Reasoning]\n" + reasoning
        return reasoning
    return content


class UnslothLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional image batch. The model loaded in Unsloth Studio must support vision input.",
                    },
                ),
                "unsloth_server_url": (
                    "STRING",
                    {
                        "default": DEFAULT_UNSLOTH_STUDIO_URL,
                        "multiline": False,
                        "tooltip": (
                            "Leave the default to discover Studio's active local llama-server "
                            "port from its logs. Enter a reachable non-default llama-server "
                            "base URL to bypass discovery. Studio must have a model loaded; "
                            "image input requires a vision-capable model."
                        ),
                    },
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "disable_thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Request a direct answer by disabling supported model thinking mode. Models that ignore this chat-template option are unaffected.",
                    },
                ),
                "include_reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Append reasoning_content returned by the server to the response output when the loaded model provides it.",
                    },
                ),
                "clear_prompt_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Erase llama-server slot 0 after generation. This reduces prompt retention but prevents reuse of that prompt cache.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "CRT/LLM"
    DESCRIPTION = (
        "Connects ComfyUI to the model currently loaded in Unsloth Studio. "
        "Unsloth Studio must remain open; image input requires a vision model."
    )

    def generate(self, prompt, seed, unsloth_server_url=DEFAULT_UNSLOTH_STUDIO_URL, system_prompt="", disable_thinking=True, include_reasoning=False, clear_prompt_cache=True, image=None):
        if (not prompt or not prompt.strip()) and image is None:
            print("[Unsloth Studio Bridge][WARN] Prompt is empty", file=sys.stderr)
        print(
            f"[Unsloth Studio Bridge][INFO] Request prepared: "
            f"prompt_chars={len(prompt or '')}, image_connected={image is not None}",
            file=sys.stderr,
        )
        llama_base, connection_source = _resolve_llama_server(unsloth_server_url)
        server_log = _find_latest_server_log()
        fallback_params = _get_model_params(server_log)
        server_state = _get_server_state(llama_base)
        params = _merge_active_params(fallback_params, server_state)
        print(
            f"[Unsloth Studio Bridge][INFO] Connected via {connection_source}: "
            f"{llama_base}, model={params['model_name']}, family={params['family']}",
            file=sys.stderr,
        )
        print(
            f"[Unsloth Studio Bridge] Active server: context={server_state['context_length']}, "
            f"temp={params['temperature']}, top_p={params['top_p']}, "
            f"top_k={params['top_k']}, min_p={params['min_p']}",
            file=sys.stderr,
        )
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        user_content, image_count = _build_user_content(prompt, image)
        if image_count:
            print(f"[Unsloth Studio Bridge] Attached {image_count} image(s) to request", file=sys.stderr)
        messages.append({"role": "user", "content": user_content})
        prompt_tokens = _count_chat_tokens(
            llama_base,
            _messages_for_token_count(messages),
            disable_thinking,
        )
        available_tokens = _validate_context_budget(
            server_state["context_length"],
            prompt_tokens,
        )
        if image_count:
            print(
                "[Unsloth Studio Bridge] Token budget counts text prompt only; image tokens are handled by llama-server.",
                file=sys.stderr,
            )
        print(
            f"[Unsloth Studio Bridge] Token budget: prompt={prompt_tokens}, "
            f"context={server_state['context_length']}, available={available_tokens}, "
            "completion_limit=server-native",
            file=sys.stderr,
        )
        print(
            f"[Unsloth Studio Bridge] Sending request (seed={seed}, max_tokens=omitted, "
            f"disable_thinking={disable_thinking})",
            file=sys.stderr,
        )
        response = _chat_completion(
            llama_base,
            messages,
            seed,
            params,
            include_reasoning,
            disable_thinking,
            clear_prompt_cache,
        )
        print(
            f"[Unsloth Studio Bridge][INFO] Response received: {len(response)} chars",
            file=sys.stderr,
        )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "UnslothLLM": UnslothLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnslothLLM": "Unsloth Studio Bridge (CRT)",
}

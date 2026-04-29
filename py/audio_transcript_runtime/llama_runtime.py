import gc
import inspect
import json
import os

import folder_paths
import torch
from llama_cpp import Llama, llama_backend_free

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
except ImportError:
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    MTMDChatHandler = Llava15ChatHandler

import llama_cpp.llama_chat_format as lcf

from .download import download_file


TRANSLATION_REPO = "prithivMLmods/Gliese-Qwen3.5-4B-Abliterated-Caption"
TRANSLATION_MODEL = "Gliese-Qwen3.5-4B-Abliterated-Caption.Q8_0.gguf"
TRANSLATION_MMPROJ = "Gliese-Qwen3.5-4B-Abliterated-Caption.mmproj-q8_0.gguf"
TRANSLATION_CHAT_FORMAT = "vision-qwen35"

_GLOBAL_LLM = None
FIXED_TRANSLATION_SEED = 0

VISION_HANDLERS = {}
for name, obj in inspect.getmembers(lcf):
    if inspect.isclass(obj) and issubclass(obj, MTMDChatHandler):
        vision_name = f"vision-{name.lower().replace('chathandler', '')}"
        VISION_HANDLERS[vision_name] = obj


def _text_encoder_dir():
    try:
        model_dirs = folder_paths.get_folder_paths("text_encoders")
        if model_dirs:
            return model_dirs[0]
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, "text_encoders")


def ensure_translation_assets():
    model_dir = _text_encoder_dir()
    model_path = os.path.join(model_dir, TRANSLATION_MODEL)
    mmproj_path = os.path.join(model_dir, TRANSLATION_MMPROJ)

    download_file(
        f"https://huggingface.co/{TRANSLATION_REPO}/resolve/main/GGUF/{TRANSLATION_MODEL}?download=true",
        model_path,
        TRANSLATION_MODEL,
    )
    download_file(
        f"https://huggingface.co/{TRANSLATION_REPO}/resolve/main/GGUF/{TRANSLATION_MMPROJ}?download=true",
        mmproj_path,
        TRANSLATION_MMPROJ,
    )
    return model_path, mmproj_path


def translation_system_prompt(target_language):
    return (
        "You are a translation engine.\n"
        f'Translate the input text to "{target_language}".\n'
        "Output only the translation.\n"
        "Keep the exact formatting.\n"
        "Do not add anything before or after the translation.\n"
        "Never output reasoning, <think> tags, explanations, or quotes."
    )


def _clean_translation_output(text):
    result = str(text or "").strip()
    if not result:
        return ""

    think_idx = result.lower().find("<think>")
    if think_idx != -1:
        result = result[:think_idx].strip()

    prefixes = (
        "Thinking Process:",
        "Reasoning:",
        "Thought process:",
        "Let's translate",
        "1. **Analyze the Request**",
    )
    for prefix in prefixes:
        if result.startswith(prefix):
            raise RuntimeError(
                "Translation model returned reasoning instead of translation-only output."
            )

    if len(result) >= 2 and result[0] == '"' and result[-1] == '"':
        result = result[1:-1].strip()

    return result


def cleanup_llm(mode="full_cleanup"):
    global _GLOBAL_LLM

    if mode == "persistent":
        return

    if _GLOBAL_LLM is not None:
        if hasattr(_GLOBAL_LLM, "chat_handler") and _GLOBAL_LLM.chat_handler is not None:
            try:
                chat_handler = _GLOBAL_LLM.chat_handler
                if hasattr(chat_handler, "_exit_stack") and chat_handler._exit_stack is not None:
                    try:
                        chat_handler._exit_stack.close()
                    except Exception:
                        pass

                for attr_name in ("clip_model", "_clip_model", "mmproj", "_mmproj", "clf"):
                    if hasattr(chat_handler, attr_name):
                        attr = getattr(chat_handler, attr_name)
                        if attr is not None:
                            if hasattr(attr, "close"):
                                try:
                                    attr.close()
                                except Exception:
                                    pass
                            elif hasattr(attr, "__del__"):
                                try:
                                    attr.__del__()
                                except Exception:
                                    pass
                            setattr(chat_handler, attr_name, None)
                if hasattr(chat_handler, "close"):
                    try:
                        chat_handler.close()
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            _GLOBAL_LLM.close()
        except Exception:
            pass
        _GLOBAL_LLM = None
        gc.collect()

    if mode in ["backend_free", "full_cleanup"]:
        try:
            llama_backend_free()
        except Exception:
            pass
        gc.collect()

    if mode == "full_cleanup" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()


def _build_vision_chat_handler(mmproj_path):
    handler_class = VISION_HANDLERS.get(TRANSLATION_CHAT_FORMAT, MTMDChatHandler)
    base_sig = inspect.signature(MTMDChatHandler)
    handler_params = set(base_sig.parameters.keys())
    handler_sig = inspect.signature(handler_class)
    handler_params.update(handler_sig.parameters.keys())

    handler_kwargs = {
        "clip_model_path": mmproj_path,
        "verbose": False,
    }
    if "use_gpu" in handler_params:
        handler_kwargs["use_gpu"] = torch.cuda.is_available()
    if "enable_thinking" in handler_params:
        handler_kwargs["enable_thinking"] = False
    if "force_reasoning" in handler_params:
        handler_kwargs["force_reasoning"] = False
    if "add_vision_id" in handler_params:
        handler_kwargs["add_vision_id"] = False
    return handler_class(**handler_kwargs)


def translate_text(text, target_language):
    global _GLOBAL_LLM

    clean_text = str(text or "").strip()
    if not clean_text:
        return ""

    model_path, mmproj_path = ensure_translation_assets()
    llama_kwargs = {
        "model_path": model_path,
        "n_gpu_layers": -1,
        "n_ctx": 2048,
        "n_threads": -1,
        "n_threads_batch": -1,
        "n_batch": 2048,
        "n_ubatch": 512,
        "main_gpu": 0 if torch.cuda.is_available() else -1,
        "offload_kqv": True,
        "numa": True,
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False,
        "chat_handler": _build_vision_chat_handler(mmproj_path),
    }

    messages = [
        {"role": "system", "content": translation_system_prompt(target_language)},
        {"role": "user", "content": f'"{clean_text}"'},
    ]

    try:
        cleanup_llm("close")
        torch.manual_seed(FIXED_TRANSLATION_SEED)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(FIXED_TRANSLATION_SEED)
                torch.cuda.manual_seed_all(FIXED_TRANSLATION_SEED)
            except Exception:
                pass
        _GLOBAL_LLM = Llama(**llama_kwargs)
        response = _GLOBAL_LLM.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
            top_k=2,
            repeat_penalty=1.1,
            present_penalty=0.0,
            frequency_penalty=0.0,
            min_p=0.0,
            seed=FIXED_TRANSLATION_SEED,
            response_format={"type": "text"},
            stop=["<think>", "Thinking Process:", "Reasoning:"],
        )
        if not response or "choices" not in response or not response["choices"]:
            raise RuntimeError("No response generated by the model")
        return _clean_translation_output(response["choices"][0]["message"]["content"])
    finally:
        cleanup_llm("full_cleanup")

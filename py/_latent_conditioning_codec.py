import importlib
import json

import torch
from safetensors import safe_open

from comfy.nested_tensor import NestedTensor

METADATA_KEY = "crt_latent_conditioning"
FORMAT_VERSION = 2

_CLIP_VISION_OUTPUT_CLASS = "comfy.clip_vision.Output"


def encode_latent_conditioning(latent, conditioning):
    """Build the tensor dict and metadata dict for one combined .safetensors file."""
    tensors = {}
    seen_ids = {}
    part_counter = 0

    def add_tensor(key, tensor):
        source_key = seen_ids.get(id(tensor))
        if source_key is not None:
            tensors[key] = tensors[source_key].clone()
            return
        stored = tensor.detach().cpu()
        if not stored.is_contiguous():
            stored = stored.contiguous()
        tensors[key] = stored
        seen_ids[id(tensor)] = key

    def encode_structure(value, path):
        """Tag a value as a JSON tree; tensors are stored under generated safetensors keys."""
        nonlocal part_counter
        if isinstance(value, torch.Tensor):
            part_counter += 1
            key = f"{path}__part{part_counter}"
            add_tensor(key, value)
            return {"t": "tensor", "k": key}
        if isinstance(value, NestedTensor):
            items = [encode_structure(t, path) for t in value.tensors]
            if any(item is None for item in items):
                return None
            return {"t": "nested_tensor", "items": items}
        if isinstance(value, bool):
            return {"t": "bool", "v": value}
        if isinstance(value, int):
            return {"t": "int", "v": value}
        if isinstance(value, float):
            return {"t": "float", "v": value}
        if isinstance(value, str):
            return {"t": "str", "v": value}
        if value is None:
            return {"t": "none"}
        if isinstance(value, (list, tuple)):
            items = [encode_structure(item, path) for item in value]
            if any(item is None for item in items):
                return None
            return {"t": "tuple" if isinstance(value, tuple) else "list", "items": items}
        if isinstance(value, dict):
            items = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    return None
                encoded = encode_structure(v, path)
                if encoded is None:
                    return None
                items[k] = encoded
            return {"t": "dict", "items": items}
        if f"{type(value).__module__}.{type(value).__name__}" == _CLIP_VISION_OUTPUT_CLASS:
            attrs = {}
            for k, v in vars(value).items():
                encoded = encode_structure(v, path)
                if encoded is None:
                    return None
                attrs[k] = encoded
            return {"t": "object", "class": _CLIP_VISION_OUTPUT_CLASS, "attrs": attrs}
        return None

    meta = {"version": FORMAT_VERSION, "cond": [], "latent_tensors": [], "latent_values": {}}

    if isinstance(latent, dict):
        samples = latent.get("samples")
        extras = [(k, v) for k, v in latent.items() if k != "samples"]
    else:
        samples = latent
        extras = []
    if isinstance(samples, torch.Tensor):
        add_tensor("latent", samples)
    else:
        structure = encode_structure(samples, "latent")
        if structure is None:
            raise ValueError(f"Unsupported latent 'samples' type: {type(samples).__name__}")
        meta["latent_structure"] = structure

    for key, value in extras:
        if isinstance(value, torch.Tensor):
            add_tensor(f"latent_extra_{key}", value)
            meta["latent_tensors"].append(key)
        else:
            encoded = encode_structure(value, f"latent_extra_{key}")
            if encoded is None:
                print(f"[WARN] Skipping non-serializable latent key '{key}' ({type(value).__name__})")
            else:
                meta["latent_values"][key] = encoded

    for index, entry in enumerate(conditioning):
        cond_tensor, cond_dict = entry[0], entry[1]
        add_tensor(f"cond_{index}", cond_tensor)
        entry_meta = {"tensors": [], "values": {}}
        for key, value in cond_dict.items():
            if isinstance(value, torch.Tensor):
                add_tensor(f"cond_{index}_{key}", value)
                entry_meta["tensors"].append(key)
            else:
                encoded = encode_structure(value, f"cond_{index}_{key}")
                if encoded is None:
                    print(f"[WARN] Skipping non-serializable conditioning key '{key}' ({type(value).__name__})")
                else:
                    entry_meta["values"][key] = encoded
        meta["cond"].append(entry_meta)

    return tensors, meta


def _decode_structure(entry, get_tensor):
    kind = entry["t"]
    if kind == "tensor":
        return get_tensor(entry["k"])
    if kind == "nested_tensor":
        return NestedTensor([_decode_structure(item, get_tensor) for item in entry["items"]])
    if kind == "object":
        module_name, class_name = entry["class"].rsplit(".", 1)
        obj = getattr(importlib.import_module(module_name), class_name)()
        for k, v in entry["attrs"].items():
            setattr(obj, k, _decode_structure(v, get_tensor))
        return obj
    if kind == "list":
        if "items" in entry:
            return [_decode_structure(item, get_tensor) for item in entry["items"]]
        return entry["v"]
    if kind == "tuple":
        if "items" in entry:
            return tuple(_decode_structure(item, get_tensor) for item in entry["items"])
        return tuple(entry["v"])
    if kind == "dict":
        return {k: _decode_structure(v, get_tensor) for k, v in entry["items"].items()}
    if kind == "none":
        return None
    return entry["v"]


def decode_latent_conditioning(file_handle):
    """Rebuild (latent, conditioning) from an open safetensors handle."""
    meta = json.loads(file_handle.metadata()[METADATA_KEY])

    if "latent_structure" in meta:
        samples = _decode_structure(meta["latent_structure"], file_handle.get_tensor)
    else:
        samples = file_handle.get_tensor("latent")
    latent = {"samples": samples}
    for key in meta.get("latent_tensors", []):
        latent[key] = file_handle.get_tensor(f"latent_extra_{key}")
    for key, encoded in meta.get("latent_values", {}).items():
        latent[key] = _decode_structure(encoded, file_handle.get_tensor)

    conditioning = []
    for index, entry_meta in enumerate(meta["cond"]):
        cond_tensor = file_handle.get_tensor(f"cond_{index}")
        cond_dict = {}
        for key in entry_meta.get("tensors", []):
            cond_dict[key] = file_handle.get_tensor(f"cond_{index}_{key}")
        for key, encoded in entry_meta.get("values", {}).items():
            cond_dict[key] = _decode_structure(encoded, file_handle.get_tensor)
        conditioning.append([cond_tensor, cond_dict])

    return latent, conditioning


def file_has_latent_conditioning(path):
    """Cheap header check: True if the file holds the combined latent+conditioning format."""
    try:
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
            return metadata is not None and METADATA_KEY in metadata
    except Exception:
        return False

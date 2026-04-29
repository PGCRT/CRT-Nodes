import hashlib

import numpy as np
import torch


def _update_hasher(hasher, value):
    if value is None:
        hasher.update(b"none")
        return

    if isinstance(value, bool):
        hasher.update(f"bool:{int(value)}".encode("utf-8"))
        return

    if isinstance(value, int):
        hasher.update(f"int:{value}".encode("utf-8"))
        return

    if isinstance(value, float):
        hasher.update(f"float:{repr(value)}".encode("utf-8"))
        return

    if isinstance(value, str):
        hasher.update(b"str:")
        hasher.update(value.encode("utf-8"))
        return

    if isinstance(value, bytes):
        hasher.update(b"bytes:")
        hasher.update(value)
        return

    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        hasher.update(f"torch:{tensor.dtype}:{tuple(tensor.shape)}".encode("utf-8"))
        hasher.update(memoryview(tensor.numpy()).cast("B"))
        return

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(f"numpy:{array.dtype}:{array.shape}".encode("utf-8"))
        hasher.update(memoryview(array).cast("B"))
        return

    if isinstance(value, dict):
        hasher.update(b"dict{")
        for key in sorted(value.keys(), key=lambda item: repr(item)):
            _update_hasher(hasher, key)
            _update_hasher(hasher, value[key])
        hasher.update(b"}")
        return

    if isinstance(value, (list, tuple)):
        hasher.update(f"seq:{type(value).__name__}[".encode("utf-8"))
        for item in value:
            _update_hasher(hasher, item)
        hasher.update(b"]")
        return

    if isinstance(value, set):
        hasher.update(b"set{")
        for item in sorted(value, key=lambda item: repr(item)):
            _update_hasher(hasher, item)
        hasher.update(b"}")
        return

    hasher.update(
        f"object:{type(value).__module__}.{type(value).__qualname__}:{id(value)}".encode(
            "utf-8"
        )
    )


def stable_fingerprint(*values):
    hasher = hashlib.sha256()
    for value in values:
        _update_hasher(hasher, value)
    return hasher.hexdigest()

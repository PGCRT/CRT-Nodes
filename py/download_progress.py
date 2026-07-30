from __future__ import annotations

import inspect
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

try:
    from tqdm.auto import tqdm as _Tqdm
except ImportError:  # pragma: no cover - huggingface_hub normally provides tqdm.
    _Tqdm = None


ProgressCallback = Callable[[int, int], None]


if _Tqdm is not None:

    class CRTDownloadTqdm(_Tqdm):
        """A tqdm variant that always renders in the ComfyUI console."""

        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            kwargs.setdefault("file", sys.stdout)
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("mininterval", 0.25)
            kwargs.setdefault("maxinterval", 1.0)
            kwargs.setdefault("leave", True)
            super().__init__(*args, **kwargs)

else:
    CRTDownloadTqdm = None


def _content_length(response) -> int:
    raw_value = response.headers.get("Content-Length")
    if raw_value is None:
        return 0
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def download_url_with_progress(
    url: str,
    destination: str | os.PathLike,
    *,
    label: str | None = None,
    user_agent: str = "CRT-Nodes/1.0",
    timeout: float | None = None,
    temp_path: str | os.PathLike | None = None,
    progress_callback: ProgressCallback | None = None,
    chunk_size: int = 1024 * 1024,
    console_prefix: str = "CRT Download",
) -> str:
    """Download one file atomically while reporting bytes, speed, and ETA."""

    destination_path = Path(destination)
    if destination_path.is_file():
        return str(destination_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = (
        Path(temp_path)
        if temp_path is not None
        else destination_path.with_name(f"{destination_path.name}.part")
    )
    display_name = label or destination_path.name
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
    )
    open_kwargs = {} if timeout is None else {"timeout": timeout}

    print(f"[{console_prefix}] Starting: {display_name}", flush=True)
    downloaded = 0
    fallback_log_interval = 64 * 1024 * 1024
    last_fallback_log = 0
    try:
        with urllib.request.urlopen(request, **open_kwargs) as response:
            total_size = _content_length(response)
            if CRTDownloadTqdm is None:
                print(
                    f"[{console_prefix}] tqdm is unavailable; "
                    "downloaded byte counts will be logged without a progress bar.",
                    flush=True,
                )
                progress = None
            else:
                progress = CRTDownloadTqdm(
                    total=total_size or None,
                    desc=display_name,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )

            try:
                with partial_path.open("wb") as output:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
                        elif downloaded - last_fallback_log >= fallback_log_interval:
                            if total_size:
                                percent = downloaded * 100.0 / total_size
                                detail = f"{downloaded / (1024 * 1024):.1f} MiB ({percent:.1f}%)"
                            else:
                                detail = f"{downloaded / (1024 * 1024):.1f} MiB"
                            print(f"[{console_prefix}] Progress: {display_name}: {detail}", flush=True)
                            last_fallback_log = downloaded
                        if progress_callback is not None:
                            progress_callback(downloaded, total_size)
            finally:
                if progress is not None:
                    progress.close()

        if total_size and downloaded != total_size:
            raise IOError(
                f"Incomplete download for {display_name}: "
                f"received {downloaded} of {total_size} bytes."
            )

        os.replace(partial_path, destination_path)
    except Exception:
        try:
            partial_path.unlink()
        except OSError:
            pass
        print(f"[{console_prefix}] Failed: {display_name}", flush=True)
        raise

    print(
        f"[{console_prefix}] Complete: {display_name} "
        f"({downloaded / (1024 * 1024):.1f} MiB)",
        flush=True,
    )
    return str(destination_path)


def snapshot_download_with_progress(
    *,
    repo_id: str,
    local_dir: str | os.PathLike,
    label: str | None = None,
    console_prefix: str = "CRT Hugging Face",
    **kwargs,
) -> str:
    """Run snapshot_download with progress bars forced on in the console."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for this model download."
        ) from exc

    signature = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" not in signature.parameters:
        kwargs.pop("local_dir_use_symlinks", None)
    if CRTDownloadTqdm is not None and "tqdm_class" in signature.parameters:
        kwargs["tqdm_class"] = CRTDownloadTqdm

    display_name = label or repo_id
    print(
        f"[{console_prefix}] Starting/resuming: {display_name} "
        f"({repo_id}) -> {local_dir}",
        flush=True,
    )
    result = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        **kwargs,
    )
    print(f"[{console_prefix}] Complete: {display_name}", flush=True)
    return str(result)

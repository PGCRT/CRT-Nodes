import importlib
import importlib.util
import inspect
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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


_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)$"
)


def _hf_token() -> str | None:
    """Return an explicit HF_TOKEN, or the cached huggingface-cli token."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub.utils import get_token
        return get_token()
    except Exception:
        return None


def _content_length(response) -> int:
    raw_value = response.headers.get("Content-Length")
    if raw_value is None:
        return 0
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _hf_download_url(url: str, destination: str | os.PathLike, label: str | None, console_prefix: str) -> str:
    """Use huggingface_hub for Hugging Face files (resume + fast transfer paths)."""
    # Max throughput: Xet chunked downloads when hf_xet is present, plus the
    # multi-connection Rust accelerator for classic CDN files. Some custom
    # nodes set HF_HUB_DISABLE_XET process-wide; undo that so downloads always
    # get the fast path.
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    url = url.split("?")[0]

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    match = _HF_URL_RE.match(url)
    repo_id = match.group(1)
    revision = match.group(2)
    repo_path = match.group(3)

    display_name = label or destination_path.name

    # Best-effort: announce total size up front (hf_transfer gives no live bar).
    try:
        head_req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=15) as head:
            total = int(head.headers.get("Content-Length") or 0)
        if total:
            print(
                f"[{console_prefix}] Starting (HF Hub): {display_name} "
                f"({total / (1024 * 1024):.1f} MiB)",
                flush=True,
            )
    except Exception:
        pass

    print(
        f"[{console_prefix}] Starting (HF Hub): {display_name} from {repo_id}",
        flush=True,
    )

    kwargs = {
        "repo_id": repo_id,
        "filename": repo_path,
        "revision": revision,
        "repo_type": "model",
        "token": os.environ.get("HF_TOKEN") or True,
    }

    try:
        cached_path = hf_hub_download(**kwargs)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"[{console_prefix}] {repo_id} is gated/private on Hugging Face.\n"
            f"Access it here: https://huggingface.co/{repo_id}\n"
            "Steps:\n"
            "1. Log in and click Accept / Request access on the model page.\n"
            "2. Create a token at https://huggingface.co/settings/tokens\n"
            "3. Run: huggingface-cli login --token hf_xxxxxxxx\n"
            "   Or set env var: set HF_TOKEN=hf_xxxxxxxx\n"
            "4. Restart ComfyUI."
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            raise RuntimeError(
                f"[{console_prefix}] {repo_id} requires Hugging Face authentication.\n"
                f"Model page: https://huggingface.co/{repo_id}\n"
                "Steps:\n"
                "1. Create a token at https://huggingface.co/settings/tokens\n"
                "2. Run: huggingface-cli login --token hf_xxxxxxxx\n"
                "   Or set env var: set HF_TOKEN=hf_xxxxxxxx\n"
                "3. Restart ComfyUI."
            ) from exc
        raise

    if str(Path(cached_path).resolve()) != str(destination_path.resolve()):
        shutil.copyfile(cached_path, destination_path)

    size_mb = destination_path.stat().st_size / (1024 * 1024)
    print(
        f"[{console_prefix}] Complete: {display_name} ({size_mb:.1f} MiB)",
        flush=True,
    )
    return str(destination_path)


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

    is_hf = bool(_HF_URL_RE.match(url))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = (
        Path(temp_path)
        if temp_path is not None
        else destination_path.with_name(f"{destination_path.name}.part")
    )
    display_name = label or destination_path.name

    headers = {"User-Agent": user_agent}
    if is_hf:
        token = _hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    open_kwargs = {} if timeout is None else {"timeout": timeout}

    # Prefer huggingface_hub for HF files: resumable, CDN-correct, and uses the
    # hf_transfer/Xet fast paths when installed. urllib single-stream gets
    # throttled to dial-up speeds by the CDN on large LFS files.
    if is_hf:
        try:
            return _hf_download_url(url, destination_path, display_name, console_prefix)
        except ImportError:
            print(
                f"[{console_prefix}] huggingface_hub unavailable; using direct download for {display_name}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[{console_prefix}] HF hub download failed ({exc.__class__.__name__}); "
                f"falling back to direct download for {display_name}",
                flush=True,
            )

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
    except urllib.error.HTTPError as exc:
        try:
            partial_path.unlink()
        except OSError:
            pass
        if is_hf and exc.code in (401, 403):
            print(
                f"[{console_prefix}] urllib got {exc.code}; falling back to huggingface_hub for {display_name}",
                flush=True,
            )
            try:
                return _hf_download_url(url, destination_path, label, console_prefix)
            except ImportError:
                pass
        print(f"[{console_prefix}] Failed: {display_name}", flush=True)
        raise
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

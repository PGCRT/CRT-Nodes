import logging
import os
import urllib.request


LOGGER = logging.getLogger(__name__)


def download_file(url, dest_path, label, user_agent="CRT-Nodes AudioTranscript/1.0"):
    if os.path.isfile(dest_path):
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.part"
    LOGGER.info("Downloading %s from %s", label, url)
    print(f"[CRT Audio Transcript] Downloading {label}...")

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print(f"[CRT Audio Transcript] Model downloaded: {dest_path}")
    return dest_path

import logging
import os
from ..download_progress import download_url_with_progress


LOGGER = logging.getLogger(__name__)


def download_file(url, dest_path, label, user_agent="CRT-Nodes AudioTranscript/1.0"):
    if os.path.isfile(dest_path):
        return dest_path

    LOGGER.info("Downloading %s from %s", label, url)
    download_url_with_progress(
        url,
        dest_path,
        label=label,
        user_agent=user_agent,
        temp_path=f"{dest_path}.part",
        console_prefix="CRT Audio Transcript",
    )

    print(f"[CRT Audio Transcript] Model downloaded: {dest_path}")
    return dest_path

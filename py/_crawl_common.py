import os
from pathlib import Path


# ANSI colors: terminals and the ComfyUI web console render these; they degrade
# to plain text elsewhere. Crawl fallbacks warn (yellow) and never stop the run.
YELLOW = "\033[93m"
ENDC = "\033[0m"


def warn(message):
    """Print a non-fatal yellow warning without interrupting execution."""
    print(f"{YELLOW}[CRT Crawl] {message}{ENDC}")


def to_int(value):
    try:
        value = int(value)
    except Exception:
        return 0
    if value != value:  # NaN check
        return 0
    return value


def _depth_of(path, root):
    return len(path.relative_to(root).parts)


def scan_tree(root, max_depth):
    """Walk the tree under root once.

    Returns (folders, files_by_folder): folders lists every visited directory
    (root first); files_by_folder maps each folder's normalized path to the
    files it directly contains. max_depth: 0 = root only, N = N levels deep,
    -1 = infinite. Inaccessible folders print a warning instead of aborting.
    """
    folders = [root]
    files_by_folder = {os.path.normpath(str(root)): []}

    def onerror(error):
        print(f"[CRT Crawl] cannot access '{getattr(error, 'filename', '?')}': {getattr(error, 'strerror', error)}")

    for current, dirs, names in os.walk(root, onerror=onerror):
        cur = Path(current)
        if max_depth == 0:
            dirs[:] = []
            if cur != root:
                continue
        elif max_depth != -1 and _depth_of(cur, root) >= max_depth:
            dirs[:] = []
        cur_key = os.path.normpath(str(cur))
        for name in dirs:
            child = cur / name
            folders.append(child)
            files_by_folder.setdefault(os.path.normpath(str(child)), [])
        f_list = files_by_folder.setdefault(cur_key, [])
        for name in names:
            path = cur / name
            if path.is_file():
                f_list.append(path)
    return folders, files_by_folder


def pick_folder(folders, subfolder_seed, file_seed):
    """Choose a folder and the seed used to index its files.

    subfolder_seed None -> column-major: file_seed cycles folders first
    (folder = seed % count), then advances one file per full folder sweep
    (file seed = seed // count).
    subfolder_seed set -> folder = subfolder_seed % count, file seed unchanged.
    """
    num_folders = len(folders)
    if num_folders == 0:
        return None, file_seed
    if subfolder_seed is None:
        return folders[file_seed % num_folders], file_seed // num_folders
    return folders[subfolder_seed % num_folders], file_seed


def folders_with_files(folders, files_by_folder, predicate):
    """Keep only the folders that directly contain at least one file matching
    ``predicate(Path) -> bool``.

    Seed selection then ranges over folders that actually hold the requested
    type, so an empty folder never consumes a seed slot nor produces a
    placeholder while other folders still have matches.
    """

    def has_match(folder):
        for path in files_by_folder.get(os.path.normpath(str(folder)), []):
            if predicate(path):
                return True
        return False

    return [folder for folder in folders if has_match(folder)]

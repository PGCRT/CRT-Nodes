import folder_paths
import json
import os
import math
import re

from typing import Union

from nodes import LoraLoader


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    def __init__(self, type, data: Union[dict, None] = None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


any_type = AnyType("*")


def log_node_warn(node_name, message):
    print(f"[magic-lora-loader][{node_name}] {message}")


def _file_exists(path):
    return path is not None and os.path.isfile(path)


def _path_exists(path):
    return path is not None and os.path.exists(path)


def _abspath(file_path: str):
    if file_path and not _path_exists(file_path):
        maybe_path = os.path.abspath(os.path.realpath(os.path.expanduser(file_path)))
        file_path = maybe_path if _path_exists(maybe_path) else file_path
    return file_path


def _load_json_file(file: str, default=None):
    if _path_exists(file):
        with open(file, "r", encoding="UTF-8") as f:
            config = f.read()
            try:
                return json.loads(config)
            except json.decoder.JSONDecodeError:
                try:
                    config = re.sub(r"^\s*//\s.*", "", config, flags=re.MULTILINE)
                    return json.loads(config)
                except json.decoder.JSONDecodeError:
                    try:
                        config = re.sub(r"(?:^|\s)//.*", "", config, flags=re.MULTILINE)
                        return json.loads(config)
                    except json.decoder.JSONDecodeError:
                        pass
    return default


def get_lora_by_filename(file_path, lora_paths=None, log_node=None):
    lora_paths = (
        lora_paths if lora_paths is not None else folder_paths.get_filename_list("loras")
    )

    if file_path in lora_paths:
        return file_path

    no_ext = [os.path.splitext(x)[0] for x in lora_paths]
    if file_path in no_ext:
        return lora_paths[no_ext.index(file_path)]

    file_path_no_ext = os.path.splitext(file_path)[0]
    if file_path_no_ext in no_ext:
        return lora_paths[no_ext.index(file_path_no_ext)]

    names = [os.path.basename(x) for x in lora_paths]
    if file_path in names:
        return lora_paths[names.index(file_path)]

    file_name = os.path.basename(file_path)
    if file_name in names:
        return lora_paths[names.index(file_name)]

    names_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    if file_path in names_no_ext:
        return lora_paths[names_no_ext.index(file_path)]

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_name_no_ext in names_no_ext:
        return lora_paths[names_no_ext.index(file_name_no_ext)]

    for i, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            return lora_paths[i]

    if log_node is not None:
        log_node_warn(log_node, f'LoRA "{file_path}" not found, skipping.')
    return None


def _get_info_file(file_path: str, force=False):
    info_path = f"{file_path}.rgthree-info.json"
    return info_path if _file_exists(info_path) or force else None


def _get_folder_path(file: str, model_type) -> str | None:
    file_path = folder_paths.get_full_path(model_type, file)
    if not _file_exists(file_path):
        file_path = _abspath(file_path)
    if not _file_exists(file_path):
        file_path = None
    return file_path


def get_model_info_file_data(file: str, model_type, default=None):
    file_path = _get_folder_path(file, model_type)
    if file_path is None:
        return default
    return _load_json_file(_get_info_file(file_path), default=default)

NODE_NAME = "Magic LoRA Loader"
_PRESETS_FILE = None
_MODEL_TYPES = {"Flux2Klein", "LTX2.3"}
_DEFAULT_MT = "Flux2Klein"


def _is_all_ones(block_weights: dict) -> bool:
    if not block_weights:
        return True
    for vals in block_weights.values():
        for w in vals:
            if float(w) != 1.0:
                return False
    return True


def _key_block_weight(key: str, block_weights: dict) -> float:
    m = re.search(r"double_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("double", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"single_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("single", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"(?:diffusion_model\.)?transformer_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("transformer", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    return 1.0


def _scale_new_patches(patcher, block_weights: dict, pre_counts: dict):
    if patcher is None:
        return
    patches = getattr(patcher, "patches", None)
    if not patches:
        return
    for key, entries in patches.items():
        weight = _key_block_weight(key, block_weights)
        if weight == 1.0:
            continue
        pre = pre_counts.get(key, 0)
        new_entries = entries[pre:]
        if not new_entries:
            continue
        patches[key] = entries[:pre] + [(e[0] * weight,) + e[1:] for e in new_entries]


def load_lora_with_blocks(
    model,
    clip,
    lora_name: str,
    strength_model: float,
    strength_clip: float,
    block_weights: dict | None,
):
    if block_weights is None or _is_all_ones(block_weights):
        return LoraLoader().load_lora(model, clip, lora_name, strength_model, strength_clip)

    model_pre = {k: len(v) for k, v in (getattr(model, "patches", None) or {}).items()}
    clip_pre = {k: len(v) for k, v in (getattr(clip, "patches", None) or {}).items()}

    new_model, new_clip = LoraLoader().load_lora(
        model, clip, lora_name, strength_model, strength_clip
    )

    _scale_new_patches(new_model, block_weights, model_pre)
    _scale_new_patches(new_clip, block_weights, clip_pre)

    return new_model, new_clip


class MagicLoraLoader:
    """Magic LoRA Loader with optional wet/cap/block controls."""

    NAME = NODE_NAME
    CATEGORY = "CRT/LoRA"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(
                type=any_type,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"

    def load_loras(self, model=None, clip=None, **kwargs):
        wet = 1.0
        wet_raw = kwargs.pop("wet", None)
        if isinstance(wet_raw, dict) and wet_raw.get("__pgc_wet"):
            wet = max(0.0, min(2.0, float(wet_raw.get("value", 1.0))))
        elif isinstance(wet_raw, (int, float)):
            wet = max(0.0, min(2.0, float(wet_raw)))

        cap = 0.0
        cap_raw = kwargs.pop("pgc_cap", None)
        if isinstance(cap_raw, dict) and cap_raw.get("__pgc_cap") is True:
            cap = max(0.0, float(cap_raw.get("value", 0.0)))
        elif isinstance(cap_raw, (int, float)):
            cap = max(0.0, float(cap_raw))

        kwargs.pop("pgc_model_type", None)

        global_blocks = None
        gb_raw = kwargs.pop("pgc_global_blocks", None)
        kwargs.pop("global_blocks", None)
        if isinstance(gb_raw, dict) and gb_raw.get("__pgc_global_blocks") is True:
            candidate = {}
            for bkey in ("double", "single", "transformer"):
                raw_vals = gb_raw.get(bkey)
                if raw_vals:
                    candidate[bkey] = [float(v) for v in raw_vals]
            if candidate and any(v != 1.0 for vals in candidate.values() for v in vals):
                global_blocks = candidate

        for key, value in kwargs.items():
            key = key.upper()
            if (
                key.startswith("LORA_")
                and "on" in value
                and "lora" in value
                and "strength" in value
            ):
                strength_model = value["strength"] * wet
                strength_clip_raw = value.get("strengthTwo")

                if clip is None:
                    if strength_clip_raw is not None and strength_clip_raw != 0:
                        log_node_warn(
                            NODE_NAME,
                            "Received clip strength even though no clip supplied!",
                        )
                    strength_clip = 0
                else:
                    strength_clip = (
                        strength_clip_raw * wet
                        if strength_clip_raw is not None
                        else strength_model
                    )

                if cap > 0.0:
                    strength_model = max(-cap, min(cap, strength_model))
                    if clip is not None:
                        strength_clip = max(-cap, min(cap, strength_clip))

                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = get_lora_by_filename(value["lora"], log_node=self.NAME)
                    if model is not None and lora is not None:
                        per_lora_blocks = value.get("blocks")
                        if isinstance(per_lora_blocks, dict):
                            merged = {}
                            all_keys = set(per_lora_blocks.keys()) | (
                                set(global_blocks.keys()) if global_blocks else set()
                            )
                            for bkey in all_keys:
                                pl = [float(v) for v in per_lora_blocks.get(bkey, [])]
                                if global_blocks:
                                    gb = [float(v) for v in global_blocks.get(bkey, [])]
                                    if pl and gb:
                                        merged[bkey] = [
                                            pl[i] * (gb[i] if i < len(gb) else 1.0)
                                            for i in range(len(pl))
                                        ]
                                    elif pl:
                                        merged[bkey] = pl
                                    elif gb:
                                        merged[bkey] = gb
                                else:
                                    merged[bkey] = pl
                            blocks = merged if merged else None
                        else:
                            blocks = global_blocks

                        if blocks is not None:
                            model, clip = load_lora_with_blocks(
                                model, clip, lora, strength_model, strength_clip, blocks
                            )
                        else:
                            model, clip = LoraLoader().load_lora(
                                model, clip, lora, strength_model, strength_clip
                            )

        return (model, clip)

    @classmethod
    def get_enabled_loras_from_prompt_node(
        cls, prompt_node: dict
    ) -> list[dict[str, Union[str, float]]]:
        result = []
        for name, lora in prompt_node["inputs"].items():
            if name.startswith("lora_") and lora["on"]:
                lora_file = get_lora_by_filename(lora["lora"], log_node=cls.NAME)
                if lora_file is not None:
                    lora_dict = {
                        "name": lora["lora"],
                        "strength": lora["strength"],
                        "path": folder_paths.get_full_path("loras", lora_file),
                    }
                    if "strengthTwo" in lora:
                        lora_dict["strength_clip"] = lora["strengthTwo"]
                    result.append(lora_dict)
        return result

    @classmethod
    def get_enabled_triggers_from_prompt_node(
        cls, prompt_node: dict, max_each: int = 1
    ):
        loras = [l["name"] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
        trained_words = []
        for lora in loras:
            info = get_model_info_file_data(lora, "loras", default={})
            if not info or not info.keys():
                log_node_warn(
                    NODE_NAME,
                    f"No info found for lora {lora} when grabbing triggers. Have you generated an info file"
                    ' from the Magic LoRA Loader "Show Info" dialog?',
                )
                continue
            if "trainedWords" not in info or not info["trainedWords"]:
                log_node_warn(
                    NODE_NAME,
                    f"No trained words for lora {lora} when grabbing triggers. Have you fetched data from"
                    " civitai or manually added words?",
                )
                continue
            trained_words += [
                w
                for wi in info["trainedWords"][:max_each]
                if (wi and (w := wi["word"]))
            ]
        return trained_words


def _load_presets_raw():
    if _PRESETS_FILE and os.path.exists(_PRESETS_FILE):
        try:
            with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_presets_raw(data: dict):
    os.makedirs(os.path.dirname(_PRESETS_FILE), exist_ok=True)
    with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _migrate_presets(data: dict) -> dict:
    if not data:
        return {mt: {} for mt in _MODEL_TYPES}
    if set(data.keys()) <= _MODEL_TYPES:
        result = {mt: {} for mt in _MODEL_TYPES}
        for mt in _MODEL_TYPES:
            result[mt] = data.get(mt, {})
        return result
    migrated = {mt: {} for mt in _MODEL_TYPES}
    migrated[_DEFAULT_MT] = data
    return migrated


def _load_presets(model_type=_DEFAULT_MT):
    data = _migrate_presets(_load_presets_raw())
    if model_type not in _MODEL_TYPES:
        model_type = _DEFAULT_MT
    return data.get(model_type, {})


def _save_presets(model_type: str, presets: dict):
    data = _migrate_presets(_load_presets_raw())
    if model_type not in _MODEL_TYPES:
        model_type = _DEFAULT_MT
    data[model_type] = presets
    _save_presets_raw(data)


_ARCH = {
    "Flux2Klein": {
        "block_types": [
            ("double", r"double_blocks[._](\d+)[._]", 8),
            ("single", r"single_blocks[._](\d+)[._]", 24),
        ],
    },
    "LTX2.3": {
        "block_types": [
            (
                "transformer",
                r"(?:diffusion_model[._])?transformer_blocks[._](\d+)[._]",
                48,
            ),
        ],
    },
}


def _flat_weights(model_type):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    return {btype: [1.0] * count for btype, _, count in cfg["block_types"]}


def _compute_profile(lora_sd, model_type=_DEFAULT_MT):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    accum = {btype: [0.0] * count for btype, _, count in cfg["block_types"]}
    for key, tensor in lora_sd.items():
        try:
            norm_sq = float(tensor.float().norm().item() ** 2)
        except Exception:
            continue
        for btype, pattern, count in cfg["block_types"]:
            m = re.search(pattern, key)
            if m:
                idx = int(m.group(1))
                if idx < count:
                    accum[btype][idx] += norm_sq
                break
    return {btype: [math.sqrt(v) for v in vals] for btype, vals in accum.items()}


def _normalise_profile(profile):
    all_vals = [v for vals in profile.values() for v in vals]
    peak = max(all_vals) if all_vals else 0.0
    if peak < 1e-9:
        peak = 1.0
    return {btype: [round(v / peak, 4) for v in vals] for btype, vals in profile.items()}


def _competitive_weights(profiles, model_type=_DEFAULT_MT):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    block_types = [(btype, count) for btype, _, count in cfg["block_types"]]
    total_blocks = sum(count for _, count in block_types)
    n = len(profiles)
    weights = [_flat_weights(model_type) for _ in range(n)]

    frac = []
    for p in profiles:
        total = sum(v for vals in p.values() for v in vals)
        if total < 1e-9:
            uniform = 1.0 / total_blocks
            frac.append({btype: [uniform] * count for btype, count in block_types})
        else:
            frac.append({btype: [v / total for v in p[btype]] for btype, _ in block_types})

    avg = {
        btype: [sum(frac[i][btype][b] for i in range(n)) / n for b in range(count)]
        for btype, count in block_types
    }

    rel = []
    for i in range(n):
        ri = {}
        for btype, count in block_types:
            ri[btype] = [
                frac[i][btype][b] / avg[btype][b] if avg[btype][b] > 1e-12 else 1.0
                for b in range(count)
            ]
        rel.append(ri)

    for btype, count in block_types:
        for b in range(count):
            imps = [rel[i][btype][b] for i in range(n)]
            peak = max(imps)
            if peak < 1e-9:
                continue
            for i in range(n):
                weights[i][btype][b] = round(max(imps[i] / peak, 0.1), 3)
    return weights


def setup_routes(base_dir: str):
    global _PRESETS_FILE
    _PRESETS_FILE = os.path.join(base_dir, "magic_lora_loader_presets.json")

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/crt-pll/api/loras")
    async def _get_loras(request):
        names = folder_paths.get_filename_list("loras")
        return web.json_response([{"file": n} for n in names])

    @PromptServer.instance.routes.get("/crt-pll/api/loras/info")
    async def _get_lora_info(request):
        file = request.rel_url.query.get("file", "")
        if not file:
            return web.json_response({}, status=200)
        info = get_model_info_file_data(file, "loras", default={})
        return web.json_response(info or {})

    @PromptServer.instance.routes.get("/crt-pll/api/presets")
    async def _get_presets(request):
        mt = request.rel_url.query.get("model_type", _DEFAULT_MT)
        return web.json_response({"status": 200, "data": _load_presets(mt)})

    @PromptServer.instance.routes.post("/crt-pll/api/presets")
    async def _save_preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"status": 400, "error": "Invalid JSON"}, status=400)
        name = body.get("name", "").strip()
        data = body.get("data")
        mt = body.get("model_type", _DEFAULT_MT)
        if not name or data is None:
            return web.json_response(
                {"status": 400, "error": "name and data required"}, status=400
            )
        presets = _load_presets(mt)
        presets[name] = data
        _save_presets(mt, presets)
        return web.json_response({"status": 200, "data": presets})

    @PromptServer.instance.routes.delete("/crt-pll/api/presets/{name}")
    async def _delete_preset(request):
        name = request.match_info.get("name", "")
        mt = request.rel_url.query.get("model_type", _DEFAULT_MT)
        presets = _load_presets(mt)
        if name in presets:
            del presets[name]
            _save_presets(mt, presets)
        return web.json_response({"status": 200, "data": presets})

    async def _automix_handler(request):
        try:
            import comfy.utils

            body = await request.json()
            loras = body.get("loras", [])
            model_type = body.get("model_type", _DEFAULT_MT)
            if model_type not in _ARCH:
                model_type = _DEFAULT_MT

            flat = _flat_weights(model_type)
            profiles = []
            errors = []
            for item in loras:
                name = item.get("name")
                if not name:
                    profiles.append(None)
                    continue
                path = folder_paths.get_full_path("loras", name)
                if not path:
                    errors.append(f"LoRA not found: {name}")
                    profiles.append(None)
                    continue
                try:
                    lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
                    profiles.append(_compute_profile(lora_sd, model_type=model_type))
                    del lora_sd
                except Exception as e:
                    errors.append(f"Error reading {name}: {e}")
                    profiles.append(None)

            valid = [p if p is not None else dict(flat) for p in profiles]
            weights = _competitive_weights(valid, model_type=model_type)
            return web.json_response(
                {
                    "profiles": [_normalise_profile(p) for p in valid],
                    "weights": weights,
                    "errors": errors,
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _block_profile_handler(request):
        try:
            import comfy.utils

            body = await request.json()
            name = body.get("name")
            model_type = body.get("model_type", _DEFAULT_MT)
            if model_type not in _ARCH:
                model_type = _DEFAULT_MT

            if not name:
                return web.json_response({"error": "no name"}, status=400)

            path = folder_paths.get_full_path("loras", name)
            if not path:
                return web.json_response({"error": f"LoRA not found: {name}"}, status=404)

            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            profile = _compute_profile(lora_sd, model_type=model_type)
            del lora_sd
            return web.json_response({"profile": _normalise_profile(profile)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    try:
        PromptServer.instance.app.router.add_post("/crt-pll/api/automix", _automix_handler)
        PromptServer.instance.app.router.add_post("/crt-pll/api/block_profile", _block_profile_handler)
        print("[crt-pll] presets + automix routes registered")
    except RuntimeError:
        pass

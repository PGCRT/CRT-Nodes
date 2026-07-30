from __future__ import annotations

from pathlib import Path


def patch_checkpoint_compatibility(model_dir: Path) -> None:
    """Patch two upstream custom-code assumptions that break Transformers 5.5."""

    vision_path = model_dir / "modeling_intern_vit.py"
    if vision_path.is_file():
        source = vision_path.read_text(encoding="utf-8")
        old = (
            "        dpr = [x.item() for x in torch.linspace("
            "0, config.drop_path_rate, config.num_hidden_layers)]"
        )
        new = (
            "        dpr = [\n"
            "            config.drop_path_rate * index / "
            "max(config.num_hidden_layers - 1, 1)\n"
            "            for index in range(config.num_hidden_layers)\n"
            "        ]"
        )
        if old in source:
            vision_path.write_text(source.replace(old, new, 1), encoding="utf-8")
            print("[ERNIE Image Aes] Applied meta-device vision initialization fix.")

    chat_path = model_dir / "modeling_internvl_chat.py"
    if chat_path.is_file():
        source = chat_path.read_text(encoding="utf-8")
        old = (
            "    config_class = InternVLChatConfig\n"
            "    main_input_name = 'pixel_values'"
        )
        new = (
            "    config_class = InternVLChatConfig\n"
            "    all_tied_weights_keys = {}\n"
            "    main_input_name = 'pixel_values'"
        )
        if old in source:
            chat_path.write_text(source.replace(old, new, 1), encoding="utf-8")
            print("[ERNIE Image Aes] Applied Transformers 5.5 tied-weights fix.")

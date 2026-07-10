from __future__ import annotations
import os
from transformers import AutoConfig, AutoModel


class ModelLoadError(Exception):
    pass


def load_model_from_path(local_path: str):
    """Build a model from a local directory using AutoConfig + AutoModel.from_config.

    Deliberately does NOT load pretrained weights: from_config() gives a
    randomly-initialized model with the exact same architecture, shapes,
    and param counts as the checkpoint, without needing to read/allocate
    the actual weight tensors. This makes architecture inspection fast
    and independent of checkpoint size.

    local_path must contain a config.json (standard HF repo layout,
    e.g. a snapshot dir from huggingface_hub cache, or a manually
    downloaded/saved model directory).
    """
    if not os.path.isdir(local_path):
        raise ModelLoadError(f"Not a directory: {local_path}")

    config_path = os.path.join(local_path, "config.json")
    if not os.path.isfile(config_path):
        raise ModelLoadError(
            f"No config.json found in {local_path}. "
            "Expected a local HF model directory (e.g. containing "
            "config.json, tokenizer files, etc.)."
        )

    try:
        config = AutoConfig.from_pretrained(local_path, local_files_only=True)
    except Exception as e:
        raise ModelLoadError(f"Failed to parse config.json: {e}") from e

    try:
        model = AutoModel.from_config(config)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to construct model from config (architecture: "
            f"{getattr(config, 'architectures', 'unknown')}): {e}"
        ) from e

    model.eval()
    return model, config
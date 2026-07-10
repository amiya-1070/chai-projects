"""Manual sanity check: point this at a real local HF model directory
(one containing config.json) and inspect the resulting tree.

Usage:
    python scripts/inspect_local_model.py /path/to/local/model/dir
"""
import sys

from llm_explorer.core.model_loader import load_model_from_path, ModelLoadError
from llm_explorer.core.tree_builder import build_tree, collapse_repeats, model_total_params


def print_tree(node, indent=0):
    prefix = "  " * indent
    rep = f" [x{node.repeat_count}]" if node.repeat_count else ""
    shapes = f" {node.shape_info}" if node.shape_info else ""
    print(f"{prefix}{node.local_name} ({node.module_type}){rep}{shapes}")
    for child in node.children:
        print_tree(child, indent + 1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_local_model.py /path/to/model/dir")
        sys.exit(1)

    path = sys.argv[1]
    try:
        model, config = load_model_from_path(path)
    except ModelLoadError as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Architecture: {getattr(config, 'architectures', 'unknown')}")
    print(f"Config class: {type(config).__name__}")
    print()

    tree = build_tree(model, root_name="model")
    tree = collapse_repeats(tree, min_repeat=3)

    print_tree(tree)
    print()
    print(f"Total params (from tree): {model_total_params(tree):,}")
    print(f"Total params (from model directly): {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
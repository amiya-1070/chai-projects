from __future__ import annotations
import json

from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, NodeMode
from llm_explorer.core.resource_estimator import estimate as compute_estimate
from . import templates


def _make_cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook(model_path: str, tree: LayerNode, selection: SelectionState) -> dict:
    """Assemble a complete .ipynb (as a dict, ready for json.dump) from the
    current model tree + user selections. Causal LM fine-tuning only, for now.
    """
    cells = []

    def add(cell_type_source: tuple[str, str]):
        cell_type, source = cell_type_source
        cells.append(_make_cell(cell_type, source))

    add(templates.title_markdown(model_path))

    result = compute_estimate(tree, selection)
    add(templates.resource_summary_markdown(result))

    pruned_names = selection.pruned_names()
    if pruned_names:
        add(templates.pruning_notice_markdown(pruned_names))

    add(templates.install_deps_code())
    add(templates.imports_code())
    add(templates.load_model_code(model_path))

    peft_target_names = selection.peft_target_names()
    peft_targets = [(name, selection.peft_configs[name]) for name in peft_target_names]
    add(templates.peft_config_code(peft_targets))

    add(templates.dataset_placeholder_code())
    add(templates.training_args_code())
    add(templates.trainer_code())
    add(templates.save_and_export_code())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_notebook(notebook: dict, path: str):
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
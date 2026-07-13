import json
import pytest

from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, PeftConfig
from llm_explorer.codegen.notebook_builder import build_notebook, save_notebook


def make_linear_node(name, out_features, in_features):
    return LayerNode(
        name=name,
        local_name=name.split(".")[-1],
        module_type="Linear",
        param_count=out_features * in_features,
        shape_info={"weight": (out_features, in_features)},
        dtype="float16",
    )


def make_container(name, children):
    return LayerNode(name=name, local_name=name.split(".")[-1],
                      module_type="Container", children=children)


def build_test_tree():
    q_proj = make_linear_node("model.layers.0.self_attn.q_proj", 2048, 2048)
    k_proj = make_linear_node("model.layers.0.self_attn.k_proj", 512, 2048)
    attn = make_container("model.layers.0.self_attn", [q_proj, k_proj])

    gate_proj = make_linear_node("model.layers.0.mlp.gate_proj", 8192, 2048)
    mlp = make_container("model.layers.0.mlp", [gate_proj])

    layer0 = make_container("model.layers.0", [attn, mlp])
    layers = make_container("model.layers", [layer0])
    root = make_container("model", [layers])
    return root, q_proj, k_proj, gate_proj


# ---------- Structural validity ----------

def test_notebook_is_valid_nbformat_structure():
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)

    assert nb["nbformat"] == 4
    assert "cells" in nb
    assert len(nb["cells"]) > 0
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])


def test_notebook_is_json_serializable_and_roundtrips(tmp_path):
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)

    out_path = tmp_path / "test_notebook.ipynb"
    save_notebook(nb, str(out_path))

    assert out_path.exists()
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded["nbformat"] == 4
    assert len(reloaded["cells"]) == len(nb["cells"])


# ---------- Content correctness ----------

def _all_source(nb) -> str:
    return "\n".join("".join(cell["source"]) for cell in nb["cells"])


def test_model_path_appears_in_load_cell():
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("meta-llama/Llama-3.2-1B", tree, selection)
    assert "meta-llama/Llama-3.2-1B" in _all_source(nb)


def test_no_peft_targets_produces_full_finetune_note():
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)
    assert "peft_config = None" in source
    assert "LoraConfig(" not in source


def test_peft_targets_produce_lora_config_with_correct_modules():
    tree, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8, alpha=16, dropout=0.05))
    selection.set_peft_target(gate_proj.name, PeftConfig(method="LoRA", rank=8, alpha=16, dropout=0.05))

    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)

    assert "LoraConfig(" in source
    assert "r=8" in source
    assert "lora_alpha=16" in source
    assert "'q_proj'" in source
    assert "'gate_proj'" in source
    assert "get_peft_model(model, peft_config)" in source


def test_mixed_peft_methods_include_warning_comment():
    tree, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8))
    selection.set_peft_target(gate_proj.name, PeftConfig(method="DoRA", rank=8))

    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)
    assert "multiple PEFT methods" in source.lower() or "NOTE" in source


def test_unsupported_peft_method_flagged_not_silently_dropped():
    tree, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="AdaLoRA", rank=4))

    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)
    assert "not yet supported" in source.lower()
    # q_proj should still end up in target_modules as a fallback, not vanish
    assert "'q_proj'" in source


def test_pruned_nodes_produce_warning_not_silent_removal():
    tree, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_pruned("model.layers.0.self_attn", True)

    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)
    assert "model.layers.0.self_attn" in source
    assert "not yet implemented" in source.lower()


def test_no_pruning_notice_when_nothing_pruned():
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)
    assert "Note on pruning" not in source


def test_resource_summary_numbers_match_estimator():
    from llm_explorer.core.resource_estimator import estimate as compute_estimate

    tree, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8))

    expected = compute_estimate(tree, selection)
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)
    source = _all_source(nb)

    assert f"{expected.trainable_params:,}" in source
    assert f"{expected.total_params:,}" in source


def test_cell_ordering_title_first_deps_before_model_load():
    tree, *_ = build_test_tree()
    selection = SelectionState()
    nb = build_notebook("/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct", tree, selection)

    sources = ["".join(c["source"]) for c in nb["cells"]]
    title_idx = next(i for i, s in enumerate(sources) if "# Fine-tuning Notebook" in s)
    load_idx = next(i for i, s in enumerate(sources) if "AutoModelForCausalLM.from_pretrained" in s)
    trainer_idx = next(i for i, s in enumerate(sources) if "SFTTrainer(" in s)

    assert title_idx < load_idx < trainer_idx
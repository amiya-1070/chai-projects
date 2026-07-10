import pytest
from transformers import AutoConfig, AutoModel
from llm_explorer.core.tree_builder import (
    build_tree,
    collapse_repeats,
    model_total_params,
    _structural_signature,
)


def _tiny_gpt2_model(n_layer=4, n_embd=32, n_head=2):
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = n_layer
    config.n_embd = n_embd
    config.n_head = n_head
    return AutoModel.from_config(config)


def test_repeat_group_collapses_to_one_representative():
    model = _tiny_gpt2_model(n_layer=4)
    tree = build_tree(model, root_name="model")
    tree = collapse_repeats(tree, min_repeat=3)

    h_node = tree.find("model.h")
    assert h_node is not None
    # collapsed: only 1 child should remain under h, representing all 4 layers
    assert len(h_node.children) == 1
    assert h_node.children[0].repeat_count == 4
    assert h_node.children[0].repeat_group == "model.h"


def test_total_params_matches_real_model_after_collapse():
    model = _tiny_gpt2_model(n_layer=6)
    tree = build_tree(model, root_name="model")
    tree = collapse_repeats(tree, min_repeat=3)

    computed = model_total_params(tree)
    real = sum(p.numel() for p in model.parameters())
    assert computed == real


def test_below_min_repeat_does_not_collapse():
    # only 2 layers, min_repeat=3 -> nothing should collapse
    model = _tiny_gpt2_model(n_layer=2)
    tree = build_tree(model, root_name="model")
    tree = collapse_repeats(tree, min_repeat=3)

    h_node = tree.find("model.h")
    assert h_node is not None
    assert len(h_node.children) == 2
    assert all(c.repeat_count is None for c in h_node.children)


def test_structural_signature_distinguishes_different_shapes():
    # Two Linear layers with same param count but different shapes
    # must NOT be considered identical.
    import torch.nn as nn
    from llm_explorer.core.tree_builder import build_tree

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 32, bias=False)   # 2048 params
            self.b = nn.Linear(32, 64, bias=False)   # 2048 params, different shape

    tree = build_tree(Wrapper(), root_name="model")
    a_node = tree.find("model.a")
    b_node = tree.find("model.b")

    assert a_node.param_count == b_node.param_count  # same count
    assert _structural_signature(a_node) != _structural_signature(b_node)  # different shape


def test_non_consecutive_identical_modules_not_merged():
    # Two structurally identical Linear layers separated by a different
    # module should NOT be merged, since they're not consecutive.
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(16, 16)
            self.mid = nn.ReLU()
            self.c = nn.Linear(16, 16)

    tree = build_tree(Wrapper(), root_name="model")
    tree = collapse_repeats(tree, min_repeat=2)

    # a, mid, c are 3 distinct children -> no group of size >=2 is consecutive+identical
    assert len(tree.children) == 3
    assert all(c.repeat_count is None for c in tree.children)


def test_real_gpt2_small_config_end_to_end():
    """Sanity check against a config close to real gpt2 (12 layers)."""
    model = _tiny_gpt2_model(n_layer=12, n_embd=48, n_head=4)
    tree = build_tree(model, root_name="model")
    tree = collapse_repeats(tree, min_repeat=3)

    h_node = tree.find("model.h")
    assert len(h_node.children) == 1
    assert h_node.children[0].repeat_count == 12

    assert model_total_params(tree) == sum(p.numel() for p in model.parameters())
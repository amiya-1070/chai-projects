import pytest
from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, PeftConfig
from llm_explorer.core.resource_estimator import (
    estimate,
    _peft_trainable_params,
    DTYPE_BYTES,
    ADAM_STATE_MULTIPLIER,
)


# ---------- Helpers ----------

def make_linear_node(name, out_features, in_features, dtype="float16", bias=False):
    shape_info = {"weight": (out_features, in_features)}
    if bias:
        shape_info["bias"] = (out_features,)
    return LayerNode(
        name=name,
        local_name=name.split(".")[-1],
        module_type="Linear",
        param_count=out_features * in_features + (out_features if bias else 0),
        shape_info=shape_info,
        dtype=dtype,
    )


def make_container(name, children):
    return LayerNode(name=name, local_name=name.split(".")[-1],
                      module_type="Container", children=children)


# ---------- _peft_trainable_params: exact formula checks ----------

def test_lora_param_count():
    node = make_linear_node("q_proj", out_features=2048, in_features=2048)
    config = PeftConfig(method="LoRA", rank=8)
    count, warning = _peft_trainable_params(node, config)
    # A: (8, 2048), B: (2048, 8) -> 8*2048 + 2048*8
    assert count == 8 * 2048 + 2048 * 8 == 32768
    assert warning is None


def test_qlora_same_trainable_count_as_lora():
    # QLoRA's difference is in frozen-weight quantization, not adapter size
    node = make_linear_node("q_proj", out_features=2048, in_features=2048)
    lora_count, _ = _peft_trainable_params(node, PeftConfig(method="LoRA", rank=8))
    qlora_count, _ = _peft_trainable_params(node, PeftConfig(method="QLoRA", rank=8))
    assert lora_count == qlora_count


def test_dora_adds_magnitude_vector_on_top_of_lora():
    node = make_linear_node("gate_proj", out_features=8192, in_features=2048)
    rank = 8
    lora_count, _ = _peft_trainable_params(node, PeftConfig(method="LoRA", rank=rank))
    dora_count, warning = _peft_trainable_params(node, PeftConfig(method="DoRA", rank=rank))
    assert dora_count == lora_count + 8192  # +1 magnitude scalar per output feature
    assert warning is None


def test_adalora_returns_warning_about_dynamic_rank():
    node = make_linear_node("v_proj", out_features=512, in_features=2048)
    count, warning = _peft_trainable_params(node, PeftConfig(method="AdaLoRA", rank=4))
    # P: r*in, Λ: r, Q: r*out
    expected = 4 * 2048 + 4 + 4 * 512
    assert count == expected
    assert warning is not None and "initial" in warning.lower()


def test_ia3_uses_out_features():
    node = make_linear_node("up_proj", out_features=8192, in_features=2048)
    count, warning = _peft_trainable_params(node, PeftConfig(method="IA3"))
    assert count == 8192
    assert warning is None  # not an attn k/v proj, so no warning expected


def test_ia3_warns_on_attention_kv_projections():
    node = make_linear_node("model.layers.0.self_attn.k_proj", out_features=512, in_features=2048)
    count, warning = _peft_trainable_params(node, PeftConfig(method="IA3"))
    assert count == 512
    assert warning is not None and "head-dim" in warning.lower()


def test_full_finetune_without_bias():
    node = make_linear_node("down_proj", out_features=2048, in_features=8192, bias=False)
    count, warning = _peft_trainable_params(node, PeftConfig(method="Full Finetune"))
    assert count == 2048 * 8192
    assert warning is None


def test_full_finetune_with_bias():
    node = make_linear_node("classifier", out_features=10, in_features=768, bias=True)
    count, warning = _peft_trainable_params(node, PeftConfig(method="Full Finetune"))
    assert count == 10 * 768 + 10


def test_non_linear_node_returns_zero():
    node = LayerNode(name="norm", local_name="norm", module_type="RMSNorm",
                      shape_info={"weight": (2048,)})  # 1D, not a Linear weight
    count, warning = _peft_trainable_params(node, PeftConfig(method="LoRA", rank=8))
    assert count == 0
    assert warning is None


def test_prefix_tuning_flagged_unsupported():
    node = make_linear_node("q_proj", out_features=2048, in_features=2048)
    count, warning = _peft_trainable_params(node, PeftConfig(method="Prefix Tuning"))
    assert count == 0
    assert warning is not None


# ---------- estimate(): integration across a small tree ----------

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


def test_estimate_no_selection_all_frozen_no_trainable():
    root, *_ = build_test_tree()
    selection = SelectionState()
    result = estimate(root, selection)

    expected_total = 2048 * 2048 + 512 * 2048 + 8192 * 2048
    assert result.total_params == expected_total
    assert result.active_params == expected_total
    assert result.trainable_params == 0
    assert result.trainable_pct == 0.0
    assert result.pruned_node_count == 0
    assert result.peft_target_count == 0
    assert result.warnings == []


def test_estimate_with_lora_target_on_single_node():
    root, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8))

    result = estimate(root, selection)

    expected_trainable = 8 * (2048 + 2048)
    assert result.trainable_params == expected_trainable
    assert result.peft_target_count == 1
    assert result.trainable_pct == pytest.approx(
        expected_trainable / result.active_params * 100
    )
    # optimizer state = trainable * 8 bytes (Adam fp32 momentum+variance)
    assert result.optimizer_state_bytes == expected_trainable * ADAM_STATE_MULTIPLIER


def test_estimate_pruning_excludes_subtree_entirely():
    root, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()

    # Prune the whole self_attn block under layer 0
    selection.set_pruned("model.layers.0.self_attn", True)

    result = estimate(root, selection)

    # active params should now only include mlp.gate_proj
    assert result.active_params == 8192 * 2048
    # total_params should be unaffected — total means "if nothing were pruned"
    expected_total = 2048 * 2048 + 512 * 2048 + 8192 * 2048
    assert result.total_params == expected_total
    assert result.pruned_node_count >= 1  # self_attn itself counted (children implicitly excluded via active_params)


def test_estimate_pruning_a_peft_target_removes_its_trainable_params():
    root, q_proj, k_proj, gate_proj = build_test_tree()
    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8))

    # now prune the parent of q_proj — set_pruned on self_attn should
    # override/exclude q_proj's peft targeting since it's now inactive
    selection.set_pruned("model.layers.0.self_attn", True)

    result = estimate(root, selection)
    assert result.trainable_params == 0
    assert result.peft_target_count == 0  # q_proj is unreachable as an active PEFT target


def test_estimate_repeat_group_multiplies_correctly():
    q_proj = make_linear_node("model.layers.0.self_attn.q_proj", 2048, 2048)
    attn = make_container("model.layers.0.self_attn", [q_proj])
    layer0 = make_container("model.layers.0", [attn])
    layer0.repeat_count = 16  # simulate a collapsed 16-layer stack
    layers = make_container("model.layers", [layer0])
    root = make_container("model", [layers])

    selection = SelectionState()
    selection.set_peft_target(q_proj.name, PeftConfig(method="LoRA", rank=8))

    result = estimate(root, selection)

    # LoRA on q_proj should be counted once per repeated layer (x16)
    per_layer = 8 * (2048 + 2048)
    assert result.trainable_params == per_layer * 16
    assert result.peft_target_count == 16
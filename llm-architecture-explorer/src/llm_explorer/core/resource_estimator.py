from __future__ import annotations
from dataclasses import dataclass, field
from .layer_node import LayerNode
from .selection_state import SelectionState, NodeMode, PeftConfig




DTYPE_BYTES = {
    "float32": 4, "float": 4,
    "float16": 2, "half": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1,
    "int4": 0.5,
}

# Optimizer state multiplier (Adam: momentum + variance, both fp32) per
# trainable param, on top of the param itself. Used for full finetune VRAM.
ADAM_STATE_MULTIPLIER = 8  # bytes/param: 2 fp32 buffers x 4 bytes


@dataclass
class ResourceEstimate:
    total_params: int
    active_params: int          # total minus pruned subtrees
    trainable_params: int       # params actually updated during training
    trainable_pct: float
    frozen_param_memory_bytes: int
    trainable_param_memory_bytes: int
    optimizer_state_bytes: int
    estimated_vram_bytes: int   # rough: frozen (inference dtype) + trainable + optimizer state
    pruned_node_count: int
    peft_target_count: int


def _dtype_bytes(dtype: str | None) -> float:
    if dtype is None:
        return 4.0  # assume fp32 if unknown
    return DTYPE_BYTES.get(dtype, 4.0)



DTYPE_BYTES = {
    "float32": 4, "float": 4,
    "float16": 2, "half": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1,
    "int4": 0.5,
}

# Optimizer state multiplier (Adam: momentum + variance, both fp32) per
# trainable param, on top of the param itself. Used for full finetune VRAM.
ADAM_STATE_MULTIPLIER = 8  # bytes/param: 2 fp32 buffers x 4 bytes


@dataclass
class ResourceEstimate:
    total_params: int
    active_params: int
    trainable_params: int
    trainable_pct: float
    frozen_param_memory_bytes: int
    trainable_param_memory_bytes: int
    optimizer_state_bytes: int
    estimated_vram_bytes: int
    pruned_node_count: int
    peft_target_count: int
    warnings: list[str] = field(default_factory=list)


def _dtype_bytes(dtype: str | None) -> float:
    if dtype is None:
        return 4.0  # assume fp32 if unknown
    return DTYPE_BYTES.get(dtype, 4.0)


def _peft_trainable_params(node: LayerNode, config: PeftConfig) -> tuple[int, str | None]:
    """Estimate trainable param count added by a PEFT method targeting this node.

    Returns (param_count, warning). warning is None if the estimate is exact
    for the given assumptions, or a short string explaining a known
    approximation if not.

    Assumes node is a Linear-like layer with shape_info["weight"] = (out, in).
    Returns (0, None) for non-Linear targets — rank-decomposition methods
    don't apply the same way to norm/embedding layers in this model.
    """
    weight_shape = node.shape_info.get("weight")
    if weight_shape is None or len(weight_shape) != 2:
        return 0, None
    out_features, in_features = weight_shape
    has_bias = "bias" in node.shape_info

    if config.method in ("LoRA", "QLoRA"):
        return config.rank * (in_features + out_features), None

    if config.method == "DoRA":
        lora = config.rank * (in_features + out_features)
        magnitude = out_features
        return lora + magnitude, None

    if config.method == "AdaLoRA":
        initial_budget = config.rank * in_features + config.rank + config.rank * out_features
        return initial_budget, "AdaLoRA rank shrinks during training; this is the initial upper bound, not the final count"

    if config.method == "IA3":
        warning = None
        if "attn" in node.name.lower() and node.local_name in ("k_proj", "v_proj"):
            warning = "IA3 on attention K/V projections may use head-dim-scoped vectors in some implementations; this estimate uses raw out_features"
        return out_features, warning

    if config.method == "Full Finetune":
        weight_params = out_features * in_features
        bias_params = out_features if has_bias else 0
        return weight_params + bias_params, None

    return 0, f"Unsupported method for per-module estimation: {config.method}"


def estimate(tree: LayerNode, selection: SelectionState) -> ResourceEstimate:
    total_params = 0
    active_params = 0
    trainable_params = 0
    frozen_memory = 0
    trainable_memory = 0
    pruned_count = 0
    peft_count = 0

    def walk(node: LayerNode, repeat_multiplier: int = 1):
        nonlocal total_params, active_params, trainable_params
        nonlocal frozen_memory, trainable_memory, pruned_count, peft_count

        mult = repeat_multiplier * (node.repeat_count or 1)

        own_params = node.param_count * mult
        total_params += own_params

        pruned = selection.is_descendant_pruned(node.name)
        if pruned:
            pruned_count += mult
        else:
            active_params += own_params

            mode = selection.get_mode(node.name)
            if mode == NodeMode.PEFT_TARGET:
                config = selection.peft_configs.get(node.name, PeftConfig())
                added, warning = _peft_trainable_params(node, config)
                added *= mult
                trainable_params += added
                peft_count += mult
                if warning:
                    warnings.append(f"{node.name}: {warning}")
                trainable_memory += added * 4
                frozen_memory += own_params * _dtype_bytes(node.dtype)
            else:
                frozen_memory += own_params * _dtype_bytes(node.dtype)

        for child in node.children:
            walk(child, repeat_multiplier=mult)

    walk(tree)

    optimizer_state_bytes = int(trainable_params * ADAM_STATE_MULTIPLIER)
    estimated_vram = int(frozen_memory + trainable_memory + optimizer_state_bytes)
    trainable_pct = (trainable_params / active_params * 100) if active_params else 0.0

    return ResourceEstimate(
        total_params=total_params,
        active_params=active_params,
        trainable_params=trainable_params,
        trainable_pct=trainable_pct,
        frozen_param_memory_bytes=int(frozen_memory),
        trainable_param_memory_bytes=int(trainable_memory),
        optimizer_state_bytes=optimizer_state_bytes,
        estimated_vram_bytes=estimated_vram,
        pruned_node_count=pruned_count,
        peft_target_count=peft_count,
    )
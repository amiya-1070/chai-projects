from __future__ import annotations
from collections import defaultdict
import torch.nn as nn

from .layer_node import LayerNode


def _shape_info_for_module(module: nn.Module) -> dict:
    info = {}
    for pname, p in module.named_parameters(recurse=False):
        info[pname] = tuple(p.shape)
    return info


def _dtype_for_module(module: nn.Module) -> str | None:
    for p in module.parameters(recurse=False):
        return str(p.dtype).replace("torch.", "")
    return None


def build_tree(model: nn.Module, root_name: str = "model") -> LayerNode:
    """Walk the module tree and build a LayerNode tree.

    Each LayerNode's param_count/trainable_param_count reflects ONLY that
    module's own direct parameters (recurse=False) — children hold their
    own params separately. Use model_total_params() for subtree totals.
    """

    def build(module: nn.Module, name: str) -> LayerNode:
        own_params = list(module.parameters(recurse=False))
        param_count = sum(p.numel() for p in own_params)
        trainable_count = sum(p.numel() for p in own_params if p.requires_grad)

        node = LayerNode(
            name=name,
            local_name=name.split(".")[-1] if "." in name else name,
            module_type=type(module).__name__,
            param_count=param_count,
            trainable_param_count=trainable_count,
            shape_info=_shape_info_for_module(module),
            dtype=_dtype_for_module(module),
        )

        for child_name, child_module in module.named_children():
            qualified = f"{name}.{child_name}" if name else child_name
            node.children.append(build(child_module, qualified))

        return node

    return build(model, root_name)


def _structural_signature(node: LayerNode) -> tuple:
    """A recursive fingerprint of a subtree's *shape*, independent of names.

    Two subtrees have the same signature iff they have the same module_type
    at every position, the same parameter shapes (not just counts — shape
    matters, since e.g. a (4096,4096) Linear and a (2048,8192) Linear can
    have equal param counts but very different roles/behavior), and the
    same tree structure (same number of children, recursively identical).

    This is what actually justifies calling two nodes "identical" for
    collapsing purposes — param-count matching alone is a coincidence-prone
    proxy (e.g. an 8-head attn block and an unrelated module could tie on
    count but differ in shape).
    """
    own_shapes = tuple(sorted(node.shape_info.items()))
    child_sigs = tuple(_structural_signature(c) for c in node.children)
    return (node.module_type, own_shapes, child_sigs)


def collapse_repeats(node: LayerNode, min_repeat: int = 3) -> LayerNode:
    """Detect runs of structurally-identical children (typically nn.ModuleList
    transformer blocks) and collapse them into a single representative node.

    Identity is decided by _structural_signature: full recursive match of
    module types + parameter shapes, not just a param-count proxy.

    Only *consecutive* runs are collapsed — this matches the real case
    (an nn.ModuleList of transformer blocks are already consecutive
    children of their parent module) and avoids accidentally merging
    non-contiguous, unrelated modules that happen to share a signature.

    The representative node keeps children[0]'s full subtree (so it can
    still be expanded to inspect real shapes) and gets repeat_count set.
    Use model_total_params() to get true totals across all repeats.
    """
    # Recurse first so nested repeat groups are handled bottom-up
    node.children = [collapse_repeats(c, min_repeat) for c in node.children]

    if len(node.children) < min_repeat:
        return node

    sigs = [_structural_signature(c) for c in node.children]

    groups: list[list[LayerNode]] = []
    group_sigs: list = []
    for child, sig in zip(node.children, sigs):
        if groups and group_sigs[-1] == sig:
            groups[-1].append(child)
        else:
            groups.append([child])
            group_sigs.append(sig)

    new_children: list[LayerNode] = []
    for group in groups:
        if len(group) >= min_repeat:
            rep = group[0]
            rep.repeat_group = node.name
            rep.repeat_count = len(group)
            new_children.append(rep)
        else:
            new_children.extend(group)

    node.children = new_children
    return node


def model_total_params(node: LayerNode) -> int:
    """Total params across the whole tree, correctly accounting for
    collapsed repeat groups (multiplies the representative's subtree
    total by repeat_count)."""
    own = node.param_count
    children_total = sum(model_total_params(c) for c in node.children)
    subtotal = own + children_total
    if node.repeat_count:
        return subtotal * node.repeat_count
    return subtotal
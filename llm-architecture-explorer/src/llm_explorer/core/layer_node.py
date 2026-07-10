from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LayerNode:
    """A single node in the model's module tree.

    One LayerNode == one nn.Module, matched with its own parameters
    (not children's). Children are nested recursively.
    """

    name: str                      # qualified name, e.g. "model.layers.0.self_attn.q_proj"
    local_name: str                # last path segment, e.g. "q_proj"
    module_type: str               # class name, e.g. "Linear", "RMSNorm"
    param_count: int = 0
    trainable_param_count: int = 0
    shape_info: dict = field(default_factory=dict)   # {"weight": (out, in), "bias": (out,)}
    dtype: str | None = None
    children: list["LayerNode"] = field(default_factory=list)

    # Populated during repeat-group collapsing (tree_builder.collapse_repeats)
    repeat_group: str | None = None      # qualified name of the parent ModuleList, if collapsed
    repeat_count: int | None = None      # how many identical siblings this represents

    def total_params(self) -> int:
        """Own params + all descendants. Assumes tree is not yet collapsed,
        or that collapsed nodes already carry the full subtotal (see tree_builder)."""
        return self.param_count + sum(c.total_params() for c in self.children)

    def find(self, qualified_name: str) -> "LayerNode | None":
        if self.name == qualified_name:
            return self
        for c in self.children:
            hit = c.find(qualified_name)
            if hit is not None:
                return hit
        return None
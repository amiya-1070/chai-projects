from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class NodeMode(Enum):
    ACTIVE = "active"       # untouched, normal
    PRUNED = "pruned"       # excluded from the model entirely
    PEFT_TARGET = "peft"    # selected as a fine-tuning target (LoRA/DoRA/etc.)


PEFT_METHODS = ("LoRA", "QLoRA", "DoRA", "IA3", "AdaLoRA", "Prefix Tuning", "Full Finetune")


@dataclass
class PeftConfig:
    method: str = "LoRA"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    # IA3 / prefix tuning ignore rank/alpha; kept generic here, refined per-method later


@dataclass
class SelectionState:
    """User-driven choices layered on top of a LayerNode tree.

    Keyed by LayerNode.name (qualified module name), so this is independent
    of the tree object itself — reloading/rebuilding the tree from the same
    model path keeps selections valid as long as names match.
    """

    modes: dict[str, NodeMode] = field(default_factory=dict)
    peft_configs: dict[str, PeftConfig] = field(default_factory=dict)

    def get_mode(self, node_name: str) -> NodeMode:
        return self.modes.get(node_name, NodeMode.ACTIVE)

    def set_pruned(self, node_name: str, pruned: bool):
        if pruned:
            self.modes[node_name] = NodeMode.PRUNED
            self.peft_configs.pop(node_name, None)
        else:
            self.modes.pop(node_name, None)

    def set_peft_target(self, node_name: str, config: PeftConfig | None = None):
        self.modes[node_name] = NodeMode.PEFT_TARGET
        self.peft_configs[node_name] = config or PeftConfig()

    def clear(self, node_name: str):
        self.modes.pop(node_name, None)
        self.peft_configs.pop(node_name, None)

    def peft_target_names(self) -> list[str]:
        return [n for n, m in self.modes.items() if m == NodeMode.PEFT_TARGET]

    def pruned_names(self) -> list[str]:
        return [n for n, m in self.modes.items() if m == NodeMode.PRUNED]

    def is_descendant_pruned(self, node_name: str) -> bool:
        """A node is effectively pruned if it or any ancestor is pruned."""
        parts = node_name.split(".")
        for i in range(len(parts)):
            ancestor = ".".join(parts[: i + 1])
            if self.modes.get(ancestor) == NodeMode.PRUNED:
                return True
        return False
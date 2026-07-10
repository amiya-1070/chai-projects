from __future__ import annotations
import dearpygui.dearpygui as dpg

from llm_explorer.core.model_loader import load_model_from_path, ModelLoadError
from llm_explorer.core.tree_builder import build_tree, collapse_repeats, model_total_params
from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, NodeMode, PeftConfig, PEFT_METHODS
from llm_explorer.core.resource_estimator import estimate


def _is_linear_like(node: LayerNode) -> bool:
    weight_shape = node.shape_info.get("weight")
    return weight_shape is not None and len(weight_shape) == 2


class ExplorerApp:
    def __init__(self):
        self.tree: LayerNode | None = None
        self.selected_node: LayerNode | None = None
        self.selection = SelectionState()
        self._tag_to_node: dict[str, LayerNode] = {}
        self._tag_counter = 0

    def _next_tag(self) -> str:
        self._tag_counter += 1
        return f"node_{self._tag_counter}"

    # ---------- Loading ----------

    def on_load_clicked(self, sender, app_data):
        path = dpg.get_value("path_input")
        dpg.set_value("status_text", "Loading...")

        try:
            model, config = load_model_from_path(path)
        except ModelLoadError as e:
            dpg.set_value("status_text", f"Error: {e}")
            return

        tree = build_tree(model, root_name="model")
        tree = collapse_repeats(tree, min_repeat=3)
        self.tree = tree
        self.selection = SelectionState()  # fresh selection state per model load
        self.selected_node = None

        arch = getattr(config, "architectures", ["unknown"])
        dpg.set_value("status_text", f"Loaded {arch[0] if arch else 'unknown'}")

        self._rebuild_tree_view()
        self._update_detail_panel()
        self._update_summary_panel()

    # ---------- Tree rendering ----------

    def _rebuild_tree_view(self):
        dpg.delete_item("tree_container", children_only=True)
        self._tag_to_node.clear()
        if self.tree is not None:
            self._render_node(self.tree, parent="tree_container")

    def _node_display_color(self, node: LayerNode) -> tuple[int, int, int]:
        mode = self.selection.get_mode(node.name)
        if self.selection.is_descendant_pruned(node.name):
            return (110, 110, 110)  # greyed out
        if mode == NodeMode.PEFT_TARGET:
            return (120, 200, 255)  # highlighted blue
        return (255, 255, 255)

    def _render_node(self, node: LayerNode, parent: str):
        tag = self._next_tag()
        self._tag_to_node[tag] = node

        label = node.local_name or node.name
        if node.repeat_count:
            label += f"  [x{node.repeat_count}]"
        label += f"   ({node.module_type})"

        mode = self.selection.get_mode(node.name)
        pruned = self.selection.is_descendant_pruned(node.name)
        if pruned:
            label = f"[PRUNED] {label}"
        elif mode == NodeMode.PEFT_TARGET:
            config = self.selection.peft_configs.get(node.name)
            label = f"[{config.method} r={config.rank}] {label}"

        color = self._node_display_color(node)

        row_group = f"{tag}_row"
        with dpg.group(horizontal=True, parent=parent, tag=row_group):
            if node.children:
                with dpg.tree_node(label=label, tag=tag, default_open=False) as tn:
                    pass
                with dpg.item_handler_registry() as handler:
                    dpg.add_item_clicked_handler(callback=self._on_node_clicked, user_data=tag)
                dpg.bind_item_handler_registry(tag, handler)
                dpg.bind_item_theme(tag, self._color_theme(color))
            else:

                dpg.add_selectable(label=label, tag=tag, width=300,
                                    callback=self._on_node_clicked, user_data=tag)
                dpg.bind_item_theme(tag, self._color_theme(color))

                # Inline controls only for leaf Linear-like nodes (the actual
                # PEFT/prune targets in practice — q_proj, k_proj, gate_proj, etc.)
                if _is_linear_like(node):
                    print(f"Creating prune/peft buttons for {node.name}")
                    dpg.add_button(label="Prune", small=True,
                                    callback=self._on_prune_toggle, user_data=tag)
                    dpg.add_button(label="PEFT", small=True,
                                    callback=self._on_peft_target_clicked, user_data=tag)

        if node.children:
            for child in node.children:
                self._render_node(child, parent=tag)

    def _color_theme(self, rgb: tuple[int, int, int]):
        # Dear PyGui themes must be created once and reused; creating a new
        # theme per node per rebuild is wasteful but functionally fine at
        # this tree size. Revisit if tree rebuilds become a perf bottleneck.
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, rgb, category=dpg.mvThemeCat_Core)
        return theme

    # ---------- Selection handling ----------

    def _on_node_clicked(self, sender, app_data, user_data):
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        self.selected_node = node
        self._update_detail_panel()

    def _on_prune_toggle(self, sender, app_data, user_data):
        print(f"PRUNE CLICKED: tag={user_data}")
        tag = user_data
        node = self._tag_to_node.get(tag)
        print(f"  resolved node: {node}")
        if node is None:
            return
        currently_pruned = self.selection.get_mode(node.name) == NodeMode.PRUNED
        self.selection.set_pruned(node.name, not currently_pruned)
        self.selected_node = node
        self._rebuild_tree_view()
        self._update_detail_panel()
        self._update_summary_panel()

    def _on_peft_target_clicked(self, sender, app_data, user_data):
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        self.selected_node = node
        # Selecting via the PEFT button opens the config form in the detail
        # panel rather than immediately committing a default config — the
        # user picks method/rank/alpha there, then applies.
        self._update_detail_panel(peft_config_mode=True)

    def _apply_peft_config(self, sender, app_data, user_data):
        node = self.selected_node
        if node is None:
            return
        method = dpg.get_value("peft_method_combo")
        rank = dpg.get_value("peft_rank_input")
        alpha = dpg.get_value("peft_alpha_input")
        dropout = dpg.get_value("peft_dropout_input")
        config = PeftConfig(method=method, rank=rank, alpha=alpha, dropout=dropout)
        self.selection.set_peft_target(node.name, config)
        self._rebuild_tree_view()
        self._update_detail_panel()
        self._update_summary_panel()

    def _clear_node_selection(self, sender, app_data, user_data):
        node = self.selected_node
        if node is None:
            return
        self.selection.clear(node.name)
        self._rebuild_tree_view()
        self._update_detail_panel()
        self._update_summary_panel()

    # ---------- Detail panel ----------

    def _update_detail_panel(self, peft_config_mode: bool = False):
        dpg.delete_item("detail_container", children_only=True)
        node = self.selected_node
        if node is None:
            dpg.add_text("Select a node to see details.", parent="detail_container")
            return

        def row(label, value):
            with dpg.group(horizontal=True, parent="detail_container"):
                dpg.add_text(f"{label}:", color=(150, 150, 150))
                dpg.add_text(str(value))

        is_container = len(node.children) > 0 and node.param_count == 0

        dpg.add_text(node.name, parent="detail_container", color=(255, 255, 100))
        dpg.add_separator(parent="detail_container")
        row("Type", node.module_type)

        if is_container:
            dpg.add_text(
                "(container module — no parameters of its own; "
                "see children / subtree total below)",
                parent="detail_container", color=(120, 120, 120), wrap=500,
            )
        else:
            row("Param count (this module)", f"{node.param_count:,}")
            row("Trainable params", f"{node.trainable_param_count:,}")
            row("Dtype", node.dtype or "—")

        if node.repeat_count:
            row("Repeats", f"x{node.repeat_count} (identical blocks)")

        if node.shape_info:
            dpg.add_separator(parent="detail_container")
            dpg.add_text("Parameters:", parent="detail_container")
            for pname, shape in node.shape_info.items():
                row(f"  {pname}", shape)

        subtree_total = node.total_params()
        if node.repeat_count:
            subtree_total *= node.repeat_count
        dpg.add_separator(parent="detail_container")
        row("Subtree total params", f"{subtree_total:,}")
        if node.children:
            row("Direct children", len(node.children))

        # Current selection state for this node
        mode = self.selection.get_mode(node.name)
        dpg.add_separator(parent="detail_container")
        row("Current state", mode.value)

        if mode != NodeMode.ACTIVE:
            dpg.add_button(label="Clear (reset to active)", parent="detail_container",
                            callback=self._clear_node_selection)

        # PEFT config form — shown when explicitly requested (PEFT button
        # clicked) or when this node is already a PEFT target (so its config
        # can be edited/reapplied).
        if _is_linear_like(node) and (peft_config_mode or mode == NodeMode.PEFT_TARGET):
            dpg.add_separator(parent="detail_container")
            dpg.add_text("PEFT configuration:", parent="detail_container")

            existing = self.selection.peft_configs.get(node.name, PeftConfig())
            dpg.add_combo(list(PEFT_METHODS), default_value=existing.method,
                          tag="peft_method_combo", parent="detail_container")
            dpg.add_input_int(label="rank", default_value=existing.rank,
                               tag="peft_rank_input", parent="detail_container", width=120)
            dpg.add_input_int(label="alpha", default_value=existing.alpha,
                               tag="peft_alpha_input", parent="detail_container", width=120)
            dpg.add_input_float(label="dropout", default_value=existing.dropout,
                                 tag="peft_dropout_input", parent="detail_container", width=120)
            dpg.add_button(label="Apply PEFT target", parent="detail_container",
                            callback=self._apply_peft_config)

    # ---------- Summary panel ----------

    def _update_summary_panel(self):
        dpg.delete_item("summary_container", children_only=True)
        if self.tree is None:
            return

        result = estimate(self.tree, self.selection)

        def row(label, value, color=(255, 255, 255)):
            with dpg.group(horizontal=True, parent="summary_container"):
                dpg.add_text(f"{label}:", color=(150, 150, 150))
                dpg.add_text(str(value), color=color)

        row("Total params", f"{result.total_params:,}")
        row("Active params (after pruning)", f"{result.active_params:,}")
        row("Trainable params", f"{result.trainable_params:,}",
            color=(120, 200, 255) if result.trainable_params else (255, 255, 255))
        row("Trainable %", f"{result.trainable_pct:.3f}%")
        row("Estimated VRAM (params + optimizer, no activations)",
            f"{result.estimated_vram_bytes / 1e9:.2f} GB")
        row("Pruned nodes", result.pruned_node_count)
        row("PEFT targets", result.peft_target_count)

        if result.warnings:
            dpg.add_separator(parent="summary_container")
            dpg.add_text("Warnings:", parent="summary_container", color=(255, 180, 80))
            for w in result.warnings:
                dpg.add_text(f"  • {w}", parent="summary_container",
                             color=(255, 180, 80), wrap=500)


def main():
    dpg.create_context()
    dpg.create_viewport(title="LLM Architecture Explorer")

    app = ExplorerApp()

    with dpg.window(label="Main", tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="path_input",
                hint="Path to local HF model directory (contains config.json)",
                width=600,
            )
            dpg.add_button(label="Load", callback=app.on_load_clicked)
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=550, tag="tree_container"):
                pass
            with dpg.child_window(width=450, tag="detail_container"):
                dpg.add_text("Select a node to see details.")
            with dpg.child_window(tag="summary_container"):
                dpg.add_text("Load a model to see resource estimates.")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
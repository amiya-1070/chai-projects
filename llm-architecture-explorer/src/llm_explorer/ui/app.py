from __future__ import annotations
import dearpygui.dearpygui as dpg

from llm_explorer.core.model_loader import load_model_from_path, ModelLoadError
from llm_explorer.core.tree_builder import build_tree, collapse_repeats, model_total_params
from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, NodeMode, PeftConfig, PEFT_METHODS
from llm_explorer.core.resource_estimator import estimate
from llm_explorer.codegen.notebook_builder import build_notebook, save_notebook



def _is_linear_like(node: LayerNode) -> bool:
    weight_shape = node.shape_info.get("weight")
    return (
        node.module_type == "Linear"
        and weight_shape is not None
        and len(weight_shape) == 2
    )

class ExplorerApp:
    def __init__(self):
        self.tree: LayerNode | None = None
        self.model = None  # NEW: keep the loaded model around so we can rebuild the tree on toggle
        self.selected_node: LayerNode | None = None
        self.selection = SelectionState()
        self._tag_to_node: dict[str, LayerNode] = {}
        self._node_to_tag: dict[str, str] = {}
        self._tag_counter = 0
        self.collapse_layers = True  # NEW: current tree view mode
        self.global_peft_config = PeftConfig()  # NEW: single shared LoRA config for all targets
        self._diagram_expanded: set[str] = set()
        self._diagram_box_positions: dict[str, tuple[int, int, int, int]] = {}

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
        
        self.model = model
        self.selection = SelectionState()
        self.selected_node = None
        self.global_peft_config = PeftConfig()  # NEW: reset to defaults on fresh model load
        self._tag_to_node.clear()
        self._node_to_tag.clear()
        self._tag_counter = 0
        

        if dpg.does_item_exist("summary_results_container"):
            dpg.delete_item("summary_container", children_only=True)  # force full rebuild of PEFT controls too

        arch = getattr(config, "architectures", ["unknown"])
        dpg.set_value("status_text", f"Loaded {arch[0] if arch else 'unknown'}")

        self._rebuild_tree_from_model()
        self._render_diagram()

    # ---------- Tree rendering ----------
    def _refresh_node_display(self, node: LayerNode):
        """Update just this node's label/color in place — no rebuild, no
        expand-state loss, since the underlying DPG item is never destroyed."""
        tag = self._node_to_tag.get(node.name)
        if tag is None or not dpg.does_item_exist(tag):
            return
        label, color = self._compute_label_and_color(node)
        dpg.configure_item(tag, label=label)
        dpg.bind_item_theme(tag, self._color_theme(color))


    def _node_display_color(self, node: LayerNode) -> tuple[int, int, int]:
        mode = self.selection.get_mode(node.name)
        if self.selection.is_descendant_pruned(node.name):
            return (110, 110, 110)  # greyed out
        if mode == NodeMode.PEFT_TARGET:
            return (120, 200, 255)  # highlighted blue
        return (255, 255, 255)
    
    def _rebuild_tree_from_model(self):
        if self.model is None:
            return
        tree = build_tree(self.model, root_name="model")
        if self.collapse_layers:
            tree = collapse_repeats(tree, min_repeat=3)
        self.tree = tree

        self._tag_to_node.clear()
        self._node_to_tag.clear()
        self._tag_counter = 0
        dpg.delete_item("tree_container", children_only=True)
        self._render_node(self.tree, parent="tree_container")
        self._update_detail_panel()
        self._update_summary_panel()

    def _on_collapse_toggle(self, sender, app_data, user_data):
        self.collapse_layers = app_data  # checkbox's bool value
        self._rebuild_tree_from_model()

    def _render_node(self, node: LayerNode, parent: str):
        tag = self._next_tag()
        self._tag_to_node[tag] = node
        self._node_to_tag[node.name] = tag  

        label, color = self._compute_label_and_color(node)

        with dpg.group(horizontal=True, parent=parent):
            if node.children:
                with dpg.tree_node(label=label, tag=tag, default_open=False):
                    pass
                with dpg.item_handler_registry() as handler:
                    dpg.add_item_clicked_handler(callback=self._on_node_clicked, user_data=tag)
                dpg.bind_item_handler_registry(tag, handler)
                dpg.bind_item_theme(tag, self._color_theme(color))
            else:
                dpg.add_selectable(label=label, tag=tag, width=300,
                                    callback=self._on_node_clicked, user_data=tag)
                dpg.bind_item_theme(tag, self._color_theme(color))

                if _is_linear_like(node):
                    dpg.add_button(label="Prune", small=True,
                                    callback=self._on_prune_toggle, user_data=tag)
                    dpg.add_button(label="PEFT", small=True,
                                    callback=self._on_peft_target_clicked, user_data=tag)

        if node.children:
            for child in node.children:
                self._render_node(child, parent=tag)

    def _compute_label_and_color(self, node: LayerNode) -> tuple[str, tuple[int, int, int]]:
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
        return label, color

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
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        currently_pruned = self.selection.get_mode(node.name) == NodeMode.PRUNED
        self.selection.set_pruned(node.name, not currently_pruned)
        self.selected_node = node
        self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()
        self._render_diagram()

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
        self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()
        self._render_diagram()

    def _clear_node_selection(self, sender, app_data, user_data):
        node = self.selected_node
        if node is None:
            return
        self.selection.clear(node.name)
        self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()
        self._render_diagram()

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
        
    def _on_peft_target_clicked(self, sender, app_data, user_data):
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        currently_targeted = self.selection.get_mode(node.name) == NodeMode.PEFT_TARGET
        if currently_targeted:
            self.selection.clear(node.name)
        else:
            self.selection.set_peft_target(node.name, self.global_peft_config)
        self.selected_node = node
        self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()

    # ---------- Summary panel ----------

    def _build_peft_config_controls(self):
        """Create the global PEFT config input widgets once. These are never
        deleted/recreated afterward — only their values are updated via
        set_value, so they never lose keyboard focus while being edited."""
        dpg.add_text("PEFT configuration (applies to all targeted layers):",
                     parent="summary_container")
        dpg.add_combo(list(PEFT_METHODS), default_value=self.global_peft_config.method,
                      tag="global_peft_method", parent="summary_container",
                      callback=self._on_global_peft_config_changed)
        dpg.add_input_int(label="rank", default_value=self.global_peft_config.rank,
                           tag="global_peft_rank", parent="summary_container", width=120,
                           callback=self._on_global_peft_config_changed)
        dpg.add_input_int(label="alpha", default_value=self.global_peft_config.alpha,
                           tag="global_peft_alpha", parent="summary_container", width=120,
                           callback=self._on_global_peft_config_changed)
        dpg.add_input_float(label="dropout", default_value=self.global_peft_config.dropout,
                             tag="global_peft_dropout", parent="summary_container", width=120,
                             callback=self._on_global_peft_config_changed)
        dpg.add_separator(parent="summary_container")

        # Everything below this point IS destroyed/recreated on refresh —
        # give it its own child container so the PEFT inputs above are
        # untouched by that churn.
        dpg.add_child_window(tag="summary_results_container", parent="summary_container",
                              height=-80)  # leave room for the notebook-save controls below

        dpg.add_separator(parent="summary_container")
        dpg.add_input_text(label="Save as", default_value="finetune_notebook.ipynb",
                            tag="notebook_filename_input", parent="summary_container", width=250)
        dpg.add_button(label="Generate Notebook", parent="summary_container",
                        callback=self._on_generate_notebook)
        dpg.add_text("", tag="notebook_status_text", parent="summary_container")

    def _update_summary_panel(self):
        if self.tree is None:
            return
        if not dpg.does_item_exist("summary_results_container"):
            # First time — build the static controls once.
            self._build_peft_config_controls()

        dpg.delete_item("summary_results_container", children_only=True)

        result = estimate(self.tree, self.selection)

        def row(label, value, color=(255, 255, 255)):
            with dpg.group(horizontal=True, parent="summary_results_container"):
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

        row("PEFT targets", result.peft_target_count)

        peft_names = self.selection.peft_target_names()
        if peft_names:
            dpg.add_separator(parent="summary_results_container")
            dpg.add_text("PEFT-targeted layers:", parent="summary_results_container")

            grouped = self._group_peft_targets_by_layer_name(peft_names)
            for layer_type, layer_indices in sorted(grouped.items()):
                indices_str = ", ".join(str(i) for i in sorted(layer_indices, key=lambda x: (x is None, x)))
                dpg.add_text(f"  {layer_type}: layer #{indices_str}",
                             parent="summary_results_container")

        if result.warnings:
            dpg.add_separator(parent="summary_results_container")
            dpg.add_text("Warnings:", parent="summary_results_container", color=(255, 180, 80))
            for w in result.warnings:
                dpg.add_text(f"  • {w}", parent="summary_results_container",
                             color=(255, 180, 80), wrap=500)
                

    def _group_peft_targets_by_layer_name(self, node_names: list[str]) -> dict[str, list]:
        import re
        grouped: dict[str, list] = {}

        for name in node_names:
            match = re.search(r"\.layers\.(\d+)\.(.+)$", name)
            if match:
                layer_index = int(match.group(1))
                layer_type = match.group(2)  # e.g. "self_attn.q_proj"
            else:
                layer_index = None
                # not part of a repeated layer stack (e.g. "model.embed_tokens")
                layer_type = name.split(".", 1)[-1] if "." in name else name

            grouped.setdefault(layer_type, []).append(layer_index)

        return grouped


    def _on_global_peft_config_changed(self, sender, app_data, user_data):
        method = dpg.get_value("global_peft_method")
        rank = dpg.get_value("global_peft_rank")
        alpha = dpg.get_value("global_peft_alpha")
        dropout = dpg.get_value("global_peft_dropout")
        self.global_peft_config = PeftConfig(method=method, rank=rank, alpha=alpha, dropout=dropout)

        for node_name in self.selection.peft_target_names():
            self.selection.peft_configs[node_name] = self.global_peft_config

        self._update_summary_panel()  # only rebuilds summary_results_container now, not the inputs
        self._render_diagram()


    def _on_generate_notebook(self, sender, app_data, user_data):
        if self.tree is None:
            dpg.set_value("notebook_status_text", "Load a model first.")
            return

        filename = dpg.get_value("notebook_filename_input") or "finetune_notebook.ipynb"
        model_path = dpg.get_value("path_input")

        try:
            notebook = build_notebook(model_path, self.tree, self.selection)
            save_notebook(notebook, filename)
            dpg.set_value("notebook_status_text", f"Saved to {filename}")
        except Exception as e:
            dpg.set_value("notebook_status_text", f"Error: {e}")


    def _render_diagram(self):
        dpg.delete_item("diagram_drawlist", children_only=True)
        dpg.delete_item("diagram_buttons_layer", children_only=True)
        if self.tree is None:
            return

        self._diagram_box_positions = {}  # node.name -> (x, y, w, h), rebuilt each render

        box_w = 160
        box_h = 40
        x_gap = 40
        y_gap = 10

        # Recursively compute layout, then draw. Two passes: first compute
        # the vertical extent (total height) each subtree needs so parents
        # can be vertically centered against their children, then draw.
        def subtree_height(node: LayerNode) -> int:
            is_expanded = node.name in self._diagram_expanded
            if not node.children or not is_expanded:
                return box_h
            total = 0
            for child in node.children:
                total += subtree_height(child) + y_gap
            return max(total - y_gap, box_h)

        def draw_node(node: LayerNode, x: int, y_top: int) -> int:
            """Draws node and (if expanded) its children to the right.
            Returns the total height consumed."""
            is_expanded = node.name in self._diagram_expanded
            height = subtree_height(node)
            y_center = y_top + height // 2
            box_y = y_center - box_h // 2

            label, color = self._compute_label_and_color(node)
            display_label = node.local_name
            if node.repeat_count:
                display_label += f" ×{node.repeat_count}"

            fill_color = (60, 60, 60, 255)
            if self.selection.is_descendant_pruned(node.name):
                fill_color = (90, 90, 90, 255)
            elif self.selection.get_mode(node.name) == NodeMode.PEFT_TARGET:
                fill_color = (40, 90, 130, 255)

            dpg.draw_rectangle(
                (x, box_y), (x + box_w, box_y + box_h),
                color=(200, 200, 200, 255), fill=fill_color,
                parent="diagram_drawlist",
            )
            dpg.draw_text((x + 8, box_y + 12), display_label,
                          color=(255, 255, 255, 255), size=14,
                          parent="diagram_drawlist")

            if node.children:
                marker = "-" if is_expanded else "+"
                dpg.draw_text((x + box_w - 20, box_y + 12), marker,
                              color=(255, 255, 100, 255), size=16,
                              parent="diagram_drawlist")

            self._diagram_box_positions[node.name] = (x, box_y, box_w, box_h)

            if node.children and is_expanded:
                child_x = x + box_w + x_gap
                cursor_y = y_top
                for child in node.children:
                    child_height = draw_node(child, child_x, cursor_y)
                    cursor_y += child_height + y_gap

            return height

        total_height = draw_node(self.tree, 10, 10)

        # Resize the drawlist to fit content, so the panel's scrollbars
        # (from the enclosing child_window) actually have something
        # meaningful to scroll against, in both directions.
        max_x = max((x + w for x, y, w, h in self._diagram_box_positions.values()), default=800)
        dpg.configure_item("diagram_drawlist", width=int(max_x) + 20, height=int(total_height) + 20)
        drawlist_pos = dpg.get_item_pos("diagram_drawlist")
        dpg.set_item_pos("diagram_buttons_layer", drawlist_pos)  # force overlay instead of vertical stacking

        # Invisible click targets, one per drawn box, positioned exactly
        # over each rectangle — reuses ImGui's native button click handling
        # instead of hand-rolled mouse-position hit-testing.
        for node_name, (x, y, w, h) in self._diagram_box_positions.items():
            btn_tag = f"diagram_btn_{node_name}"
            dpg.add_button(label="", tag=btn_tag, width=w, height=h,
                            parent="diagram_buttons_layer",
                            callback=self._on_diagram_node_clicked, user_data=node_name)
            button_layer_pos = dpg.get_item_pos("diagram_buttons_layer")
            print(f"drawlist_pos: {drawlist_pos}, button_layer_pos: {button_layer_pos}")
            
            dpg.set_item_pos(btn_tag, (x + drawlist_pos[0], y + drawlist_pos[1]))
            with dpg.theme() as invisible_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 0, 0, 120), category=dpg.mvThemeCat_Core)  # TEMP: visible red for debugging
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 255, 30), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255, 60), category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(btn_tag, invisible_theme)

    def _on_diagram_node_clicked(self, sender, app_data, user_data):
        print(f"DIAGRAM CLICK: {user_data}")
        node_name = user_data
        node = self.tree.find(node_name) if self.tree else None
        if node is None or not node.children:
            return  # only expandable nodes toggle; leaves do nothing here
        if node_name in self._diagram_expanded:
            self._diagram_expanded.discard(node_name)
        else:
            self._diagram_expanded.add(node_name)
        self._render_diagram()

    

def main():
    dpg.create_context()
    dpg.create_viewport(title="LLM Architecture Explorer")

    app = ExplorerApp()

    with dpg.window(label="Main", tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="path_input", hint="...", width=600)
            dpg.add_button(label="Load", callback=app.on_load_clicked)
            dpg.add_checkbox(label="Collapse identical layers", default_value=True,
                              callback=app._on_collapse_toggle)
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=550, tag="tree_container"):
                pass

            # Right side: detail+summary on top (half height), diagram below
            with dpg.child_window(tag="right_side_container"):
                with dpg.group(horizontal=True, height=400):  # top half: existing two panels
                    with dpg.child_window(width=450, tag="detail_container"):
                        dpg.add_text("Select a node to see details.")
                    with dpg.child_window(tag="summary_container"):
                        dpg.add_text("Load a model to see resource estimates.")

                dpg.add_separator()
                with dpg.child_window(tag="diagram_container", horizontal_scrollbar=True):
                    dpg.add_drawlist(width=800, height=400, tag="diagram_drawlist")
                    dpg.add_group(tag="diagram_buttons_layer")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
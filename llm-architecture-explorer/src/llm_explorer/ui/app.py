from __future__ import annotations
import dearpygui.dearpygui as dpg

from llm_explorer.core.model_loader import load_model_from_path, ModelLoadError
from llm_explorer.core.tree_builder import build_tree, collapse_repeats, model_total_params
from llm_explorer.core.layer_node import LayerNode
from llm_explorer.core.selection_state import SelectionState, NodeMode, PeftConfig, PEFT_METHODS
from llm_explorer.core.resource_estimator import estimate
from llm_explorer.codegen.notebook_builder import build_notebook, save_notebook

import re

def _is_linear_like(node: LayerNode) -> bool:
    weight_shape = node.shape_info.get("weight")
    return (
        node.module_type == "Linear"
        and weight_shape is not None
        and len(weight_shape) == 2
    )

def _is_whole_layer(node: LayerNode) -> bool:
        """True if this node represents an entire repeated decoder layer
        (e.g. model.layers.7, or the collapsed representative model.layers.0),
        as opposed to a leaf module or an intermediate container like self_attn."""
        return bool(re.fullmatch(r".*\.layers\.\d+", node.name))

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
        self._diagram_dirty = False

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

    def _refresh_node_display_recursive(self, node: LayerNode):
        """Refreshes this node's display AND all its descendants' — needed
        when toggling prune state, since pruning a layer cascades visually
        to everything inside it (is_descendant_pruned already accounts for
        this in the resource estimate; this makes the tree's labels/colors
        reflect it too)."""
        self._refresh_node_display(node)
        for child in node.children:
            self._refresh_node_display_recursive(child)


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

                # NEW: Prune button for whole-layer branch nodes only
                if _is_whole_layer(node) and not self.collapse_layers:
                    dpg.add_button(label="Prune", small=True,
                                    callback=self._on_prune_toggle, user_data=tag)
            else:
                dpg.add_selectable(label=label, tag=tag, width=300,
                                    callback=self._on_node_clicked, user_data=tag)
                dpg.bind_item_theme(tag, self._color_theme(color))

                # PEFT stays leaf-level, same as before — just remove Prune from here
                if _is_linear_like(node):
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
        self._refresh_node_display_recursive(node)   # was: self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()
        self._diagram_dirty = True
        if dpg.does_item_exist("diagram_stale_indicator"):
            dpg.set_value("diagram_stale_indicator", "  (diagram out of date — click Update)")
        

    def _on_peft_target_clicked(self, sender, app_data, user_data):
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        

        if self.selection.is_descendant_pruned(node.name):
            dpg.set_value("status_text", f"Cannot PEFT-target '{node.local_name}' — it is pruned...")
            print("  -> BLOCKED")
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
    
    def set_pruned(self, node_name: str, pruned: bool):
        if pruned:
            self.modes[node_name] = NodeMode.PRUNED
            self.peft_configs.pop(node_name, None)
            # Also clear PEFT-target state from any descendant nodes —
            # otherwise a node PEFT-targeted before its parent layer was
            # pruned keeps a stale PEFT_TARGET entry that's inconsistent
            # with the tree/notebook now treating it as pruned.
            prefix = node_name + "."
            stale_descendants = [
                n for n in list(self.modes.keys())
                if n.startswith(prefix) and self.modes[n] == NodeMode.PEFT_TARGET
            ]
            for n in stale_descendants:
                self.modes.pop(n, None)
                self.peft_configs.pop(n, None)
        else:
            self.modes.pop(node_name, None)

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
        self._diagram_dirty = True
        if dpg.does_item_exist("diagram_stale_indicator"):
            dpg.set_value("diagram_stale_indicator", "  (diagram out of date — click Update)")
        

    def _clear_node_selection(self, sender, app_data, user_data):
        node = self.selected_node
        if node is None:
            return
        self.selection.clear(node.name)
        self._refresh_node_display(node)
        self._update_detail_panel()
        self._update_summary_panel()
        self._diagram_dirty = True
        if dpg.does_item_exist("diagram_stale_indicator"):
            dpg.set_value("diagram_stale_indicator", "  (diagram out of date — click Update)")

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
        self._diagram_dirty = True
        if dpg.does_item_exist("diagram_stale_indicator"):
            dpg.set_value("diagram_stale_indicator", "  (diagram out of date — click Update)")
        


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
        if self.tree is None:
            return

        embed = self.tree.find("model.embed_tokens") or self._find_by_suffix("embed_tokens")
        layers_container = self.tree.find("model.layers") or self._find_by_suffix("layers")
        norm = self.tree.find("model.norm") or self._find_by_suffix("norm")

        if layers_container is None or not layers_container.children:
            dpg.add_text("Diagram unavailable: expected a standard decoder-only "
                         "structure (embed_tokens -> layers -> norm) but couldn't "
                         "find a 'layers' block in this model.",
                         parent="diagram_drawlist", wrap=600, color=(255, 180, 80))
            return

        decoder_block = layers_container.children[0]  # representative block (collapsed or layers.0)
        self_attn = self._find_child(decoder_block, "self_attn")
        mlp = self._find_child(decoder_block, "mlp")

        if self_attn is None or mlp is None:
            dpg.add_text("Diagram unavailable: this doesn't look like a standard "
                         "decoder block (expected self_attn + mlp submodules).",
                         parent="diagram_drawlist", wrap=600, color=(255, 180, 80))
            return

        box_h = 40
        small_box_h = 30
        gap = 30
        x = 10

        def box(x, y, w, h, text, node=None, small=False):
            if small:
                fill = (55, 75, 95, 255)  # distinct muted blue-grey base for sub-component boxes
            else:
                fill = (60, 60, 60, 255)  # existing default for major stage boxes

            if node is not None:
                if self.selection.is_descendant_pruned(node.name):
                    fill = (90, 90, 90, 255)
                elif self.selection.get_mode(node.name) == NodeMode.PEFT_TARGET:
                    fill = (40, 90, 130, 255)

            dpg.draw_rectangle((x, y), (x + w, y + h), color=(200, 200, 200, 255),
                                fill=fill, parent="diagram_drawlist")
            dpg.draw_text((x + 6, y + h // 2 - 8), text, color=(255, 255, 255, 255),
                          size=12 if small else 14, parent="diagram_drawlist")
            return x + w

        def arrow(x1, y, x2):
            dpg.draw_arrow((x2, y), (x1, y), color=(180, 180, 180, 255), thickness=1.5,
                           parent="diagram_drawlist")

        y_main = 150  # vertical center for the main horizontal flow

        # ---- embed_tokens ----
        x_end = box(x, y_main, 120, box_h, "embed_tokens", node=embed)
        arrow(x_end, y_main + box_h // 2, x_end + gap)
        x = x_end + gap

        # ---- decoder block (expanded internals) ----
        block_x_start = x
        block_top = 20
        block_bottom = 280

        inner_x = block_x_start + 20
        inner_y = block_top + 35

        # input_layernorm
        norm1 = self._find_child(decoder_block, "input_layernorm") or self._find_child(decoder_block, "norm1")
        x_end = box(inner_x, inner_y, 150, small_box_h, "RMSNorm (pre-attn)", node=norm1, small=True)
        arrow(x_end, inner_y + small_box_h // 2, x_end + 15)
        inner_x = x_end + 15

        # self_attn internals
        attn_y = inner_y
        qkv_x = inner_x
        q_child = self._find_child(self_attn, "q_proj")
        k_child = self._find_child(self_attn, "k_proj")
        v_child = self._find_child(self_attn, "v_proj")

        q_y = attn_y
        k_y = attn_y + small_box_h + 5
        v_y = attn_y + 2 * (small_box_h + 5)

        q_end_x = box(qkv_x, q_y, 70, small_box_h, "q_proj", node=q_child, small=True) if q_child else qkv_x
        k_end_x = box(qkv_x, k_y, 70, small_box_h, "k_proj", node=k_child, small=True) if k_child else qkv_x
        v_end_x = box(qkv_x, v_y, 70, small_box_h, "v_proj", node=v_child, small=True) if v_child else qkv_x

        rope_x = max(q_end_x, k_end_x) + 15
        rope_y = k_y  # vertically between q and k, closer to their midpoint
        rope_end_x = box(rope_x, rope_y, 70, small_box_h, "RoPE\n(Q,K only)", small=True)

        # Q and K converge into RoPE — diagonal lines since they start at
        # different heights than RoPE's box.
        dpg.draw_line((q_end_x, q_y + small_box_h // 2), (rope_x, rope_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")
        dpg.draw_line((k_end_x, k_y + small_box_h // 2), (rope_x, rope_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")

        o_proj = self._find_child(self_attn, "o_proj")
        o_x = rope_end_x + 20
        o_y = k_y  # centered roughly against the q/k/v stack
        x_end = box(o_x, o_y, 70, small_box_h, "o_proj", node=o_proj, small=True)

        # RoPE output and V both converge into o_proj
        dpg.draw_line((rope_end_x, rope_y + small_box_h // 2), (o_x, o_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")
        dpg.draw_line((v_end_x, v_y + small_box_h // 2), (o_x, o_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")

        arrow(x_end, o_y + small_box_h // 2, x_end + 20)
        inner_x = x_end + 20

        # residual add #1 (ASSUMED — see disclaimer below)
        x_end = box(inner_x, inner_y + small_box_h, 40, small_box_h, "+", small=True)
        arrow(x_end, inner_y + small_box_h + small_box_h // 2, x_end + 15)
        inner_x = x_end + 15

        # post_attention_layernorm
        norm2 = self._find_child(decoder_block, "post_attention_layernorm") or self._find_child(decoder_block, "norm2")
        x_end = box(inner_x, inner_y + small_box_h, 150, small_box_h, "RMSNorm (pre-mlp)", node=norm2, small=True)
        arrow(x_end, inner_y + small_box_h + small_box_h // 2, x_end + 15)
        inner_x = x_end + 15

        # mlp internals: gate/up feeding into down
        gate = self._find_child(mlp, "gate_proj")
        up = self._find_child(mlp, "up_proj")
        down = self._find_child(mlp, "down_proj")

        gate_y = inner_y + small_box_h  # right below the residual-add/norm2 row
        up_y = gate_y + small_box_h + 5
        gu_x = inner_x

        gate_end_x = box(gu_x, gate_y, 70, small_box_h, "gate_proj", node=gate, small=True) if gate else gu_x
        up_end_x = box(gu_x, up_y, 70, small_box_h, "up_proj", node=up, small=True) if up else gu_x

        down_x = max(gate_end_x, up_end_x) + 15
        down_y = gate_y + (up_y - gate_y) // 2
        x_end = box(down_x, down_y, 70, small_box_h, "down_proj", node=down, small=True)

        dpg.draw_line((gate_end_x, gate_y + small_box_h // 2), (down_x, down_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")
        dpg.draw_line((up_end_x, up_y + small_box_h // 2), (down_x, down_y + small_box_h // 2),
                      color=(180, 180, 180, 255), thickness=1.5, parent="diagram_drawlist")

        arrow(x_end, down_y + small_box_h // 2, x_end + 15)

        # residual add #2 (ASSUMED)
        x_end = box(x_end + 15, down_y, 40, small_box_h, "+", small=True)

        dpg.draw_text((block_x_start + 8, block_bottom - 20),
                      "Note: residual adds (+) and their positions assume the "
                      "standard pre-norm decoder pattern; not verified against "
                      "this model's actual forward() code.",
                      color=(200, 160, 100, 255), size=11, parent="diagram_drawlist")

        
        block_content_end_x = x_end + 20

        dpg.draw_text((block_x_start + 8, block_bottom - 20),
                      "Note: residual adds (+) and their positions assume the "
                      "standard pre-norm decoder pattern; not verified against "
                      "this model's actual forward() code.",
                      color=(200, 160, 100, 255), size=11, parent="diagram_drawlist")

        dpg.draw_rectangle((block_x_start, block_top), (block_content_end_x, block_bottom),
                            color=(140, 140, 100, 255), parent="diagram_drawlist")
        repeat_label = "Decoder Block"
        if decoder_block.repeat_count:
            repeat_label += f"  (×{decoder_block.repeat_count})"
        dpg.draw_text((block_x_start + 8, block_top + 4), repeat_label,
                      color=(220, 220, 150, 255), size=15, parent="diagram_drawlist")

        block_x_end = block_content_end_x
        arrow(block_x_end, y_main + box_h // 2, block_x_end + gap)
        x = block_x_end + gap

        # ---- final norm ----
        x_end = box(x, y_main, 100, box_h, "norm", node=norm)
        arrow(x_end, y_main + box_h // 2, x_end + gap)
        x = x_end + gap

        # ---- lm_head (often tied to embeddings, may not exist as separate node) ----
        box(x, y_main, 100, box_h, "lm_head")

        dpg.configure_item("diagram_drawlist", width=x + 150, height=320)

    def _find_child(self, node: LayerNode, local_name: str) -> LayerNode | None:
        for c in node.children:
            if c.local_name == local_name:
                return c
        return None

    def _find_by_suffix(self, suffix: str) -> LayerNode | None:
        result = [None]
        def walk(n):
            if n.local_name == suffix:
                result[0] = n
            for c in n.children:
                walk(c)
        if self.tree:
            walk(self.tree)
        return result[0]

    def _on_diagram_toggle(self, sender, app_data, user_data):
        node_name = user_data
        if node_name in self._diagram_expanded:
            self._diagram_expanded.discard(node_name)
        else:
            self._diagram_expanded.add(node_name)
        self._render_diagram()

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

    
    def _render_layer_heatmap(self):
        dpg.delete_item("heatmap_drawlist", children_only=True)
        print(f"heatmap render: self.model={'set' if self.model else 'None'}")
        if self.model is None:
            return

        full_tree = build_tree(self.model, root_name="model")
        layers_container = self._find_child(full_tree, "layers") or self._find_by_suffix_in(full_tree, "layers")
        print(f"layers_container: {layers_container}, num layer_nodes: {len(layers_container.children) if layers_container else 0}")
        if layers_container is None or not layers_container.children:
            dpg.add_text("No repeated decoder layers found in this model.",
                         parent="heatmap_drawlist", color=(255, 180, 80))
            return

        layer_nodes = layers_container.children  # one real LayerNode per layer index, uncollapsed
        sublayer_paths = [
            ("self_attn", "q_proj"), ("self_attn", "k_proj"),
            ("self_attn", "v_proj"), ("self_attn", "o_proj"),
            ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
            ("input_layernorm",), ("post_attention_layernorm",),
        ]

        cell_w, cell_h = 26, 26
        label_col_w = 130
        header_row_h = 30

        # Row labels (sublayer types)
        for row_idx, path in enumerate(sublayer_paths):
            row_label = ".".join(path)
            y = header_row_h + row_idx * cell_h
            dpg.draw_text((5, y + 6), row_label, color=(220, 220, 220, 255),
                          size=11, parent="heatmap_drawlist")

        # Column headers (layer indices) + grid cells
        for col_idx, layer_node in enumerate(layer_nodes):
            x = label_col_w + col_idx * cell_w
            dpg.draw_text((x + 6, 5), str(col_idx), color=(180, 180, 180, 255),
                          size=11, parent="heatmap_drawlist")

            for row_idx, path in enumerate(sublayer_paths):
                target = layer_node
                for part in path:
                    target = self._find_child(target, part) if target else None
                if target is None:
                    continue  # this model doesn't have this sublayer (e.g. different naming) — leave blank

                fill = (55, 75, 95, 255)  # default active
                if self.selection.is_descendant_pruned(target.name):
                    fill = (90, 90, 90, 255)
                elif self.selection.get_mode(target.name) == NodeMode.PEFT_TARGET:
                    fill = (40, 90, 130, 255)

                y = header_row_h + row_idx * cell_h
                dpg.draw_rectangle((x, y), (x + cell_w - 2, y + cell_h - 2),
                                    color=(120, 120, 120, 255), fill=fill,
                                    parent="heatmap_drawlist")

        total_w = label_col_w + len(layer_nodes) * cell_w + 20
        total_h = header_row_h + len(sublayer_paths) * cell_h + 20
        dpg.configure_item("heatmap_drawlist", width=total_w, height=total_h)

    def _find_by_suffix_in(self, root: LayerNode, suffix: str) -> LayerNode | None:
        result = [None]
        def walk(n):
            if n.local_name == suffix:
                result[0] = n
            for c in n.children:
                walk(c)
        walk(root)
        return result[0]
    
    def _on_update_diagram_clicked(self, sender, app_data):
        self._render_diagram()
        self._render_layer_heatmap()
        self._diagram_dirty = False
        dpg.set_value("diagram_stale_indicator", "")

    

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
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Update Diagram", callback=app._on_update_diagram_clicked)
                        dpg.add_text("", tag="diagram_stale_indicator")
                    with dpg.tab_bar(tag="diagram_tab_bar"):
                        with dpg.tab(label="Architecture Diagram"):
                            dpg.add_drawlist(width=1200, height=350, tag="diagram_drawlist")
                        with dpg.tab(label="Layer Heatmap"):
                            dpg.add_drawlist(width=1200, height=350, tag="heatmap_drawlist")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
from __future__ import annotations
import dearpygui.dearpygui as dpg

from llm_explorer.core.model_loader import load_model_from_path, ModelLoadError
from llm_explorer.core.tree_builder import build_tree, collapse_repeats, model_total_params
from llm_explorer.core.layer_node import LayerNode


class ExplorerApp:
    def __init__(self):
        self.tree: LayerNode | None = None
        self.selected_node: LayerNode | None = None
        # Maps dpg tree-node tag -> LayerNode, so click callbacks can look up
        # which node was clicked without re-walking the tree.
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

        arch = getattr(config, "architectures", ["unknown"])
        total = model_total_params(tree)
        dpg.set_value(
            "status_text",
            f"Loaded {arch[0] if arch else 'unknown'} — {total:,} total params",
        )

        self._rebuild_tree_view()

    # ---------- Tree rendering ----------

    def _rebuild_tree_view(self):
        dpg.delete_item("tree_container", children_only=True)
        self._tag_to_node.clear()
        if self.tree is not None:
            self._render_node(self.tree, parent="tree_container")

    def _render_node(self, node: LayerNode, parent: str):
        tag = self._next_tag()
        self._tag_to_node[tag] = node

        label = node.local_name or node.name
        if node.repeat_count:
            label += f"  [x{node.repeat_count}]"
        label += f"   ({node.module_type})"

        if node.children:
            with dpg.tree_node(label=label, tag=tag, parent=parent, default_open=False):
                # Make the tree_node header itself clickable to select this node
                pass
            # selection handler bound after creation (tree_node doesn't take on_click directly)
            with dpg.item_handler_registry() as handler:
                dpg.add_item_clicked_handler(callback=self._on_node_clicked, user_data=tag)
            dpg.bind_item_handler_registry(tag, handler)

            for child in node.children:
                self._render_node(child, parent=tag)
        else:
            # Leaf node: render as selectable so it's visually distinct and clickable
            dpg.add_selectable(label=label, tag=tag, parent=parent,
                                callback=self._on_node_clicked, user_data=tag)

    def _on_node_clicked(self, sender, app_data, user_data):
        tag = user_data
        node = self._tag_to_node.get(tag)
        if node is None:
            return
        self.selected_node = node
        self._update_detail_panel()

    # ---------- Detail panel ----------

    def _update_detail_panel(self):
        dpg.delete_item("detail_container", children_only=True)
        node = self.selected_node
        if node is None:
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
                parent="detail_container",
                color=(120, 120, 120),
                wrap=500,
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


def main():
    dpg.create_context()
    dpg.create_viewport(title="LLM Architecture Explorer", width=1100, height=700)

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
            with dpg.child_window(tag="detail_container"):
                dpg.add_text("Select a node to see details.")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
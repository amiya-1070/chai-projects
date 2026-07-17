# LLM Architecture Explorer

A desktop app for visualizing, analyzing, pruning, and PEFT-configuring local Hugging Face Transformer models — without needing to download full model weights just to inspect architecture.

## What it does

- Loads a local HF model directory (just needs `config.json`) and builds an interactive tree view of its architecture — embeddings, transformer layers, attention/MLP submodules, norms.
- Click any node to see its parameter count, tensor shapes, and dtype.
- Toggle between a **collapsed view** (identical repeated layers shown once, e.g. "×16") and an **uncollapsed view** (every layer shown individually, by index).
- Mark leaf modules (`q_proj`, `gate_proj`, etc.) as **PEFT targets** (LoRA/QLoRA/DoRA/AdaLoRA/IA³) with a shared rank/alpha/dropout config, and see live trainable-parameter and estimated-VRAM numbers update.
- Mark whole decoder layers as **pruned** (only available in uncollapsed view, to avoid ambiguity) and see the resource estimate update accordingly.
- View a horizontal **architecture diagram** of one representative decoder block (embeddings → attention internals incl. RoPE → MLP → norm → output), plus a **layer heatmap** tab showing per-layer, per-sublayer PEFT/prune state across the whole model at a glance.
- Generate a complete, runnable **Jupyter notebook** for causal-LM fine-tuning, with your PEFT config and whole-layer pruning already wired in — model loading, LoRA setup, training loop, save/export.

## Requirements

- Python 3.10+
- A local HF model directory containing at least `config.json` (doesn't need to be a full weight download — the tool builds an architecture-only model via `AutoModel.from_config`, so it never downloads or requires weight files)

## Installation

### Linux

```bash
git clone <your-repo-url>
cd llm-architecture-explorer
pip install -e . --break-system-packages
```

(The `--break-system-packages` flag is needed on distros where `pip` refuses to install into the system Python environment directly — if you're using a virtual environment or conda env instead, you can drop that flag.)

**Recommended: use a conda/virtualenv** to keep dependencies isolated:
```bash
conda create -n llm-explorer python=3.11
conda activate llm-explorer
pip install -e .
```

### Windows

```powershell
git clone <your-repo-url>
cd llm-architecture-explorer
python -m venv venv
venv\Scripts\activate
pip install -e .
```

If you're using Anaconda/Miniconda instead of a plain venv:
```powershell
conda create -n llm-explorer python=3.11
conda activate llm-explorer
pip install -e .
```

No platform-specific build steps are needed beyond this — Dear PyGui ships prebuilt wheels for both Windows and Linux, so there's no compilation step.

## Running

### Linux
```bash
python3 scripts/run_app.py
```
Launch from a terminal so you can see any error output — some diagnostics (e.g. model-load failures) print to stdout/stderr rather than only showing in the GUI.

### Windows
```powershell
python scripts\run_app.py
```
Same note applies — run from a terminal/PowerShell window, not by double-clicking, so you can see console output if something goes wrong.

The app launches in fullscreen by default.

## Usage

1. **Load a model**: paste the full local path to a directory containing `config.json` into the path field at the top, and click **Load**.
   - Linux example: `/home/you/models/Llama-3.2-1B-Instruct`
   - Windows example: `C:\Users\you\models\Llama-3.2-1B-Instruct`
2. **Browse the tree** (left panel): click any node to see its details in the middle panel. Branch nodes with no parameters of their own (e.g. `self_attn`) are containers — their subtree total is shown at the bottom of the detail panel.
3. **Toggle "Collapse identical layers"** at the top to switch between the compact repeated-layer view and the full per-layer-index view.
4. **PEFT targeting**: click **PEFT** next to any Linear leaf module to mark it as a fine-tuning target using the current global PEFT config (method/rank/alpha/dropout), set in the rightmost panel. Changing the global config retroactively updates all currently-targeted nodes.
5. **Pruning**: switch off "Collapse identical layers" first, then click **Prune** on a specific decoder layer's box in the tree to mark that whole layer for removal.
6. **Diagram tab**: below the detail/summary panels, click **Update Diagram** to (re)render the architecture diagram and layer heatmap — these don't auto-refresh on every click, to avoid unnecessary redraw cost.
7. **Generate a notebook**: once your PEFT/pruning selections look right in the summary panel, set a filename and click **Generate Notebook**. The `.ipynb` is saved to the working directory you launched the app from.

## Known limitations

- Only LoRA/QLoRA/DoRA are actually codegenerated for the notebook's PEFT setup — AdaLoRA, IA³, and Prefix Tuning can be selected in the UI (for resource estimation) but fall back to a LoRA-equivalent target in the generated notebook, with a warning comment.
- Whole-layer pruning is applied in the generated notebook (layers are removed from the model before training). Sub-layer pruning is not offered in the UI for this reason — removing an individual projection (e.g. just `gate_proj`) without rewriting the layer's forward pass would break the model.
- Combining layer-specific PEFT targeting with pruning in the same run can produce a layer-index mismatch (pruning re-numbers remaining layers). The generated notebook includes a warning about this — double-check `layers_to_transform` values in the PEFT cell against the post-pruning layer count before training.
- VRAM estimates cover parameter + optimizer state memory only, not activation memory (which depends on batch size and sequence length, not just the model/config).
- Best tested against standard decoder-only architectures (Llama, Gemma, GPT-2-style). Encoder-only and encoder-decoder models will load and show an accurate tree/resource estimate, but the architecture diagram assumes a decoder-only block shape and may not render correctly for other architecture families.
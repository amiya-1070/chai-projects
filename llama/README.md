## Table of Contents

- [Llama Dashboard](#llama-dashboard)
- [Linux](#linux)
- [Windows](#windows)

LLAMA DASHBOARD
===================================

WHAT THIS IS
------------
A C++ desktop GUI (Dear ImGui + ImPlot + SQLite) for benchmarking local LLM
inference using llama.cpp. It runs llama-bench across different models,
thread counts, and quantization levels, logs results to a local SQLite
database, and plots throughput (tokens/sec) comparisons. It also supports
a "Model Size Sweep" that runs a ladder of models (e.g. 1B -> 3B -> 7B ->
14B) back-to-back and a "Perf Stat" tab that wraps each run with
`perf stat -d -d -d` for detailed CPU counter data.

Everything runs 100% locally — no API keys, no network calls for
inference. You just need llama.cpp built, some GGUF model files
downloaded, and this dashboard compiled.


REQUIREMENTS (both platforms)
------------------------------
- A C++17-capable compiler
- CMake >= 3.15
- GGUF model files (see "Getting models" below)
- ~10-20 GB free disk space depending on how many model sizes you want
  to test (a 14B model at Q4_K_M quantization alone is ~9 GB)

  
GETTING THE MODELS: LLAMA 3.2 1B INSTRUCT AND THE FINETUNED GENOMICS MODEL
----------------------------------------------------------------------------
This project uses each model in TWO different ways:

  1. GGUF format — for llama-bench (Benchmark and Model Size Sweep tabs)
  2. HuggingFace format — used by kl_helper.py (invoked from the
     Configuration panel) to compute KL divergence between the base
     and finetuned model's output distributions

These require separate setup steps.

--- Part A: GGUF setup (for benchmarking) ---

1. Download the HuggingFace model folders

   Requires `huggingface_hub` (pip install huggingface_hub).

   Base model — Llama-3.2-1B-Instruct:
       huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
           --local-dir ./models/llama-3.2-1B-Instruct

   (Note: Meta's Llama models are gated on HuggingFace — request
   access on the model page and log in via `huggingface-cli login`
   with an approved account before this will work.)

   Finetuned genomics model:
       huggingface-cli download coconutpdf/genomics-llama-1b \
           --local-dir ./models/genomics-llama-1b

2. Convert each to GGUF

   From inside your llama.cpp directory:

       pip install -r requirements.txt

       python convert_hf_to_gguf.py ../models/llama-3.2-1B-Instruct \
           --outfile ../models/llama-3.2-1b-instruct-f16.gguf \
           --outtype f16

       python convert_hf_to_gguf.py ../models/genomics-llama-1b \
           --outfile ../models/genomics-llama-1b-f16.gguf \
           --outtype f16

   Optionally quantize the result (e.g. to Q4_K_M) using llama.cpp's
   quantize tool — see the "Getting models" section above.

3. In the dashboard's Configuration panel ("Paths" section), set:

     - "Base model (.gguf)"      -> the converted base .gguf file
     - "Finetuned model (.gguf)" -> the converted finetuned .gguf file

--- Part B: HuggingFace-format setup (for KL divergence analysis) ---

kl_helper.py loads both models directly via HuggingFace Transformers,
NOT via the GGUF files — this is a separate, independent code path
used for the KL divergence feature specifically.

You do NOT need to manually download anything for this part. In the
Configuration panel, set:

     - "Base model HF ID"      -> meta-llama/Llama-3.2-1B-Instruct
     - "Finetuned model HF ID" -> coconutpdf/genomics-llama-1b
     - "KL helper script"      -> full path to kl_helper.py

The first time the KL feature is used, kl_helper.py will
automatically download both models from HuggingFace Hub via
`from_pretrained()` and cache them locally (default HF cache
location: ~/.cache/huggingface). This requires:

     pip install torch transformers

installed in whatever Python environment kl_helper.py runs under.
As with the GGUF download, meta-llama/Llama-3.2-1B-Instruct is
gated — the machine running kl_helper.py needs to be logged in via
`huggingface-cli login` with an approved account, or the automatic
download will fail with a 403/permission error.

NOTE: kl_helper.py loads both models in full (float16) directly into
CPU memory for inference — expect several GB of RAM usage per model
while the KL feature is active, in addition to whatever the
Benchmark/Sweep tabs are separately using via llama-bench.


LINUX
====================================

1. Install dependencies
------------------------
Debian/Ubuntu-based:

    sudo apt update
    sudo apt install build-essential cmake git \
        libglfw3-dev libgl1-mesa-dev libsqlite3-dev linux-tools-common \
        linux-tools-generic

(`linux-tools-*` provides the `perf` binary, needed for the Perf Stat
tab. Package name may vary slightly by distro/kernel version — if
`perf stat` doesn't work after install, search your distro's package
manager for "perf" or "linux-perf".)


2. Build llama.cpp
--------------------
This dashboard is a wrapper around llama.cpp's `llama-bench` binary —
you need llama.cpp built first, separately.

    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release -j

Confirm the binary exists:

    ls build/bin/llama-bench


3. Build the dashboard
------------------------
From this project's root:

    cd llama-gui
    mkdir build && cd build
    cmake ..
    cmake --build .

This produces an executable, typically `llama_dashboard`, inside
the `build/` directory.


4. Run it
----------
    ./llama_dashboard

The GUI should open in a window. Use the Configuration panel (left
sidebar) to set the path to your `llama-bench` binary and your model
files, then use the tabs on the right (Benchmark, Model Size Sweep,
Inference, Telemetry) to run tests.


5. (Optional) Permissions for perf stat
------------------------------------------
The Perf Stat tab uses hardware performance counters, which some
Linux systems restrict by default. If perf stat produces empty or
"permission denied" output, check:

    cat /proc/sys/kernel/perf_event_paranoid

If this value is 2 or higher, temporarily relax it (resets on
reboot):

    sudo sysctl kernel.perf_event_paranoid=-1

For a permanent change, add `kernel.perf_event_paranoid = -1` to
`/etc/sysctl.conf` and run `sudo sysctl -p`.


WINDOWS
=====================================================================

This project is developed and tested on native Linux. On Windows, the
easiest path is running it inside WSL2 (Windows Subsystem for Linux),
using WSLg for GUI display — no separate X server setup required on
a reasonably modern Windows install.

NOTE ON TELEMETRY: 
the Telemetry tab and per-run temperature/power
readings rely on Linux hardware sensor files (`/sys/class/hwmon`,
`/sys/devices/.../cpufreq`) that are not meaningfully accessible inside
a WSL2 virtual machine. Expect these values to show as 0 or
placeholder data under WSL2 — this is expected and does not affect
actual model inference or benchmark throughput numbers.


1. Install WSL2 + Ubuntu
---------------------------
Open PowerShell as Administrator:

    wsl --install -d Ubuntu

Restart if prompted. This installs WSL2, Ubuntu, and WSLg together on
Windows 11 (and recent Windows 10 builds). Once installed, launch
"Ubuntu" from the Start menu and finish the initial username/password
setup.

Confirm you're on WSL version 2:

    wsl --list --verbose

If it shows VERSION 1, upgrade with:

    wsl --set-version Ubuntu 2

And make sure WSLg components are current:

    wsl --update


2. (Optional but recommended) Increase WSL2 memory limit
-------------------------------------------------------------
By default, WSL2 caps itself at ~50% of your machine's total RAM,
which may not be enough for larger models (7B-14B+). To raise this,
create/edit a file at:

    C:\Users\<YourWindowsUsername>\.wslconfig

With contents like:

    [wsl2]
    memory=24GB
    processors=8

Adjust numbers to your actual hardware. Then restart WSL from
PowerShell:

    wsl --shutdown

And reopen the Ubuntu terminal.


3. Install dependencies (inside the Ubuntu/WSL terminal)
---------------------------------------------------------------
    sudo apt update
    sudo apt install build-essential cmake git \
        libglfw3-dev libgl1-mesa-dev libsqlite3-dev linux-tools-common \
        linux-tools-generic

(Perf counter support inside WSL2 can be inconsistent depending on
your Windows/WSL kernel version — if the Perf Stat tab doesn't
produce output, this is a known limitation of running perf inside a
VM rather than a project bug.)


4. Important: Hardcoded Linux tools (taskset, perf)
------------------------------------------------------
This project shells out to two Linux command-line tools directly:
 
  - `taskset` — used for CPU core affinity (Configuration panel's
    "CPU mask" field). This is part of Ubuntu's standard util-linux
    package, so it IS available inside WSL2 without extra install.
    However, WSL2 virtualizes the CPU topology, so core numbers may
    not correspond to the same physical P-core/E-core layout you'd
    see on bare-metal Linux. Any thread-affinity tuning done on native
    Linux may not carry the same meaning here.
 
  - `perf stat` (used by the Perf Stat tab) — requires direct access
    to hardware performance counters. This is NOT reliably available
    inside WSL2; it depends on whether your specific WSL2 kernel
    build exposes PMU passthrough from the Windows host, which is
    inconsistent across systems. If the Perf Stat tab produces empty
    output or errors on Windows/WSL2, this is a virtualization
    limitation, not a bug in the dashboard — the Sweep tab (model
    ladder, throughput table, plots) will still work normally
    regardless, since that only depends on llama-bench itself.
 
Neither of these tools exists on native Windows outside of WSL2 at
all — there is no direct Windows equivalent wired into this project.
If you need real CPU-affinity control or perf-counter data on
Windows, that would require separate, unimplemented platform-specific
code (e.g. Windows' SetThreadAffinityMask for affinity, and Windows
Performance Counters or ETW for perf-equivalent data) — not something
this project currently supports.


5. Build llama.cpp (inside WSL)
-----------------------------------
    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release -j

Confirm the binary exists:

    ls build/bin/llama-bench


5. Build the dashboard (inside WSL)
---------------------------------------
    cd llama-gui
    mkdir build && cd build
    cmake ..
    cmake --build .


6. Run it
-----------
    ./llama_dashboard

WSLg will automatically display the GUI window on your Windows
desktop, the same as any other Windows application, no manual
DISPLAY variable configuration or X server installation needed.


7. Getting model files onto WSL
-----------------------------------
Your WSL Ubuntu environment has its own filesystem, separate from
Windows. Either download models directly inside WSL (recommended,
see "Getting models" above — run the huggingface-cli command inside
the Ubuntu terminal), or if you already have GGUF files on the
Windows side, they're accessible from WSL at a path like:

    /mnt/c/Users/<YourWindowsUsername>/Downloads/model.gguf

Point the dashboard's Configuration panel at whichever path applies.


TROUBLESHOOTING
------------------
- "GLFW error... X11" or blank window: confirm WSLg is active
  (`wsl --update`, then restart WSL) and that you're on Windows 11 or
  an updated Windows 10 build with WSLg support.
- Segfault on startup: usually means a missing ImGui/ImPlot context
  initialization — check that CreateContext() calls exist for both
  ImGui and ImPlot before the main loop starts.
- Model sweep runs much slower than expected under WSL2: some
  virtualization overhead is normal; also confirm your .wslconfig
  memory/processor limits aren't artificially constraining the VM.

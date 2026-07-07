## Table of Contents

- [Sobel Edge Detection Benchmark Dashboard](#sobel-edge-detection-benchmark-dashboard)
- [Linux](#linux)
- [Windows](#windows)

# Sobel Edge Detection Benchmark Dashboard

A real-time benchmarking and visualization dashboard for comparing a custom AVX2+OpenMP Sobel edge detection implementation against OpenCV's Sobel functions. Built with Dear ImGui, ImPlot, OpenGL3, and GLFW.

---

## What it does

- Displays the original grayscale input image alongside three Sobel output images: your custom AVX2+OpenMP implementation, OpenCV integer (Gx+Gy), and OpenCV float magnitude
- Shows latency statistics (mean, std, min, max) and latency history graphs for each implementation
- Measures and displays memory bandwidth achieved versus theoretical peak
- Visualizes OpenMP thread partitioning, thread-to-core mapping, and workload distribution across P and E cores
- Detects your CPU's hybrid architecture (P-core vs E-core) automatically and weights row partitions accordingly
- Shows a SIMD analysis panel listing every AVX2 intrinsic used and its purpose
- Displays a roofline model visualization showing whether the implementation is compute-bound or memory-bandwidth-bound
- Compares image quality between implementations (MAE, PSNR, pixel difference percentage)
- Lets you change thread count and OMP_PROC_BIND at runtime from the GUI without restarting manually

---

## Linux

### Dependencies

Install the following:

```
sudo apt install build-essential git libglfw3-dev libgl1-mesa-dev libopencv-dev
```

If you want dmidecode for automatic memory bandwidth detection (optional, requires root):

```
sudo apt install dmidecode
```

### Clone the repository

```
git clone https://github.com/amiya-1070/chai-projects.git
cd chai-projects/sobel-gui
```

### Dear ImGui (docking branch) and ImPlot

These are **already included** in the repository (or you can clone them manually into the project directory):

```
git clone https://github.com/ocornut/imgui.git -b docking imgui
git clone https://github.com/epezent/implot.git implot
```

### stb headers

These are **already included** in the repository (or you can curl them manually into the project directory):

```
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

### Build

Step 1 — compile the C kernel:

```
gcc -c sobel_min.c -O3 -march=native -fopenmp -o sobel_min.o
```

Step 2 — compile and link the dashboard:

```
g++ main_dashboard.cpp sobel_dashboard.cpp \
    imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
    imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
    implot/implot.cpp implot/implot_items.cpp \
    sobel_min.o \
    $(pkg-config --cflags --libs glfw3 gl opencv4) \
    -I imgui -I imgui/backends -I implot \
    -fopenmp -O3 -march=native -std=c++17 \
    -o sobel_dashboard
```

If your OpenCV package config is named opencv rather than opencv4, replace opencv4 with opencv in the pkg-config line.

### Run

```
./sobel_dashboard /path/to/your/image.png
```

If no image path is given, the dashboard will look for a hardcoded default path. You can also change the image from inside the GUI using the path bar at the top of the window.

### Optional: set OMP environment variables before launch

The dashboard lets you change thread count from the GUI at runtime. For OMP_PROC_BIND (close vs spread vs master), changing it in the GUI will automatically re-exec the process with the correct environment variable set, so you do not need to set it manually. However if you want to launch with a specific binding from the start:

```
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=8 ./sobel_dashboard /path/to/image.png
```

### Notes for hybrid CPU users (Intel Alder Lake, Raptor Lake, etc.)

The dashboard automatically detects your P-core and E-core counts by reading per-core maximum frequencies from sysfs and computes a weighted row partition ratio so P-cores receive proportionally more work. This is displayed in the Build Info panel. 

If you already know your CPU's P-core to E-core performance ratio from your own
benchmarking (for example, you measured that P-cores are 1.6x faster than E-cores
on your specific workload), you can hardcode this instead of relying on the
automatic detection.

In sobel_min.c, find the lines near the top of sobel_avx2_omp that read:

    int e_w = 100;
    int p_w = (int)(e_w * pe_ratio + 0.5);

Replace them with your measured ratio directly, for example for a 1.6 ratio:

    int e_w = 100;
    int p_w = 160;

Then remove the pe_ratio parameter from the function signature in both
sobel_min.c and sobel_min.h, and remove the argument from every call site
in main_dashboard.cpp.

To find your ratio experimentally, run the dashboard with 1 thread and note
the mean latency, then manually pin execution to only P-cores using:

    taskset -c 0,1,2,3 ./sobel_dashboard /path/to/image.png

and then to only E-cores:

    taskset -c 4,5,6,7,8,9,10,11 ./sobel_dashboard /path/to/image.png

Divide the E-core latency by the P-core latency to get the ratio. Use that
value for p_w with e_w fixed at 100, rounding to the nearest integer.

---

## Windows

### Dependencies

You will need:

- Visual Studio 2022 (Community edition is fine) with the C++ Desktop Development workload installed
- CMake 3.20 or later: https://cmake.org/download
- Git for Windows: https://git-scm.com/download/win
- vcpkg for library management: https://github.com/microsoft/vcpkg

### Install vcpkg

Open a Developer Command Prompt and run:

```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
```

### Install dependencies via vcpkg

```
vcpkg install glfw3:x64-windows
vcpkg install opengl:x64-windows
vcpkg install opencv4:x64-windows
```

### Clone the repository

```
git clone https://github.com/amiya-1070/chai-projects.git
cd chai-projects/sobel-gui
```

### Clone Dear ImGui (docking branch) and ImPlot

These are **already included** in the repository (or you can clone them manually into the project directory):

```
git clone https://github.com/ocornut/imgui.git -b docking imgui
git clone https://github.com/epezent/implot.git implot
```

### Download stb headers

These are **already included** in the repository (or you can download them manually into the project directory):


```
https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

### OpenMP on Windows

OpenMP is supported by MSVC natively. In your Visual Studio project properties, go to Configuration Properties → C/C++ → Language and set OpenMP Support to Yes (/openmp). AVX2 support is enabled under Configuration Properties → C/C++ → Code Generation → Enable Enhanced Instruction Set → AVX2.

### Build with Visual Studio

Create a new Visual Studio project, add all the following source files:

```
main_dashboard.cpp
sobel_dashboard.cpp
sobel_min.c
imgui/imgui.cpp
imgui/imgui_draw.cpp
imgui/imgui_tables.cpp
imgui/imgui_widgets.cpp
imgui/backends/imgui_impl_glfw.cpp
imgui/backends/imgui_impl_opengl3.cpp
implot/implot.cpp
implot/implot_items.cpp
```

Add the following include directories in project properties:

```
imgui
imgui/backends
implot
path/to/vcpkg/installed/x64-windows/include
```

Add the following library directories:

```
path/to/vcpkg/installed/x64-windows/lib
```

Link against:

```
glfw3.lib
opengl32.lib
opencv_world4xx.lib      (replace xx with your installed version number)
```

Set the C++ standard to C++17 under Configuration Properties → C/C++ → Language → C++ Language Standard.

Build the solution in Release x64 configuration.

### Run

```
sobel_dashboard.exe C:\path\to\your\image.png
```

You can also change the image from inside the GUI using the path bar at the top of the window.

### Notes on OMP_PROC_BIND on Windows

On Windows, setting OMP_PROC_BIND via the GUI triggers a process relaunch using CreateProcess with the updated environment block. This behaves the same as on Linux. If you want to launch with a specific binding from the start, set the environment variables before running:

```
set OMP_PROC_BIND=close
set OMP_PLACES=cores
set OMP_NUM_THREADS=8
sobel_dashboard.exe C:\path\to\image.png
```

### Notes for hybrid CPU users on Windows

The P-core and E-core detection on Windows falls back to uniform partitioning since per-core frequency sysfs files are not available. The dashboard will still run correctly but will not apply weighted row partitioning. All other panels function identically to the Linux version.

If you already know your CPU's P-core to E-core performance ratio from your own
benchmarking (for example, you measured that P-cores are 1.6x faster than E-cores
on your specific workload), you can hardcode this instead of relying on the
automatic detection.

Open Command Prompt and use the /affinity flag with a hex bitmask.
Each bit in the mask corresponds to a logical CPU, with bit 0 being CPU 0.
For example if your P-cores are CPUs 0-3 and E-cores are CPUs 4-11:

    P-cores only (CPUs 0-3, bitmask 00001111 in binary = 0xF):
    start /affinity 0xF sobel_dashboard.exe C:\path\to\image.png

    E-cores only (CPUs 4-11, bitmask 111111110000 in binary = 0xFF0):
    start /affinity 0xFF0 sobel_dashboard.exe C:\path\to\image.png

To find which CPUs are your P-cores and which are E-cores on Windows, open
Task Manager, go to Performance, click CPU, then right-click the graph and
select Change graph to, then Logical processors. P-cores will show higher
maximum frequencies. Alternatively run this in PowerShell:

    Get-WmiObject Win32_Processor | Select-Object Name, MaxClockSpeed, NumberOfCores, NumberOfLogicalProcessors

The core numbering in Windows does not always match Linux, so verify your
CPU topology using the Coreinfo tool from Microsoft Sysinternals:

    https://learn.microsoft.com/en-us/sysinternals/downloads/coreinfo

---

## Troubleshooting

OpenCV not found by pkg-config: run pkg-config --list-all | grep opencv to see what name your installation uses and substitute it in the build command.

ImGui docking features missing: make sure you cloned the docking branch specifically with -b docking and not the master branch.

Black image panels: your image path is incorrect or the file format is unsupported. The dashboard uses stb_image which supports PNG, JPG, BMP, and TGA.

Thread panels show zero rows: this means sobel_avx2_omp has not run yet. This should not happen on a normal launch but can occur if the image failed to load, in which case check the terminal for a "Could not load image" error message.

Benchmark freezes the window while running: this is expected. The benchmark runs 50 iterations per implementation on an 8192x8192 image which takes several seconds. The window will unfreeze when results are ready.
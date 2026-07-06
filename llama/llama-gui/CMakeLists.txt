cmake_minimum_required(VERSION 3.16)
project(llama_dashboard)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Paths
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/third_party/implot)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(SQLite3 REQUIRED)

# ImGui sources
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ImPlot sources
set(IMPLOT_SOURCES
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
)

# Dashboard sources
set(SOURCES
    src/main.cpp
    src/dashboard.cpp
    src/config_panel.cpp
    src/benchmark_panel.cpp
    src/telemetry.cpp
    src/inference_panel.cpp
    src/storage.cpp
    src/model_panel_sweep.cpp
    src/process_utils.cpp
    third_party/sqlite3.c
)

add_executable(llama_dashboard
    ${SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

target_include_directories(llama_dashboard PRIVATE
    src/
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    third_party/
    ${SQLite3_INCLUDE_DIRS}
)

target_link_libraries(llama_dashboard
    OpenGL::GL
    glfw
    ${SQLite3_LIBRARIES}
    pthread
    dl
)

target_compile_options(llama_dashboard PRIVATE
    -O2
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)
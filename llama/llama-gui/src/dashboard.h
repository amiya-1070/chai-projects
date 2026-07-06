//dashboard.h
#pragma once
#include "config_panel.h"
#include "benchmark_panel.h"
#include "inference_panel.h"
#include "telemetry.h"
#include "storage.h"
#include "model_panel_sweep.h"

class Dashboard {
public:
    Dashboard();
    ~Dashboard();

    // Call once after OpenGL/ImGui init
    bool init();

    // Call every frame inside the render loop
    void render();

    // Call on shutdown
    void shutdown();

private:
    void render_menu_bar();
    void render_telemetry_bar();
    void render_main_tabs();
    void render_config_window();
    void render_telemetry_window();

    ConfigPanel         m_config;
    BenchmarkPanel      m_bench;
    InferencePanel      m_infer;
    TelemetryCollector  m_telemetry;
    Storage             m_storage;
    ModelSweepPanel m_model_sweep;

    // Window visibility flags
    bool m_show_config    = true;
    bool m_show_telemetry = true;

    // Layout
    float m_config_width  = 380.0f;

    // Status bar message
    char  m_status[256]   = "Ready.";
};
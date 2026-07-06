#pragma once
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>
#include "config_panel.h"
#include "storage.h"
#include "telemetry.h"
#include "process_utils.h"
#include "benchmark_panel.h"  // reuse BenchResult

// One model to test in the size sweep
struct ModelSpec {
    std::string label;      // "1B", "3B", "7B", "14B"
    std::string path;       // full path to .gguf
    float       size_b;     // 1.0, 3.0, 7.0, 14.0
    std::string quant;       // "Q4_K_M", "Q8_0", "F16"
};

struct ModelSweepRun {
    ModelSpec   spec;
    BenchResult pp;
    BenchResult tg;
    float       avg_temp_c   = 0.0f;
    float       avg_power_w  = 0.0f;
    float       peak_rss_mb  = 0.0f;   // now system-wide peak used RAM
    float       peak_swap_mb = 0.0f;   // NEW
    bool        oom          = false;
    bool        complete     = false;
};

// Per-model raw perf stat output
struct PerfCaptureResult {
    std::string label;       // "3B", "7B", "14B"
    std::string raw_output;  // full perf stat text
};



class ModelSweepPanel {
public:
    ModelSweepPanel();
    ~ModelSweepPanel();

    void set_storage(Storage* s)              { m_storage = s; }
    void set_telemetry(TelemetryCollector* t) { m_telemetry = t; }

    void render(const DashboardConfig& cfg);

    // Editable list of models to sweep (populated from UI or a models dir scan)
    std::vector<ModelSpec>& models() { return m_models; }

private:
    void start_sweep(const DashboardConfig& cfg);
    void sweep_thread_func(DashboardConfig cfg);

    void render_controls(const DashboardConfig& cfg);
    void render_model_list();
    void render_results_table();
    void render_scaling_plot();
    
    void render_perf_tab();
    std::vector<PerfCaptureResult> m_perf_results;

    std::thread       m_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_cancel{false};
    mutable std::mutex m_mutex;

    std::vector<ModelSpec>    m_models;   // configured ladder
    std::vector<ModelSweepRun> m_results;
    int  m_progress = 0;

    std::deque<std::string> m_log;
    static constexpr int MAX_LOG = 200;

    std::vector<float> m_temp_samples;
    std::vector<float> m_power_samples;

    Storage*            m_storage   = nullptr;
    TelemetryCollector* m_telemetry = nullptr;

    char m_new_model_path[512] = "";
    char m_new_model_label[32] = "";
    float m_new_model_size = 7.0f;
    char m_new_model_quant[16] = "Q4_K_M";
};
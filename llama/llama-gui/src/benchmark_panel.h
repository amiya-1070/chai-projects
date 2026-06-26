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

// Parsed result from one llama-bench output line
struct BenchResult {
    std::string test;       // "pp512" or "tg200" etc
    float       tps  = 0.0f;
    float       std  = 0.0f;
    bool        valid = false;
};

// One complete benchmark run (pp + tg pair)
struct BenchRun {
    BenchParams  params;
    BenchResult  pp;
    BenchResult  tg;
    float        avg_temp_c  = 0.0f;
    float        avg_power_w = 0.0f;
    bool         complete    = false;
};

class BenchmarkPanel {
public:
    BenchmarkPanel();
    ~BenchmarkPanel();

    void set_storage(Storage* s)   { m_storage = s; }
    void set_telemetry(TelemetryCollector* t) { m_telemetry = t; }

    // Render the full benchmark panel
    void render(const DashboardConfig& cfg);

private:
    // Run management
    void start_run(const DashboardConfig& cfg);
    void stop_run();
    void bench_thread_func(BenchParams params);

    // Output parsing
    BenchResult parse_bench_line(const std::string& line);
    bool        is_bench_line(const std::string& line);

    // Sub-panel renderers
    void render_controls(const DashboardConfig& cfg);
    void render_live_output();
    void render_current_result();
    void render_history();
    void render_comparison_plot();
    void render_thread_sweep(const DashboardConfig& cfg);

    // Thread sweep
    void start_sweep(const DashboardConfig& cfg);
    void sweep_thread_func(DashboardConfig cfg);

    // State
    std::thread         m_thread;
    std::atomic<bool>   m_running{false};
    std::atomic<bool>   m_cancel{false};

    std::thread         m_sweep_thread;
    std::atomic<bool>   m_sweep_running{false};
    std::atomic<bool>   m_sweep_cancel{false};

    mutable std::mutex  m_mutex;

    // Live output ring buffer
    std::deque<std::string> m_log;
    static constexpr int    MAX_LOG = 200;

    // Current and completed runs
    BenchRun                m_current;
    std::vector<BenchRun>   m_completed;

    // Thread sweep results: thread_count -> BenchRun
    std::vector<std::pair<int, BenchRun>> m_sweep_results;
    int                     m_sweep_progress = 0;
    std::vector<int>        m_sweep_thread_counts = {1,2,4,6,8,10};

    // Telemetry accumulators during run
    std::vector<float>      m_temp_samples;
    std::vector<float>      m_power_samples;

    Storage*                m_storage  = nullptr;
    TelemetryCollector*     m_telemetry = nullptr;

    // History loaded from DB
    std::vector<BenchRecord> m_history;
    bool                     m_history_dirty = true;

    // UI state
    int                     m_selected_history = -1;
    bool                    m_show_sweep       = false;
    char                    m_notes_buf[256]   = "";
};
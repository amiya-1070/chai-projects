//sobel_dashboard.h
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <array>



struct DashboardData;

struct GpuTex {
    GLuint id = 0;
    int width = 0;
    int height = 0;
};


struct TexCache {
    GpuTex input;
    std::vector<GpuTex> impls;
    void Refresh(const DashboardData& d);  // defined in sobel_dashboard.cpp
};

void SobelDashboard_Draw(DashboardData& d, TexCache& tc, bool refresh_textures);

// ── Per-implementation latency stats ─────────────────────────────────────────
struct LatencyStats {
    double mean_ms   = 0.0;
    double stddev_ms = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    std::vector<double> history_ms;   // one entry per benchmark run
};

// ── Per-thread info (filled by OpenMP instrumentation) ───────────────────────
struct ThreadInfo {
    int thread_id   = 0;
    int cpu_core    = -1;   // sched_getcpu() snapshot
    int row_start   = 0;
    int row_end     = 0;    // exclusive
    double work_pct = 0.0;  // (row_end-row_start) / total_rows * 100
};

// ── Memory-bandwidth measurement ─────────────────────────────────────────────
struct BandwidthStats {
    double achieved_gb_s   = 0.0;
    double theoretical_gb_s = 0.0;   // set from cpuid / manual entry
    double pct_of_peak     = 0.0;
};

// ── Image-quality comparison ──────────────────────────────────────────────────
struct QualityMetrics {
    double max_error      = 0.0;
    double mae            = 0.0;   // mean absolute error
    double psnr_db        = 0.0;
    double differing_pct  = 0.0;   // % pixels that differ at all
};

// ── Edge statistics ───────────────────────────────────────────────────────────
struct EdgeStats {
    double  avg_intensity  = 0.0;
    double  saturation_pct = 0.0;   // % pixels == 255
    double  edge_density   = 0.0;   // % pixels > some threshold (e.g. 32)
    std::array<int,256> histogram{};
};

// ── Build / runtime info (fill at startup) ────────────────────────────────────
struct BuildInfo {
    std::string compiler_version;   // e.g. "GCC 13.2"
    std::string cflags;             // e.g. "-O3 -march=native -fopenmp"
    std::string openmp_runtime;     // e.g. "libgomp 13"
    bool        avx2_available = false;
    std::string cpu_model;
    std::string l1_cache, l2_cache, l3_cache;
    std::string opencv_version;
    bool        opencv_ipp = false, opencv_tbb = false, opencv_openmp = false;
    std::string opencv_threading_backend;
    std::string omp_num_threads, omp_proc_bind, omp_places;
    int         logical_cpu_count = 0;
};

// ── One implementation's complete result ─────────────────────────────────────
struct ImplResult {
    std::string name;                   // "AVX2+OMP", "OpenCV Gx+Gy", "OpenCV float mag"
    std::vector<uint8_t> output_gray;   // flattened H×W output pixels
    int width = 0, height = 0;

    LatencyStats   latency;
    BandwidthStats bandwidth;
    int            thread_count = 1;

    // formula used
    std::string magnitude_formula;      // "|Gx|+|Gy|" or "sqrt(Gx²+Gy²)"

    // quality vs reference (OpenCV float mag is usually the reference)
    QualityMetrics quality;
    EdgeStats      edges;
};

// ── Top-level container passed to the dashboard ──────────────────────────────
struct DashboardData {
    // Source image
    std::vector<uint8_t> input_gray;
    int width = 0, height = 0;

    // Results — order: [0]=AVX2+OMP, [1]=OpenCV Gx+Gy, [2]=OpenCV float mag
    std::vector<ImplResult> impls;

    // Thread partition (for AVX2+OMP impl)
    std::vector<ThreadInfo> threads;

    BuildInfo build;

    // Roofline
    double arithmetic_intensity = 0.0;  // FLOP/byte — compute manually or estimate
};
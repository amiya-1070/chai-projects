// main_dashboard.cpp
// Compile: g++ main_dashboard.cpp sobel_dashboard.cpp bench_sobel_opencv.cpp \
//            $(pkg-config --cflags --libs glfw3 gl opencv4) \
//            -I imgui -I imgui/backends -I implot                             \
//            imgui/*.cpp imgui/backends/imgui_impl_glfw.cpp                   \
//            imgui/backends/imgui_impl_opengl3.cpp implot/implot*.cpp         \
//            -fopenmp -O3 -march=native -std=c++17

// You'll need to compile sobel_min.c separately and link it, or include it:
//   gcc -c sobel_min.c -O3 -march=native -fopenmp -o sobel_min.o
//   then add sobel_min.o to the link step

#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS
#elif defined(__APPLE__)
    #define OS_MACOS
#elif defined(__linux__)
    #define OS_LINUX
#endif

#if defined(OS_WINDOWS)
    #include <windows.h>
    #include <intrin.h>
#else
    #include <unistd.h>
    #include <sched.h>
    #include <cpuid.h>
#endif

#include "sobel_dashboard.h"
#include <cpuid.h>
#include "sobel_min.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <numeric>
#include <cmath>

#include <omp.h>
#include <sched.h>       // sched_getcpu
#include <unistd.h>      // sysconf

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

void SobelDashboard_Draw(DashboardData& d, TexCache& tc, bool refresh_textures);

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper
// ─────────────────────────────────────────────────────────────────────────────
static double NowMs()
{
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Bandwidth estimator  (bytes read + written / elapsed_s)
// For Sobel: each pixel reads ~9 neighbours (with overlap, ≈3 rows × width bytes
//            net unique per output row ≈ 3 × width src + 1 × width dst)
// ─────────────────────────────────────────────────────────────────────────────
static double EstimateBandwidthGBs(int W, int H, double elapsed_ms)
{
    // conservative: 3 source rows read per output row + 1 dst row written
    // ≈ (3 + 1) × W × H bytes total
    double bytes = 4.0 * W * H;
    double secs  = elapsed_ms * 1e-3;
    return bytes / secs / 1e9;
}

// ─────────────────────────────────────────────────────────────────────────────
// Read environment variable safely
// ─────────────────────────────────────────────────────────────────────────────
static std::string GetEnv(const char* name)
{
    const char* v = getenv(name);
    return v ? v : "(unset)";
}

// ─────────────────────────────────────────────────────────────────────────────
// Quality metrics
// ─────────────────────────────────────────────────────────────────────────────
static QualityMetrics ComputeQuality(const std::vector<uint8_t>& a,
                                     const std::vector<uint8_t>& ref,
                                     int N)
{
    QualityMetrics q;
    if (a.size() != (size_t)N || ref.size() != (size_t)N) return q;
    double sum_sq = 0, sum_abs = 0;
    int diff_count = 0;
    for (int i = 0; i < N; ++i) {
        double d = (double)a[i] - (double)ref[i];
        if (d < 0) d = -d;
        sum_abs  += d;
        sum_sq   += d * d;
        if ((int)d > 0) ++diff_count;
        if (d > q.max_error) q.max_error = d;
    }
    q.mae           = sum_abs / N;
    q.differing_pct = 100.0 * diff_count / N;
    double mse      = sum_sq / N;
    q.psnr_db       = mse > 0 ? 10.0 * log10(255.0*255.0 / mse) : 1e18;
    return q;
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge stats
// ─────────────────────────────────────────────────────────────────────────────
static EdgeStats ComputeEdgeStats(const std::vector<uint8_t>& img, int N)
{
    EdgeStats e;
    e.histogram.fill(0);
    double sum = 0;
    int sat = 0, edge = 0;
    for (int i = 0; i < N; ++i) {
        int v = img[i];
        e.histogram[v]++;
        sum += v;
        if (v == 255)  ++sat;
        if (v > 32)    ++edge;
    }
    e.avg_intensity  = sum / N;
    e.saturation_pct = 100.0 * sat  / N;
    e.edge_density   = 100.0 * edge / N;
    return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenCV implementations
// ─────────────────────────────────────────────────────────────────────────────
static void RunOpenCVGxGy(const cv::Mat& gray, cv::Mat& out)
{
    cv::Mat gx, gy, agx, agy;
    cv::Sobel(gray, gx, CV_16S, 1, 0, 3);
    cv::Sobel(gray, gy, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(gx, agx);
    cv::convertScaleAbs(gy, agy);
    cv::addWeighted(agx, 0.5, agy, 0.5, 0, out);
}

static void RunOpenCVFloatMag(const cv::Mat& gray, cv::Mat& out)
{
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    mag.convertTo(out, CV_8U, 1.0, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bench harness — run fn N times, collect latency history
// ─────────────────────────────────────────────────────────────────────────────
template<typename Fn>
static LatencyStats BenchRuns(Fn&& fn, int runs = 50)
{
    LatencyStats s;
    s.history_ms.reserve(runs);
    s.min_ms = 1e18;
    double sum = 0, sum2 = 0;
    for (int r = 0; r < runs; ++r) {
        double t0 = NowMs();
        fn();
        double dt = NowMs() - t0;
        s.history_ms.push_back(dt);
        sum  += dt;
        sum2 += dt * dt;
        s.min_ms = std::min(s.min_ms, dt);
        s.max_ms = std::max(s.max_ms, dt);
    }
    s.mean_ms   = sum / runs;
    double var  = sum2/runs - s.mean_ms*s.mean_ms;
    s.stddev_ms = var > 0 ? sqrt(var) : 0.0;
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Collect thread partition info via OpenMP
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<ThreadInfo> CollectThreadInfo(int H)
{
    // sobel_avx2_omp must have already run before this is called
    const CThreadInfo* raw = get_thread_info();
    int NT = get_thread_count();

    std::vector<ThreadInfo> info(NT);
    for (int t = 0; t < NT; t++) {
        info[t].thread_id = raw[t].tid;
        info[t].cpu_core  = raw[t].cpu;
        info[t].row_start = raw[t].row_start;   // from sobel_avx2_omp's weighted partition
        info[t].row_end   = raw[t].row_end;
        info[t].work_pct  = 100.0 * (raw[t].row_end - raw[t].row_start) / (double)H;
    }
    return info;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill BuildInfo at startup
// ─────────────────────────────────────────────────────────────────────────────

static double ReadPeakBandwidthGBs() {
#if defined(OS_LINUX)
    FILE* f = popen("dmidecode -t memory 2>/dev/null | grep -i 'Speed:' | grep -v 'Unknown' | head -1", "r");
    if (f) {
        int speed_mhz = 0;
        char line[128];
        if (fgets(line, sizeof(line), f))
            sscanf(line, " Speed: %d MT/s", &speed_mhz);
        pclose(f);
        if (speed_mhz > 0)
            return (speed_mhz * 8.0 * 2.0) / 1000.0;
    }
    return 40.0;
#elif defined(OS_MACOS)
    // sysctl on macOS
    FILE* f = popen("system_profiler SPMemoryDataType 2>/dev/null | grep 'Speed:' | head -1", "r");
    if (f) {
        int speed_mhz = 0;
        char line[128];
        if (fgets(line, sizeof(line), f))
            sscanf(line, " Speed: %d MHz", &speed_mhz);
        pclose(f);
        if (speed_mhz > 0)
            return (speed_mhz * 8.0 * 2.0) / 1000.0;
    }
    return 40.0;
#elif defined(OS_WINDOWS)
    // WMI via powershell
    FILE* f = _popen("powershell -command \"Get-WmiObject Win32_PhysicalMemory | Select-Object -First 1 Speed | ForEach-Object { $_.Speed }\"", "r");
    if (f) {
        int speed_mhz = 0;
        fscanf(f, "%d", &speed_mhz);
        _pclose(f);
        if (speed_mhz > 0)
            return (speed_mhz * 8.0 * 2.0) / 1000.0;
    }
    return 40.0;
#else
    return 40.0;
#endif
}

    

static double ReadPeakComputeGFlops() {
    double max_freq_ghz = 0.0;

#if defined(OS_LINUX)
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (f) {
        long freq_khz = 0;
        fscanf(f, "%ld", &freq_khz);
        fclose(f);
        max_freq_ghz = freq_khz / 1e6;
    }
#elif defined(OS_MACOS)
    FILE* f = popen("sysctl -n hw.cpufrequency_max 2>/dev/null", "r");
    if (f) {
        long freq_hz = 0;
        fscanf(f, "%ld", &freq_hz);
        pclose(f);
        max_freq_ghz = freq_hz / 1e9;
    }
#elif defined(OS_WINDOWS)
    FILE* f = _popen("powershell -command \"(Get-WmiObject Win32_Processor).MaxClockSpeed\"", "r");
    if (f) {
        int freq_mhz = 0;
        fscanf(f, "%d", &freq_mhz);
        _pclose(f);
        max_freq_ghz = freq_mhz / 1e3;
    }
#endif

    if (max_freq_ghz < 0.5) max_freq_ghz = 3.5;

    int logical_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);  // POSIX
#if defined(OS_WINDOWS)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    logical_cpus = (int)si.dwNumberOfProcessors;
#endif

    unsigned int eax, ebx, ecx, edx;
    bool has_avx2 = false;
#if !defined(OS_WINDOWS)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        has_avx2 = (ebx >> 5) & 1;
#else
    int cpuinfo[4];
    __cpuidex(cpuinfo, 7, 0);
    has_avx2 = (cpuinfo[1] >> 5) & 1;
#endif

    double ops_per_cycle = has_avx2 ? 32.0 : 8.0;
    return logical_cpus * ops_per_cycle * max_freq_ghz;
}


static double ComputePEWeightRatio(int* p_count_out, int* e_count_out) {
    int num_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
#if defined(OS_WINDOWS)
    SYSTEM_INFO si; GetSystemInfo(&si);
    num_cpus = (int)si.dwNumberOfProcessors;
#endif

    *p_count_out = num_cpus;
    *e_count_out = 0;

#if defined(OS_LINUX)
    std::vector<long> max_freqs(num_cpus, 0);
    long overall_max = 0;

    for (int i = 0; i < num_cpus; i++) {
        char path[128];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) { return 1.0; }
        fscanf(f, "%ld", &max_freqs[i]);
        fclose(f);
        if (max_freqs[i] > overall_max) overall_max = max_freqs[i];
    }

    long p_thresh = (long)(overall_max * 0.95);
    long p_freq_sum = 0, e_freq_sum = 0;
    int  p_count = 0, e_count = 0;

    for (int i = 0; i < num_cpus; i++) {
        if (max_freqs[i] >= p_thresh) { p_freq_sum += max_freqs[i]; p_count++; }
        else                          { e_freq_sum += max_freqs[i]; e_count++; }
    }

    *p_count_out = p_count;
    *e_count_out = e_count;

    if (e_count == 0 || p_count == 0) return 1.0;

    double avg_p = (double)p_freq_sum / p_count;
    double avg_e = (double)e_freq_sum / e_count;
    const double ipc_factor = 1.2;
    return (avg_p / avg_e) * ipc_factor;

#elif defined(OS_MACOS)
    // macOS doesn't expose per-core freq easily; check for hybrid via sysctl
    FILE* f = popen("sysctl -n hw.nperflevels 2>/dev/null", "r");
    if (f) {
        int nlevels = 0;
        fscanf(f, "%d", &nlevels);
        pclose(f);
        if (nlevels >= 2) {
            // Apple Silicon or hybrid — read p/e counts from sysctl
            int pcount = 0, ecount = 0;
            FILE* fp = popen("sysctl -n hw.perflevel0.logicalcpu 2>/dev/null", "r");
            if (fp) { fscanf(fp, "%d", &pcount); pclose(fp); }
            FILE* fe = popen("sysctl -n hw.perflevel1.logicalcpu 2>/dev/null", "r");
            if (fe) { fscanf(fe, "%d", &ecount); pclose(fe); }
            *p_count_out = pcount;
            *e_count_out = ecount;
            // Apple M-series: P/E ratio is roughly 3.2GHz/2.1GHz ≈ 1.5
            return (pcount > 0 && ecount > 0) ? 1.5 : 1.0;
        }
    }
    return 1.0;  // homogeneous Intel Mac

#else
    // Windows or unknown: no reliable per-core freq without WMI complexity
    // assume homogeneous
    return 1.0;
#endif
}


static BuildInfo MakeBuildInfo()
{
    BuildInfo b;

    #if defined(__GNUC__)
        char buf[128];
        snprintf(buf, sizeof(buf), "GCC %d.%d.%d",
                __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
        b.compiler_version = buf;
    #elif defined(__clang__)
        b.compiler_version = "Clang " __clang_version__;
    #else
        b.compiler_version = "Unknown";
    #endif

        b.cflags = "-O3 -march=native -fopenmp";  // edit to match your actual flags

    #if defined(__AVX2__)
        b.avx2_available = true;
    #endif

    b.logical_cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    b.omp_num_threads   = GetEnv("OMP_NUM_THREADS");
    b.omp_proc_bind     = GetEnv("OMP_PROC_BIND");
    b.omp_places        = GetEnv("OMP_PLACES");

    // Read CPU model from /proc/cpuinfo
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    b.cpu_model = colon + 2;
                    if (!b.cpu_model.empty() && b.cpu_model.back() == '\n')
                        b.cpu_model.pop_back();
                    break;
                }
            }
        }
        fclose(f);
    }

    // Cache sizes from sysfs (Linux)
    auto ReadSysfs = [](const char* path) -> std::string {
        FILE* ff = fopen(path, "r");
        if (!ff) return "n/a";
        char buf2[64];
        if (fgets(buf2, sizeof(buf2), ff) == nullptr)
            buf2[0] = '\0';
        std::string s = buf2;
        if (!s.empty() && s.back() == '\n') s.pop_back();
        return s;
    };
    b.l1_cache = ReadSysfs("/sys/devices/system/cpu/cpu0/cache/index0/size");
    b.l2_cache = ReadSysfs("/sys/devices/system/cpu/cpu0/cache/index2/size");
    b.l3_cache = ReadSysfs("/sys/devices/system/cpu/cpu0/cache/index3/size");

    b.opencv_version          = cv::getVersionString();
    b.opencv_threading_backend = cv::getBuildInformation().find("TBB") != std::string::npos ? "TBB"
                           : cv::getBuildInformation().find("OpenMP") != std::string::npos ? "OpenMP"
                           : "pthreads";
    // Check build config strings for IPP/TBB/OMP
    std::string bi = cv::getBuildInformation();
    b.opencv_ipp    = bi.find("IPP:")    != std::string::npos &&
                      bi.find("NO")      == std::string::npos;
    b.opencv_tbb    = bi.find("TBB:")    != std::string::npos;
    b.opencv_openmp = bi.find("OpenMP:") != std::string::npos;

    b.openmp_runtime = "libgomp";  // adjust if using LLVM OMP

    // Read peak memory bandwidth from dmidecode
    // Falls back to a conservative estimate if not available
    b.peak_bandwidth_gbs  = ReadPeakBandwidthGBs();
    b.peak_compute_gflops = ReadPeakComputeGFlops();
    
    b.pe_weight_ratio = ComputePEWeightRatio(&b.p_core_count, &b.e_core_count);
    
    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// OMP settings
// ─────────────────────────────────────────────────────────────────────────────
static void ApplyOmpSettings(const OmpSettings& s)
{
    omp_set_num_threads(s.num_threads);
    omp_set_dynamic(s.dynamic ? 1 : 0);

    static const char* bind_strs[] = {"close", "spread", "master"};
    setenv("OMP_PROC_BIND", bind_strs[s.proc_bind], 1);
    // OMP_PROC_BIND is read-only at runtime via the API,
    // so setenv + respawning threads is the only portable way
}

// ─────────────────────────────────────────────────────────────────────────────
// Run all benchmarks and populate DashboardData
// ─────────────────────────────────────────────────────────────────────────────
static DashboardData RunBenchmarks(const char* image_path, const OmpSettings& settings)
{
    DashboardData d;
    d.build = MakeBuildInfo();
    d.omp_settings = settings;   // copy in BEFORE ApplyOmpSettings
    ApplyOmpSettings(d.omp_settings);
    
    // Load image as grayscale
    cv::Mat src_bgr = cv::imread(image_path);
    if (src_bgr.empty()) {
        fprintf(stderr, "Could not load image: %s\n", image_path);
        // Return empty data; dashboard will show "(no data)"
        return d;
    }
    cv::Mat gray;
    if (src_bgr.channels() == 3)
        cv::cvtColor(src_bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src_bgr;

    d.width  = gray.cols;
    d.height = gray.rows;
    int N    = d.width * d.height;
    d.input_gray.assign(gray.data, gray.data + N);

    // ── AVX2 + OMP ───────────────────────────────────────────────────────────
    
    {
        ImplResult im;
        im.name             = "AVX2+OMP";
        im.width            = d.width;
        im.height           = d.height;
        im.magnitude_formula= "|Gx|+|Gy|";
        im.thread_count     = d.omp_settings.num_threads;
        im.output_gray.resize(N);

        const uint8_t* src_ptr = gray.data;
        uint8_t*       dst_ptr = im.output_gray.data();
        int W = d.width, H = d.height;

        // warmup + force thread_info[] to reflect current settings
        sobel_avx2_omp(src_ptr, dst_ptr, H, W, d.omp_settings.num_threads, d.build.pe_weight_ratio);
        sobel_avx2_omp(src_ptr, dst_ptr, H, W, d.omp_settings.num_threads, d.build.pe_weight_ratio);
        sobel_avx2_omp(src_ptr, dst_ptr, H, W, d.omp_settings.num_threads, d.build.pe_weight_ratio);

        im.latency = BenchRuns([&]{
            sobel_avx2_omp(src_ptr, dst_ptr, H, W, d.omp_settings.num_threads, d.build.pe_weight_ratio);
        });

        im.bandwidth.theoretical_gb_s = 51.2;  // i7-1255U LPDDR4x-3200 dual-ch
        im.bandwidth.achieved_gb_s    = EstimateBandwidthGBs(W, H, im.latency.mean_ms);
        im.bandwidth.pct_of_peak      = 100.0 * im.bandwidth.achieved_gb_s
                                              / im.bandwidth.theoretical_gb_s;
        im.edges = ComputeEdgeStats(im.output_gray, N);
        d.impls.push_back(std::move(im));
    }

    // ── OpenCV Gx+Gy ─────────────────────────────────────────────────────────
    {
        ImplResult im;
        im.name             = "OpenCV Gx+Gy";
        im.width            = d.width;
        im.height           = d.height;
        im.magnitude_formula= "0.5|Gx|+0.5|Gy|";
        im.thread_count     = 1;  // OpenCV Sobel is single-threaded by default
        im.output_gray.resize(N);

        cv::Mat out_mat(d.height, d.width, CV_8U, im.output_gray.data());
        im.latency = BenchRuns([&]{ RunOpenCVGxGy(gray, out_mat); });

        im.bandwidth.theoretical_gb_s = 51.2;
        im.bandwidth.achieved_gb_s    = EstimateBandwidthGBs(d.width, d.height,
                                                              im.latency.mean_ms);
        im.bandwidth.pct_of_peak      = 100.0 * im.bandwidth.achieved_gb_s
                                              / im.bandwidth.theoretical_gb_s;
        im.edges = ComputeEdgeStats(im.output_gray, N);
        d.impls.push_back(std::move(im));
    }

    // ── OpenCV float magnitude ────────────────────────────────────────────────
    {
        ImplResult im;
        im.name             = "OpenCV float mag";
        im.width            = d.width;
        im.height           = d.height;
        im.magnitude_formula= "sqrt(Gx²+Gy²)";
        im.thread_count     = 1;
        im.output_gray.resize(N);

        cv::Mat out_mat(d.height, d.width, CV_8U, im.output_gray.data());
        im.latency = BenchRuns([&]{ RunOpenCVFloatMag(gray, out_mat); });

        im.bandwidth.theoretical_gb_s = 51.2;
        im.bandwidth.achieved_gb_s    = EstimateBandwidthGBs(d.width, d.height,
                                                              im.latency.mean_ms);
        im.bandwidth.pct_of_peak      = 100.0 * im.bandwidth.achieved_gb_s
                                              / im.bandwidth.theoretical_gb_s;
        im.edges = ComputeEdgeStats(im.output_gray, N);
        d.impls.push_back(std::move(im));
    }

    // ── Quality: compare everything against float-mag reference ──────────────
    const auto& ref = d.impls[2].output_gray;
    for (auto& im : d.impls)
        im.quality = ComputeQuality(im.output_gray, ref, N);

    // ── Thread info ───────────────────────────────────────────────────────────
    d.threads = CollectThreadInfo(d.height);

    // ── Arithmetic intensity estimate ─────────────────────────────────────────
    // For L1 Sobel: ~18 adds/subs + 2 abs + 1 add + 1 pack = ~22 ops
    // Bytes: 9 src reads + 1 dst write = 10 bytes/pixel (with cache reuse ≈ 4)
    d.arithmetic_intensity = 22.0 / 4.0;  // ≈ 5.5 FLOP/byte

    // at the end of RunBenchmarks, before return d:
    d.build.omp_num_threads = std::to_string(d.omp_settings.num_threads);
    d.build.omp_proc_bind   = (d.omp_settings.proc_bind == 0) ? "close"
                            : (d.omp_settings.proc_bind == 1) ? "spread" : "master";

    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLFW error callback
// ─────────────────────────────────────────────────────────────────────────────
static void GlfwError(int, const char* msg)
{
    fprintf(stderr, "GLFW error: %s\n", msg);
}


static void RelaunchWithSettings(const char* image_path, const OmpSettings& s)
{
    static const char* bind_strs[] = {"close", "spread", "master"};

    // build env var strings
    char threads_str[32];
    snprintf(threads_str, sizeof(threads_str), "%d", s.num_threads);

    setenv("OMP_NUM_THREADS", threads_str,            1);
    setenv("OMP_PROC_BIND",   bind_strs[s.proc_bind], 1);
    setenv("OMP_DYNAMIC",     s.dynamic ? "true" : "false", 1);
    setenv("OMP_PLACES",      "cores",                1);
    

    #if defined(OS_LINUX)
        char exe_path[512];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
        if (len < 0) { perror("readlink"); return; }
        exe_path[len] = '\0';
        execl(exe_path, exe_path, image_path, nullptr);
        perror("execl failed");
    #elif defined(OS_MACOS)
        #include <mach-o/dyld.h> 
        char exe_path[512];
        uint32_t size = sizeof(exe_path);
        _NSGetExecutablePath(exe_path, &size);
        execl(exe_path, exe_path, image_path, nullptr);
        perror("execl failed");
    #elif defined(OS_WINDOWS)
        char exe_path[512];
        GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        // set env vars before spawning
        SetEnvironmentVariableA("OMP_NUM_THREADS", threads_str);
        SetEnvironmentVariableA("OMP_PROC_BIND",   bind_strs[s.proc_bind]);
        SetEnvironmentVariableA("OMP_DYNAMIC",     s.dynamic ? "true" : "false");
        SetEnvironmentVariableA("OMP_PLACES",      "cores");
        // spawn new instance and exit current one
        STARTUPINFOA si = {}; si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\"", exe_path, image_path);
        if (CreateProcessA(nullptr, cmd, nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &si, &pi)) {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            ExitProcess(0);  // exit current process
        }
    #endif

    // re-exec with same image path argument
    // execl replaces the current process — GLFW/ImGui cleanup won't run,
    // but that's fine for a dev tool
    execl(exe_path, exe_path, image_path, nullptr);
    perror("execl failed");  // only reached if execl fails
}   

// ─────────────────────────────────────────────────────────────────────────────
// main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const char* image_path = argc > 1 ? argv[1] : "/media/amiyaun/New Volume/cv algos/sobel_gui/wp4472820_jpg.jpg";
    printf("Running benchmarks on %s ...\n", image_path);


    // declare BEFORE the render loop, after DashboardData data = RunBenchmarks(...)

    OmpSettings current_settings;
    current_settings.num_threads = omp_get_max_threads();
    current_settings.max_threads = omp_get_max_threads();  // before ApplyOmpSettings

    ApplyOmpSettings(current_settings);
    DashboardData data = RunBenchmarks(image_path, current_settings);
    
    data.omp_settings = current_settings;
    

    // ── GLFW + OpenGL setup ──────────────────────────────────────────────────
    glfwSetErrorCallback(GlfwError);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1600, 960,
        "Sobel Benchmark Dashboard", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ── ImGui + ImPlot setup ─────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Optional: io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    TexCache tc;
    bool first = true;

    // declare BEFORE the render loop
    int last_proc_bind = -1;  // -1 means "not yet set"

    // ── Render loop ──────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Expose a Re-run button in the menu bar
        bool rerun = false;
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Benchmark")) {
                if (ImGui::MenuItem("Re-run benchmarks")) rerun = true;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (rerun || data.omp_settings.needs_rerun) {
            data.omp_settings.needs_rerun = false;
            OmpSettings s = data.omp_settings;

            // if proc_bind changed, we need to re-exec
            
            if (s.proc_bind != last_proc_bind) {
                last_proc_bind = s.proc_bind;
                RelaunchWithSettings(image_path, s);
                // if we get here execl failed, fall through to normal rerun
            }
            last_proc_bind = s.proc_bind;

            data = RunBenchmarks(image_path, s);
            data.omp_settings = s;
            first = true;
        }

        SobelDashboard_Draw(data, tc, first);
        first = false;

        ImGui::Render();
        int fw, fh;
        glfwGetFramebufferSize(window, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.08f, 0.08f, 0.10f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}


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

#include "sobel_dashboard.h"

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

    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Run all benchmarks and populate DashboardData
// ─────────────────────────────────────────────────────────────────────────────
static DashboardData RunBenchmarks(const char* image_path)
{
    DashboardData d;
    d.build = MakeBuildInfo();

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
        im.thread_count     = omp_get_max_threads();
        im.output_gray.resize(N);

        const uint8_t* src_ptr = gray.data;
        uint8_t*       dst_ptr = im.output_gray.data();
        int W = d.width, H = d.height;

        im.latency = BenchRuns([&]{
            int threads = omp_get_max_threads();

            sobel_avx2_omp(src_ptr,dst_ptr,H,W,threads);
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

    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLFW error callback
// ─────────────────────────────────────────────────────────────────────────────
static void GlfwError(int, const char* msg)
{
    fprintf(stderr, "GLFW error: %s\n", msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const char* image_path = argc > 1 ? argv[1] : "/media/amiyaun/New Volume/cv algos/sobel_gui/rain-forest-tree-view-up_jpg.png";
    printf("Running benchmarks on %s ...\n", image_path);

    DashboardData data = RunBenchmarks(image_path);

    static int gui_threads = omp_get_max_threads();
    static bool gui_proc_bind = true;
    static int gui_chunk_weight = 160;

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
        if (rerun) {
            data  = RunBenchmarks(image_path);
            first = true;  // forces texture refresh
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

// Declaration used in sobel_dashboard.cpp
void SobelDashboard_Draw(DashboardData& d, TexCache& tc, bool refresh_textures);    
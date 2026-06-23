// sobel_dashboard.cpp
// Requires: Dear ImGui (docking branch), ImPlot, OpenGL3 + GLFW backend
// Compile alongside: sobel_min.c, bench_sobel_opencv.cpp, main_dashboard.cpp
//
// Link: -lGL -lglfw -lopencv_core -lopencv_imgproc -fopenmp

#include "sobel_dashboard.h"
#include <omp.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>


// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════


static GpuTex UploadGray(const std::vector<uint8_t>& pixels, int w, int h,
                          GpuTex existing = {})
{
    if (existing.id == 0)
        glGenTextures(1, &existing.id);

    glBindTexture(GL_TEXTURE_2D, existing.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0,
                 GL_RED, GL_UNSIGNED_BYTE, pixels.data());

    // Swizzle R→RGB so it displays as grey not red
    GLint swizzle[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);

    existing.width  = w;
    existing.height = h;
    return existing;
}

void TexCache::Refresh(const DashboardData& d)
{
    input = UploadGray(
        d.input_gray,
        d.width,
        d.height,
        input);

    impls.resize(d.impls.size());

    for(size_t i = 0; i < d.impls.size(); i++)
    {
        impls[i] = UploadGray(
            d.impls[i].output_gray,
            d.impls[i].width,
            d.impls[i].height,
            impls[i]);
    }
}

// Fit image into available content region, preserving aspect ratio
static ImVec2 FitImage(int imgW, int imgH, ImVec2 available)
{
    float scaleX = available.x / (float)imgW;
    float scaleY = available.y / (float)imgH;
    float scale  = std::min(scaleX, scaleY);
    return { imgW * scale, imgH * scale };
}

// Color for sequential / thread-parallel / SIMD stages
static ImVec4 ColSeq()    { return {0.90f, 0.40f, 0.35f, 1.f}; }  // red-ish
static ImVec4 ColThread() { return {0.35f, 0.75f, 0.45f, 1.f}; }  // green
static ImVec4 ColSIMD()   { return {0.30f, 0.60f, 0.95f, 1.f}; }  // blue
static ImVec4 ColGray()   { return {0.55f, 0.55f, 0.55f, 1.f}; }

static void LabeledSep(const char* label)
{
    ImGui::Spacing();
    ImGui::TextColored({0.8f,0.8f,0.3f,1.f}, "%s", label);
    ImGui::Separator();
    ImGui::Spacing();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 1-4 : Image viewer
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawImagePanel(const char* label, const GpuTex& tex)
{
    if (!ImGui::BeginChild(label, ImVec2(0,0), true)) {
        ImGui::EndChild(); return;
    }
    ImGui::TextColored({0.9f,0.9f,0.5f,1.f}, "%s", label);
    ImGui::Separator();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= 4;
    if (tex.id && tex.width > 0) {
        ImVec2 sz = FitImage(tex.width, tex.height, avail);
        // Centre horizontally
        float pad = (avail.x - sz.x) * 0.5f;
        if (pad > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
        ImGui::Image((ImTextureID)(intptr_t)tex.id, sz);
        if (ImGui::IsItemHovered()) {
            ImVec2 uv   = ImGui::GetMousePos();
            ImVec2 item = ImGui::GetItemRectMin();
            int px = (int)((uv.x - item.x) / sz.x * tex.width);
            int py = (int)((uv.y - item.y) / sz.y * tex.height);
            ImGui::SetTooltip("(%d, %d)", px, py);
        }
    } else {
        ImGui::TextDisabled("(no data)");
    }
    ImGui::EndChild();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 5 : Latency statistics + history graph
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawLatencyPanel(const std::vector<ImplResult>& impls)
{
    LabeledSep("Latency Statistics");

    // Table: stat columns
    if (ImGui::BeginTable("lat_tbl", 6,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Implementation", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Mean (ms)");
        ImGui::TableSetupColumn("Std (ms)");
        ImGui::TableSetupColumn("Min (ms)");
        ImGui::TableSetupColumn("Max (ms)");
        ImGui::TableSetupColumn("Runs");
        ImGui::TableHeadersRow();

        for (auto& im : impls) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(im.name.c_str());
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", im.latency.mean_ms);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", im.latency.stddev_ms);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", im.latency.min_ms);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%.3f", im.latency.max_ms);
            ImGui::TableSetColumnIndex(5); ImGui::Text("%zu",  im.latency.history_ms.size());
        }
        ImGui::EndTable();
    }

    // History line graph via ImPlot
    ImGui::Spacing();
    if (ImPlot::BeginPlot("Latency History", ImVec2(-1, 180))) {
        ImPlot::SetupAxes("Run #", "ms");
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        static const ImVec4 colors[] = {
            {0.30f,0.70f,1.00f,1.f},
            {0.40f,0.85f,0.40f,1.f},
            {1.00f,0.60f,0.20f,1.f},
        };
        int ci = 0;
        for (auto& im : impls) {
            if (im.latency.history_ms.empty()) { ++ci; continue; }
            std::vector<float> fy(im.latency.history_ms.begin(),
                                  im.latency.history_ms.end());
            ImPlot::PlotLine(im.name.c_str(), fy.data(), (int)fy.size());
            
            ++ci;
        }
        ImPlot::EndPlot();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 6 : Memory bandwidth
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawBandwidthPanel(const std::vector<ImplResult>& impls)
{
    LabeledSep("Memory Bandwidth");

    if (ImGui::BeginTable("bw_tbl", 4,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Implementation", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Achieved (GB/s)");
        ImGui::TableSetupColumn("Theoretical (GB/s)");
        ImGui::TableSetupColumn("% of Peak");
        ImGui::TableHeadersRow();

        for (auto& im : impls) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(im.name.c_str());
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", im.bandwidth.achieved_gb_s);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.2f", im.bandwidth.theoretical_gb_s);
            ImGui::TableSetColumnIndex(3);
            float pct = (float)im.bandwidth.pct_of_peak;
            ImVec4 col = pct > 70.f ? ImVec4{0.3f,0.9f,0.3f,1.f}
                       : pct > 40.f ? ImVec4{0.9f,0.8f,0.2f,1.f}
                                    : ImVec4{0.9f,0.3f,0.3f,1.f};
            ImGui::TextColored(col, "%.1f%%", pct);
        }
        ImGui::EndTable();
    }

    // Progress bars
    ImGui::Spacing();
    for (auto& im : impls) {
        float frac = (float)std::min(im.bandwidth.pct_of_peak / 100.0, 1.0);
        char ovl[64]; snprintf(ovl, sizeof(ovl), "%s  %.1f%%",
                               im.name.c_str(), im.bandwidth.pct_of_peak);
        ImVec4 col = frac > 0.7f ? ImVec4{0.2f,0.8f,0.2f,1.f}
                   : frac > 0.4f ? ImVec4{0.8f,0.7f,0.1f,1.f}
                                 : ImVec4{0.8f,0.2f,0.2f,1.f};
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        ImGui::ProgressBar(frac, ImVec2(-1,18), ovl);
        ImGui::PopStyleColor();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 7 : Thread information
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawThreadInfoPanel(const DashboardData& d)
{
    LabeledSep("Thread Information");

    const BuildInfo& b = d.build;
    ImGui::Text("Logical CPUs       : %d", b.logical_cpu_count);
    ImGui::Text("OMP_NUM_THREADS    : %s", b.omp_num_threads.c_str());
    ImGui::Text("OMP_PROC_BIND      : %s", b.omp_proc_bind.c_str());
    ImGui::Text("OMP_PLACES         : %s", b.omp_places.c_str());
    ImGui::Text("OpenMP runtime     : %s", b.openmp_runtime.c_str());

    if (!d.threads.empty()) {
        ImGui::Spacing();
        ImGui::TextColored({0.8f,0.8f,0.3f,1.f}, "Per-thread CPU affinity snapshot:");
        if (ImGui::BeginTable("thr_cpu", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_ScrollY, ImVec2(0, 140)))
        {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Thread ID");
            ImGui::TableSetupColumn("CPU Core");
            ImGui::TableSetupColumn("Rows");
            ImGui::TableHeadersRow();
            for (auto& t : d.threads) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d",  t.thread_id);
                ImGui::TableSetColumnIndex(1);
                if (t.cpu_core >= 0)
                    ImGui::Text("%d", t.cpu_core);
                else
                    ImGui::TextDisabled("n/a");
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("[%d, %d)", t.row_start, t.row_end);
            }
            ImGui::EndTable();
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Panel 9 : Thread-to-core mapping table
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawThreadCoreTable(const DashboardData& d)
{
    LabeledSep("Thread → Core Mapping");

    if (d.threads.empty()) { ImGui::TextDisabled("No thread data."); return; }

    if (ImGui::BeginTable("thr_map", 5,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
        ImVec2(0, std::min((int)d.threads.size() + 2, 10) * 22.f)))
    {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Thread");
        ImGui::TableSetupColumn("CPU Core");
        ImGui::TableSetupColumn("Row Start");
        ImGui::TableSetupColumn("Row End");
        ImGui::TableSetupColumn("% Work");
        ImGui::TableHeadersRow();

        for (auto& t : d.threads) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%d", t.thread_id);
            ImGui::TableSetColumnIndex(1);
            if (t.cpu_core >= 0) ImGui::Text("%d", t.cpu_core);
            else ImGui::TextDisabled("–");
            ImGui::TableSetColumnIndex(2); ImGui::Text("%d", t.row_start);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%d", t.row_end);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%.1f%%", t.work_pct);
        }
        ImGui::EndTable();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 10 : Execution pipeline breakdown
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawPipelinePanel()
{
    LabeledSep("Execution Pipeline Breakdown");

    struct Stage { const char* name; const char* kind; ImVec4 col; const char* detail; };
    static const Stage stages[] = {
        {"Image Load / Gray Conversion", "Sequential",       {}, "Single-threaded I/O + conversion"},
        {"Row Partitioning",             "Sequential",       {}, "Divide height into OMP chunks"},
        {"OMP Parallel Region",          "Thread-Parallel",  {}, "Each thread owns a row stripe"},
        {"AVX2 Sobel Inner Loop",        "SIMD-Parallel",    {}, "16 px/iter: load→kernel→abs→min→store"},
        {"Scalar tail (width % 16)",     "Sequential",       {}, "Handles remaining columns"},
        {"Magnitude   |Gx|+|Gy|",        "SIMD-Parallel",    {}, "_mm256_add_epi16 per vector"},
        {"Saturate -> uint8",            "SIMD-Parallel",    {}, "_mm_packus_epi16 per vector"},
        {"OMP implicit barrier",         "Sequential",       {}, "Thread sync at region exit"},
        {"Output write",                 "Sequential",       {}, "memcpy to output buffer"},
    };

    // Tag colours filled in at render time
    for (auto& s : stages) {
        ImVec4 col = (strcmp(s.kind,"Sequential")==0)      ? ColSeq()
                   : (strcmp(s.kind,"Thread-Parallel")==0) ? ColThread()
                                                           : ColSIMD();
        ImGui::TextColored(col, "%-28s", s.name);
        ImGui::SameLine();
        ImGui::TextColored(ColGray(), " [%s]", s.kind);
        ImGui::SameLine();
        ImGui::TextDisabled("  %s", s.detail);
    }

    ImGui::Spacing();
    // Legend
    auto Dot = [](ImVec4 c, const char* lbl){
        ImGui::SameLine();
        ImGui::ColorButton("##", c, ImGuiColorEditFlags_NoTooltip, {12,12});
        ImGui::SameLine();
        ImGui::TextUnformatted(lbl);
    };
    ImGui::TextUnformatted("Legend:");
    Dot(ColSeq(),    " Sequential");
    Dot(ColThread(), " Thread-Parallel");
    Dot(ColSIMD(),   " SIMD-Parallel");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 11 : SIMD analysis
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawSIMDPanel(const BuildInfo& b)
{
    LabeledSep("SIMD / AVX2 Analysis");

    ImGui::Text("AVX2 available         : %s", b.avx2_available ? "YES" : "NO");
    ImGui::Text("Vector width           : 256 bits");
    ImGui::Text("Element width          : 16 bits (int16 intermediate)");
    ImGui::Text("Pixels per vector      : 16  (two _mm256 Gx/Gy lanes)");
    ImGui::Text("Gx kernel loads        : _mm256_loadu_si256 × 3 rows × 3 cols");
    ImGui::Text("Gy kernel loads        : same (reused registers)");
    ImGui::Text("Row pointer shifts     : _mm256_srli_si128 / manual offset loads");

    ImGui::Spacing();
    ImGui::TextColored({0.8f,0.8f,0.3f,1.f}, "Intrinsics used:");

    static const struct { const char* intr; const char* purpose; } intrs[] = {
        {"_mm256_loadu_si256",   "unaligned 256-bit load (3×3 neighbourhood)"},
        {"_mm256_cvtepu8_epi16", "zero-extend uint8→int16 before arithmetic"},
        {"_mm256_add_epi16",     "row summation for Gx / Gy"},
        {"_mm256_sub_epi16",     "row difference for Gx / Gy"},
        {"_mm256_slli_epi16",    "×2 multiply for centre row weight"},
        {"_mm256_abs_epi16",     "|Gx|, |Gy| (SSSE3 but universally on AVX2)"},
        {"_mm256_min_epu16",     "saturate 16-bit sum to 255 before pack"},
        {"_mm256_adds_epu16",    "saturating add |Gx|+|Gy|"},
        {"_mm_packus_epi16",     "pack two int16 lanes → uint8 with saturation"},
        {"_mm256_storeu_si256",  "unaligned 256-bit store to output row"},
    };

    if (ImGui::BeginTable("simd_tbl", 2,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Intrinsic");
        ImGui::TableSetupColumn("Purpose");
        ImGui::TableHeadersRow();
        for (auto& r : intrs) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored({0.4f,0.85f,1.f,1.f}, "%s", r.intr);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(r.purpose);
        }
        ImGui::EndTable();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 12 : Performance comparison table
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawPerfCompTable(const std::vector<ImplResult>& impls, int W, int H)
{
    LabeledSep("Performance Comparison");

    // Reference = index 2 (OpenCV float mag) or index 0 if only one impl
    double ref_mean = 1e-9;
    for (auto& im : impls)
        if (im.name.find("float") != std::string::npos ||
            im.name.find("OpenCV") != std::string::npos)
            ref_mean = std::max(ref_mean, im.latency.mean_ms);

    double mpix = W * H / 1e6;

    if (ImGui::BeginTable("perf_cmp", 7,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Implementation", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Mean (ms)");
        ImGui::TableSetupColumn("Mpx/s");
        ImGui::TableSetupColumn("BW (GB/s)");
        ImGui::TableSetupColumn("Threads");
        ImGui::TableSetupColumn("Formula");
        ImGui::TableSetupColumn("Speedup");
        ImGui::TableHeadersRow();

        for (auto& im : impls) {
            double ms  = im.latency.mean_ms;
            double mpx = ms > 0 ? mpix / (ms * 1e-3) : 0.0;
            double spd = ref_mean / std::max(ms, 1e-9);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(im.name.c_str());
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", ms);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", mpx);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", im.bandwidth.achieved_gb_s);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%d",   im.thread_count);
            ImGui::TableSetColumnIndex(5); ImGui::TextUnformatted(im.magnitude_formula.c_str());
            ImGui::TableSetColumnIndex(6);
            ImVec4 sc = spd >= 1.0 ? ImVec4{0.3f,0.9f,0.3f,1.f}
                                   : ImVec4{0.9f,0.4f,0.3f,1.f};
            ImGui::TextColored(sc, "%.2f×", spd);
        }
        ImGui::EndTable();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 13 : Image quality metrics
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawQualityPanel(const std::vector<ImplResult>& impls)
{
    LabeledSep("Image Quality (vs OpenCV float-mag reference)");

    if (ImGui::BeginTable("qual_tbl", 5,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Implementation", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Max Error");
        ImGui::TableSetupColumn("MAE");
        ImGui::TableSetupColumn("PSNR (dB)");
        ImGui::TableSetupColumn("Diff Pixels %");
        ImGui::TableHeadersRow();

        for (auto& im : impls) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(im.name.c_str());
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f",   im.quality.max_error);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f",   im.quality.mae);
            ImGui::TableSetColumnIndex(3);
            if (im.quality.psnr_db > 999.0)
                ImGui::TextColored({0.3f,0.9f,0.3f,1.f}, "∞ (identical)");
            else
                ImGui::Text("%.2f", im.quality.psnr_db);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%.2f%%", im.quality.differing_pct);
        }
        ImGui::EndTable();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 14 : Edge analysis
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawEdgeStatsPanel(const std::vector<ImplResult>& impls)
{
    LabeledSep("Edge Analysis");

    for (auto& im : impls) {
        if (ImGui::TreeNode(im.name.c_str())) {
            ImGui::Text("Avg intensity  : %.2f", im.edges.avg_intensity);
            ImGui::Text("Saturation %%   : %.2f%%", im.edges.saturation_pct);
            ImGui::Text("Edge density   : %.2f%%", im.edges.edge_density);

            // Build float histogram for ImPlot
            std::vector<float> hist(im.edges.histogram.begin(),
                                    im.edges.histogram.end());
            // Normalise to 0–1 for display
            float hmax = *std::max_element(hist.begin(), hist.end());
            if (hmax > 0)
                for (auto& v : hist) v /= hmax;

            if (ImPlot::BeginPlot(("Histogram##" + im.name).c_str(),
                                  ImVec2(-1, 120)))
            {
                ImPlot::SetupAxes("Value [0-255]", "Density");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, 256, ImPlotCond_Always);
                // x-axis: bin centres
                static float xs[256];
                static bool xs_init = false;
                if (!xs_init) {
                    for (int i = 0; i < 256; ++i) xs[i] = (float)i;
                    xs_init = true;
                }
                ImPlot::PlotBars(im.name.c_str(), xs, hist.data(), 256, 1.0);
                ImPlot::EndPlot();
            }
            ImGui::TreePop();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 15 : Build / runtime info
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawBuildInfoPanel(const BuildInfo& b)
{
    LabeledSep("Build & Runtime Information");

    ImGui::Text("CPU model          : %s", b.cpu_model.c_str());
    ImGui::Text("Compiler           : %s", b.compiler_version.c_str());
    ImGui::Text("CFLAGS             : %s", b.cflags.c_str());
    ImGui::Text("OpenMP runtime     : %s", b.openmp_runtime.c_str());
    ImGui::Text("AVX2 available     : %s", b.avx2_available ? "YES" : "NO");
    ImGui::Text("P/E core ratio     : %.3f  (%dP + %dE logical)",b.pe_weight_ratio, b.p_core_count, b.e_core_count);
    ImGui::Spacing();
    ImGui::Text("L1 cache           : %s", b.l1_cache.c_str());
    ImGui::Text("L2 cache           : %s", b.l2_cache.c_str());
    ImGui::Text("L3 cache           : %s", b.l3_cache.c_str());
    ImGui::Spacing();
    ImGui::Text("OpenCV version     : %s", b.opencv_version.c_str());
    ImGui::Text("OpenCV IPP         : %s", b.opencv_ipp    ? "YES" : "NO");
    ImGui::Text("OpenCV TBB         : %s", b.opencv_tbb    ? "YES" : "NO");
    ImGui::Text("OpenCV OpenMP      : %s", b.opencv_openmp ? "YES" : "NO");
    ImGui::Text("OpenCV threading   : %s", b.opencv_threading_backend.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 16 : Architecture diagram (drawn with ImDrawList)
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawArchDiagram()
{
    LabeledSep("Architecture Diagram");

    struct Node {
        const char* label;
        const char* sub;
        ImVec4      col;
    };
    static const Node nodes[] = {
        {"Input Image",            "HxW\n uint8 gray",     {0.55f,0.55f,0.55f,1.f}},
        {"Row Partitioning",       "height per\n nthreads",  {0.85f,0.45f,0.35f,1.f}},
        {"OMP Thread\nAssignment", "parallel for \nrows",  {0.35f,0.75f,0.45f,1.f}},
        {"AVX2 Sobel\nKernel",     "16 px per\n iteration",  {0.30f,0.60f,0.95f,1.f}},
        {"Magnitude\n|Gx|+|Gy|",  "adds_epu16",         {0.30f,0.60f,0.95f,1.f}},
        {"Saturate",               "packus_epi16 ->\n u8",  {0.30f,0.60f,0.95f,1.f}},
        {"Output Image",           "HxW uint8 \nedges",    {0.55f,0.55f,0.55f,1.f}},
    };
    const int N = sizeof(nodes)/sizeof(nodes[0]);

    ImDrawList* dl    = ImGui::GetWindowDrawList();
    ImVec2      base  = ImGui::GetCursorScreenPos();
    float       avail = ImGui::GetContentRegionAvail().x;

    const float boxW  = std::min(110.f, (avail - 20) / N - 8.f);
    const float boxH  = 56.f;
    const float gapX  = (avail - N * boxW) / (N - 1);
    const float arrowY= base.y + boxH / 2;

    for (int i = 0; i < N; ++i) {
        float x0 = base.x + i * (boxW + gapX);
        float y0 = base.y;
        float x1 = x0 + boxW;
        float y1 = y0 + boxH;

        // Box fill + border
        ImU32 bg   = ImGui::ColorConvertFloat4ToU32(
                         ImVec4(nodes[i].col.x*0.25f, nodes[i].col.y*0.25f,
                                nodes[i].col.z*0.25f, 0.90f));
        ImU32 bord = ImGui::ColorConvertFloat4ToU32(nodes[i].col);
        dl->AddRectFilled({x0,y0},{x1,y1}, bg,  6.f);
        dl->AddRect       ({x0,y0},{x1,y1}, bord, 6.f, 0, 2.f);

        // Text centred
        ImVec2 ts = ImGui::CalcTextSize(nodes[i].label);
        float  tx = x0 + (boxW - ts.x) * 0.5f;
        float  ty = y0 + 6;
        dl->AddText({tx, ty}, IM_COL32(230,230,230,255), nodes[i].label);

        ImVec2 ss = ImGui::CalcTextSize(nodes[i].sub);
        float  sx = x0 + (boxW - ss.x) * 0.5f;
        dl->AddText({sx, ty + ts.y + 2},
                    IM_COL32(160,160,160,255), nodes[i].sub);

        // Arrow to next box
        if (i < N - 1) {
            float ax0 = x1 + 2;
            float ax1 = x1 + gapX - 2;
            dl->AddLine({ax0, arrowY}, {ax1, arrowY},
                        IM_COL32(180,180,180,200), 2.f);
            // arrowhead
            float ah = 6.f;
            dl->AddTriangleFilled(
                {ax1, arrowY},
                {ax1 - ah, arrowY - ah * 0.6f},
                {ax1 - ah, arrowY + ah * 0.6f},
                IM_COL32(180,180,180,200));
        }
    }
    ImGui::Dummy(ImVec2(avail, boxH + 8));

    // Legend
    auto Dot = [&](ImVec4 c, const char* lbl) {
        ImGui::ColorButton("##leg", c,
            ImGuiColorEditFlags_NoTooltip |
            ImGuiColorEditFlags_NoBorder, {12,12});
        ImGui::SameLine();
        ImGui::TextUnformatted(lbl);
        ImGui::SameLine(0, 20);
    };
    Dot(ColSeq(),    "Sequential");
    Dot(ColThread(), "Thread-Parallel");
    Dot(ColSIMD(),   "SIMD-Parallel");
    ImGui::NewLine();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 17 : Roofline visualisation
// ═══════════════════════════════════════════════════════════════════════════════

static void DrawRooflinePanel(const DashboardData& d)
{
    LabeledSep("Roofline Model");

    // X axis: arithmetic intensity (FLOP/byte), log scale
    // Y axis: performance (GFLOP/s), log scale
    // Ridge point = peak_compute / peak_bandwidth

    // Estimate peak GFLOP/s from CPU (very rough for Alder Lake):
    // i7-1255U: 10 cores × 2 AVX256 FMAs × 8 floats × ~4.7 GHz ≈ ~750 GFLOP/s
    // But for integer: lower; use ~100 GFLOP/s as placeholder
    const double peak_compute_gflops = 100.0;  // adjust

    // Pull reference bandwidth from first impl that has it
    double peak_bw = 40.0;  // GB/s fallback
    for (auto& im : d.impls)
        if (im.bandwidth.theoretical_gb_s > 1.0)
            peak_bw = im.bandwidth.theoretical_gb_s;

    double ridge = peak_compute_gflops / peak_bw;  // FLOP/byte

    if (ImPlot::BeginPlot("Roofline", ImVec2(-1, 240))) {
        ImPlot::SetupAxes("Arithmetic Intensity (FLOP/byte)",
                          "Performance (GFLOP/s)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.01, 100.0, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1,  0.1, peak_compute_gflops * 2.0,
                                ImPlotCond_Always);

        // Memory-bound slope
        {
            static double xs[] = {0.01, ridge};
            static double ys_mem[2];
            ys_mem[0] = 0.01 * peak_bw;
            ys_mem[1] = ridge * peak_bw;
            
            ImPlot::PlotLine("Mem BW roof", xs, ys_mem, 2);
        }
        // Compute-bound ceiling
        {
            static double xs[] = {0.0, 0.0};  // filled below
            static double ys_cmp[2];
            xs[0]       = ridge;
            xs[1]       = 100.0;
            ys_cmp[0]   = peak_compute_gflops;
            ys_cmp[1]   = peak_compute_gflops;
            
            
            ImPlot::PlotLine("Compute roof", xs, ys_cmp, 2);
        }

        // Plot each implementation as a scatter point
        for (auto& im : d.impls) {
            if (d.arithmetic_intensity <= 0.0 || im.latency.mean_ms <= 0.0)
                continue;
            // GFLOP/s = (pixels × ops_per_pixel) / time_s  (approx)
            int W = im.width, H = im.height;
            double pixels = W * H;
            // ~10 ops per pixel (adds, subs, abs, min for L1 Sobel)
            double gflops = pixels * 10.0 / (im.latency.mean_ms * 1e-3) / 1e9;
            double ai     = d.arithmetic_intensity;

            double xs[1] = {ai};
            double ys[1] = {gflops};

            
            ImPlot::PlotScatter(im.name.c_str(), xs, ys, 1);
        }

        ImPlot::EndPlot();
    }

    ImGui::TextDisabled(
        "Ridge point ≈ %.2f FLOP/byte  |  Peak BW %.1f GB/s  |  Peak compute %.0f GFLOP/s",
        ridge, peak_bw, peak_compute_gflops);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel 18 :Omp Settings
// ═══════════════════════════════════════════════════════════════════════════════
static void DrawOmpSettingsPanel(OmpSettings& s)
{
    LabeledSep("OpenMP Runtime Settings");

    ImGui::Text("Changes take effect on next run.");
    ImGui::Spacing();

    // remove: int max_t = omp_get_max_threads();
    // change slider to:
    

    // Thread count slider
    int t = s.num_threads;
    
    if (ImGui::SliderInt("Num Threads", &t, 1, s.max_threads)) {
        s.num_threads   = t;
        s.needs_rerun   = true;
    }

    // Proc bind combo
    static const char* bind_options[] = {"close", "spread", "master"};
    int b = s.proc_bind;
    if (ImGui::Combo("OMP_PROC_BIND", &b, bind_options, 3)) {
        s.proc_bind   = b;
        s.needs_rerun = true;
    }

    // Dynamic toggle
    bool dyn = s.dynamic;
    if (ImGui::Checkbox("OMP_DYNAMIC", &dyn)) {
        s.dynamic     = dyn;
        s.needs_rerun = true;
    }

    ImGui::Spacing();
    if (ImGui::Button("Run Now", ImVec2(120, 0)))
        s.needs_rerun = true;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Active: threads=%d  bind=%s  dynamic=%s",
        s.num_threads,
        bind_options[s.proc_bind],
        s.dynamic ? "true" : "false");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main entry point  (call once per frame after ImGui::NewFrame())
// ═══════════════════════════════════════════════════════════════════════════════

void SobelDashboard_Draw(DashboardData& d, TexCache& tc, bool refresh_textures)
{
    if (refresh_textures)
        tc.Refresh(d);

    // ── Full-screen dockspace ─────────────────────────────────────────────────
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
    ImGui::Begin("##root", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNav        | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoDocking    | ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar(3);

    ImGuiID dsid = ImGui::GetID("MainDockspace");
    ImGui::DockSpace(dsid, ImVec2(0,0), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    // near the top of SobelDashboard_Draw, after the dockspace
    if (d.show_omp_popup)
        ImGui::OpenPopup("OMPSettingsPopup");

    ImGui::SetNextWindowSize(ImVec2(400, 260), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("OMPSettingsPopup", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove))
    {
        if (!d.benchmark_running) {
            DrawOmpSettingsPanel(d.omp_settings);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Run", ImVec2(120, 0))) {
                d.omp_settings.needs_rerun = true;
                d.benchmark_running = true;
                d.show_omp_popup = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                d.show_omp_popup = false;
                ImGui::CloseCurrentPopup();
            }
        } else {
            ImGui::Spacing();
            ImGui::TextColored({1.f,0.8f,0.2f,1.f}, "Running benchmarks, please wait...");
            ImGui::Spacing();
            static int dot_frame = 0;
            dot_frame = (dot_frame + 1) % 120;
            const char* dots[] = {"   ", ".  ", ".. ", "..."};
            ImGui::Text("Computing%s", dots[dot_frame / 30]);
            if (!d.benchmark_running)
                ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    

    // ── Window : Images ───────────────────────────────────────────────────────
    if (ImGui::Begin("Images")) {
        if (ImGui::BeginTabBar("img_tabs")) {
            if (ImGui::BeginTabItem("Input")) {
                DrawImagePanel("Input (gray)", tc.input);
                ImGui::EndTabItem();
            }
            for (int i = 0; i < (int)d.impls.size(); ++i) {
                if (ImGui::BeginTabItem(d.impls[i].name.c_str())) {
                    DrawImagePanel(d.impls[i].name.c_str(),
                                   i < (int)tc.impls.size() ? tc.impls[i] : GpuTex{});
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    // ── Window : Performance ─────────────────────────────────────────────────
    if (ImGui::Begin("Performance")) {
        DrawLatencyPanel(d.impls);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawBandwidthPanel(d.impls);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawPerfCompTable(d.impls, d.width, d.height);
    }
    ImGui::End();

    // ── Window : Threads ─────────────────────────────────────────────────────
    if (ImGui::Begin("Threads")) {
        DrawThreadInfoPanel(d);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawThreadCoreTable(d);
    }
    ImGui::End();

    // ── Window : Architecture ────────────────────────────────────────────────
    if (ImGui::Begin("Architecture")) {
        DrawArchDiagram();
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawPipelinePanel();
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawSIMDPanel(d.build);
    }
    ImGui::End();

    // ── Window : Quality ─────────────────────────────────────────────────────
    if (ImGui::Begin("Quality")) {
        DrawQualityPanel(d.impls);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        DrawEdgeStatsPanel(d.impls);
    }
    ImGui::End();

    // ── Window : Build Info ──────────────────────────────────────────────────
    if (ImGui::Begin("Build Info")) {
        DrawBuildInfoPanel(d.build);
    }
    ImGui::End();

    // ── Window : Roofline ────────────────────────────────────────────────────
    if (ImGui::Begin("Roofline")) {
        DrawRooflinePanel(d);
    }
    ImGui::End();
}
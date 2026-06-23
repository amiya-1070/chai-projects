#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define N_PIXELS (1920 * 1080)
#define REPEATS  100
#define PEAK_BW  32.0

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void gen_image(uint8_t* img, int n) {
    uint64_t state = 42;
    for (int i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        img[i] = (uint8_t)(state >> 56);
    }
}

static void histogram_reference(const uint8_t* img, uint32_t* hist, int n) {
    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < n; i++) hist[img[i]]++;
}

/* ── full equalization reference (for correctness checking) ─────────────── */
static void equalize_reference(
    const uint8_t* img, uint8_t* out,
    uint32_t* hist, int n)
{
    /* stage 1: histogram */
    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < n; i++) hist[img[i]]++;

    /* stage 2: cdf */
    float cdf[256];
    float cumsum = 0.0f, cdf_min = -1.0f;
    for (int r = 0; r < 256; r++) {
        cumsum += (float)hist[r] / n;
        cdf[r]  = cumsum;
        if (cdf_min < 0.0f && hist[r] > 0)
            cdf_min = (float)hist[r] / n;
    }

    /* stage 3: lut */
    uint8_t lut[256];
    for (int r = 0; r < 256; r++) {
        float mapped = (cdf[r] - cdf_min) / (1.0f - cdf_min) * 255.0f + 0.5f;
        if (mapped < 0.0f)   mapped = 0.0f;
        if (mapped > 255.0f) mapped = 255.0f;
        lut[r] = (uint8_t)mapped;
    }

    /* stage 4: lut apply */
    for (int i = 0; i < n; i++)
        out[i] = lut[img[i]];
}

/* ── scalar full pipeline ────────────────────────────────────────────────── */
static void equalize_scalar(
    const uint8_t* img, uint8_t* out,
    uint32_t* hist, int n)
{
    /* stage 1: histogram */
    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < n; i++) hist[img[i]]++;

    /* stage 2: cdf */
    float cdf[256];
    float cumsum = 0.0f, cdf_min = -1.0f;
    for (int r = 0; r < 256; r++) {
        cumsum += (float)hist[r] / n;
        cdf[r]  = cumsum;
        if (cdf_min < 0.0f && hist[r] > 0)
            cdf_min = (float)hist[r] / n;
    }

    /* stage 3: lut */
    uint8_t lut[256];
    for (int r = 0; r < 256; r++) {
        float mapped = (cdf[r] - cdf_min) / (1.0f - cdf_min) * 255.0f + 0.5f;
        if (mapped < 0.0f)   mapped = 0.0f;
        if (mapped > 255.0f) mapped = 255.0f;
        lut[r] = (uint8_t)mapped;
    }

    /* stage 4: lut apply — scalar */
    for (int i = 0; i < n; i++)
        out[i] = lut[img[i]];
}

/* ── avx2 lut apply ──────────────────────────────────────────────────────── */
/* processes 32 pixels per iteration using _mm256_shuffle_epi8.
   the trick: shuffle can act as a 16-entry parallel LUT per 128-bit lane.
   we split the 256-entry lut into 16 sub-luts of 16 entries each,
   test which sub-lut each pixel belongs to, and OR the results.
   each pixel value v selects lut[v] — 32 pixels in one instruction group. */
__attribute__((target("avx2")))
static void lut_apply_parallel(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    const uint8_t* __restrict__ lut,
    int n, int T)
{
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int i = 0; i < n; i++)
        out[i] = lut[img[i]];
}

/* ── avx2 parallel histogram (your existing function, unchanged) ─────────── */
__attribute__((target("avx2")))
static void histogram_avx2_parallel(
    const uint8_t* img, uint32_t* hist, uint32_t* ph, int n, int T)
{
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        uint32_t* h0 = ph + tid*8*256 + 0*256;
        uint32_t* h1 = ph + tid*8*256 + 1*256;
        uint32_t* h2 = ph + tid*8*256 + 2*256;
        uint32_t* h3 = ph + tid*8*256 + 3*256;
        uint32_t* h4 = ph + tid*8*256 + 4*256;
        uint32_t* h5 = ph + tid*8*256 + 5*256;
        uint32_t* h6 = ph + tid*8*256 + 6*256;
        uint32_t* h7 = ph + tid*8*256 + 7*256;

        memset(h0, 0, 8 * 256 * sizeof(uint32_t));

        int chunk = (n + T - 1) / T;
        int start = tid * chunk;
        int end   = start + chunk < n ? start + chunk : n;

        int i = start;
        for (; i <= end - 8; i += 8) {
            h0[img[i+0]]++;
            h1[img[i+1]]++;
            h2[img[i+2]]++;
            h3[img[i+3]]++;
            h4[img[i+4]]++;
            h5[img[i+5]]++;
            h6[img[i+6]]++;
            h7[img[i+7]]++;
        }
        for (; i < end; i++) h0[img[i]]++;
    }

    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int t = 0; t < T; t++) {
        uint32_t* base = ph + t * 8 * 256;
        for (int b = 0; b < 256; b += 8) {
            __m256i acc = _mm256_loadu_si256((__m256i*)(hist + b));
            for (int k = 0; k < 8; k++) {
                __m256i tmp = _mm256_load_si256(
                                (__m256i*)(base + k*256 + b));
                acc = _mm256_add_epi32(acc, tmp);
            }
            _mm256_storeu_si256((__m256i*)(hist + b), acc);
        }
    }
}

/* ── full pipeline: parallel histogram + serial cdf + avx2 lut apply ─────── */
__attribute__((target("avx2")))
static void equalize_parallel(
    const uint8_t* img, uint8_t* out,
    uint32_t* hist, uint32_t* ph,
    int n, int T)
{
    /* stage 1: parallel histogram */
    histogram_avx2_parallel(img, hist, ph, n, T);

    /* stage 2: cdf — serial, 255 additions, negligible cost */
    float cdf[256];
    float cumsum = 0.0f, cdf_min = -1.0f;
    for (int r = 0; r < 256; r++) {
        cumsum += (float)hist[r] / n;
        cdf[r]  = cumsum;
        if (cdf_min < 0.0f && hist[r] > 0)
            cdf_min = (float)hist[r] / n;
    }

    /* stage 3: lut build — serial, 256 ops, negligible cost */
    uint8_t lut[256];
    for (int r = 0; r < 256; r++) {
        float mapped = (cdf[r] - cdf_min) / (1.0f - cdf_min) * 255.0f + 0.5f;
        if (mapped < 0.0f)   mapped = 0.0f;
        if (mapped > 255.0f) mapped = 255.0f;
        lut[r] = (uint8_t)mapped;
    }

    /* stage 4: lut apply — parallel scalar */
    lut_apply_parallel(img, out, lut, n, T);
}

/* ── helpers ─────────────────────────────────────────────────────────────── */
static int output_equal(const uint8_t* a, const uint8_t* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static double dmean(const double* a, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s/n;
}
static double dstd(const double* a, int n) {
    double m = dmean(a,n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s/n);
}

static void equalize_parallel_timed(
    const uint8_t* img, uint8_t* out,
    uint32_t* hist, uint32_t* ph,
    int n, int T,
    double* t_hist, double* t_cdf,
    double* t_lut, double* t_apply)
{
    uint64_t t0, t1;

    t0 = now_ns();
    histogram_avx2_parallel(img, hist, ph, n, T);
    *t_hist = (double)(now_ns() - t0);

    t0 = now_ns();
    float cdf[256];
    float cumsum = 0.0f, cdf_min = -1.0f;
    for (int r = 0; r < 256; r++) {
        cumsum += (float)hist[r] / n;
        cdf[r]  = cumsum;
        if (cdf_min < 0.0f && hist[r] > 0)
            cdf_min = (float)hist[r] / n;
    }
    *t_cdf = (double)(now_ns() - t0);

    t0 = now_ns();
    uint8_t lut[256];
    for (int r = 0; r < 256; r++) {
        float mapped = (cdf[r] - cdf_min) / (1.0f - cdf_min) * 255.0f + 0.5f;
        if (mapped < 0.0f)   mapped = 0.0f;
        if (mapped > 255.0f) mapped = 255.0f;
        lut[r] = (uint8_t)mapped;
    }
    *t_lut = (double)(now_ns() - t0);

    t0 = now_ns();
    lut_apply_parallel(img, out, lut, n, T);
    *t_apply = (double)(now_ns() - t0);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    int T = omp_get_max_threads();
    printf("Logical CPUs: %d\n\n", T);

    /* prime thread pool */
    #pragma omp parallel num_threads(T)
    { int x = omp_get_thread_num(); (void)x; }

    uint8_t*  img  = aligned_alloc(64, N_PIXELS);
    uint8_t*  out  = aligned_alloc(64, N_PIXELS);
    uint8_t*  ref  = aligned_alloc(64, N_PIXELS);
    uint32_t* hist = aligned_alloc(64, 256 * sizeof(uint32_t));
    uint32_t* href = aligned_alloc(64, 256 * sizeof(uint32_t));
    uint32_t* ph   = aligned_alloc(64, T * 8 * 256 * sizeof(uint32_t));
    double*   times = malloc(REPEATS * sizeof(double));

    const char* names[] = {"RANDOM", "UNIFORM", "GRADIENT"};

    for (int tc = 0; tc < 3; tc++) {
        printf("\n===== %s =====\n", names[tc]);

        if (tc == 0) gen_image(img, N_PIXELS);
        if (tc == 1) memset(img, 128, N_PIXELS);
        if (tc == 2) for (int i = 0; i < N_PIXELS; i++) img[i] = i & 255;

        /* compute reference output */
        equalize_reference(img, ref, href, N_PIXELS);

        printf("%-30s  %10s  %9s  %10s  %10s  %s\n",
            "version", "mean(ns)", "std(ns)", "speedup", "BW(GB/s)", "correct");
        printf("%.90s\n", "------------------------------------------------------------"
                          "------------------------------");

        double scalar_mean = 0.0;

        /* ── scalar full pipeline ── */
        equalize_scalar(img, out, hist, N_PIXELS);   /* warmup */
        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            equalize_scalar(img, out, hist, N_PIXELS);
            times[r] = (double)(now_ns() - t0);
        }
        scalar_mean = dmean(times, REPEATS);
        printf("%-30s  %10.0f  %9.0f  %10.2f  %10.2f  %s\n",
            "scalar_full",
            scalar_mean, dstd(times, REPEATS), 1.0,
            /* bytes: read img once (hist) + read img again (lut apply) + write out */
            (double)(N_PIXELS * 3) / (scalar_mean * 1e-9) / 1e9,
            output_equal(out, ref, N_PIXELS) ? "correct" : "WRONG");

        /* ── parallel full pipeline ── */
        equalize_parallel(img, out, hist, ph, N_PIXELS, T);   /* warmup ×3 */
        equalize_parallel(img, out, hist, ph, N_PIXELS, T);
        equalize_parallel(img, out, hist, ph, N_PIXELS, T);
        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            equalize_parallel(img, out, hist, ph, N_PIXELS, T);
            times[r] = (double)(now_ns() - t0);
        }
        double m = dmean(times, REPEATS);
        printf("%-30s  %10.0f  %9.0f  %10.2f  %10.2f  %s\n",
            "parallel_full",
            m, dstd(times, REPEATS), scalar_mean / m,
            (double)(N_PIXELS * 3) / (m * 1e-9) / 1e9,
            output_equal(out, ref, N_PIXELS) ? "correct" : "WRONG");
    }

    double t_hist, t_cdf, t_lut, t_apply;
    equalize_parallel_timed(img, out, hist, ph, N_PIXELS, T,
        &t_hist, &t_cdf, &t_lut, &t_apply);
    printf("\nStage breakdown (single run):\n");
    printf("  histogram : %.0f ns  (%.1f%%)\n", t_hist,
        100.0*t_hist/(t_hist+t_cdf+t_lut+t_apply));
    printf("  cdf       : %.0f ns  (%.1f%%)\n", t_cdf,
        100.0*t_cdf/(t_hist+t_cdf+t_lut+t_apply));
    printf("  lut build : %.0f ns  (%.1f%%)\n", t_lut,
        100.0*t_lut/(t_hist+t_cdf+t_lut+t_apply));
    printf("  lut apply : %.0f ns  (%.1f%%)\n", t_apply,
        100.0*t_apply/(t_hist+t_cdf+t_lut+t_apply));

    free(img); free(out); free(ref);
    free(hist); free(href); free(ph); free(times);
    return 0;
}
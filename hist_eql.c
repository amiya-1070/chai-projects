#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N_IMAGES 100
#define N_PIXELS  (1080 * 1920)
#define REPEATS   20
#define PEAK_BW   32.0  /* GB/s — change to match your RAM */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ── image generation (replaces numpy rng) ──────────────────────────────── */
static void gen_image(uint8_t* img, int n) {
    /* simple LCG — same distribution as rng.integers(0,256) */
    uint64_t state = 42;
    for (int i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        img[i] = (uint8_t)(state >> 56);
    }
}

/* ── reference histogram for correctness ───────────────────────────────── */
static void histogram_reference(const uint8_t* img, uint32_t* hist, int n) {
    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        hist[img[i]]++;
}

/* ── version 1: serial ──────────────────────────────────────────────────── */
static void histogram_serial(const uint8_t* img, uint32_t* hist, int n) {
    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        hist[img[i]]++;
}

/* ── version 2: naive parallel (false sharing, racy) ───────────────────── */
static void histogram_naive_parallel(
    const uint8_t* img, uint32_t* hist, int n, int T)
{
    memset(hist, 0, 256 * sizeof(uint32_t));
    #pragma omp parallel num_threads(T)
    {
        int tid   = omp_get_thread_num();
        int chunk = (n + T - 1) / T;
        int start = tid * chunk;
        int end   = start + chunk < n ? start + chunk : n;
        for (int i = start; i < end; i++)
            hist[img[i]]++;   /* racy write — intentional */
    }
}

/* ── version 3: privatized ──────────────────────────────────────────────── */
static void histogram_privatized(
    const uint8_t* img, uint32_t* hist, int n, int T)
{
    uint32_t* ph = (uint32_t*)aligned_alloc(
        64, T * 256 * sizeof(uint32_t));
    memset(ph, 0, T * 256 * sizeof(uint32_t));

    #pragma omp parallel num_threads(T)
    {
        int tid   = omp_get_thread_num();
        uint32_t* lh = ph + tid * 256;
        int chunk = (n + T - 1) / T;
        int start = tid * chunk;
        int end   = start + chunk < n ? start + chunk : n;
        for (int i = start; i < end; i++)
            lh[img[i]]++;
    }

    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int t = 0; t < T; t++)
        for (int b = 0; b < 256; b++)
            hist[b] += ph[t * 256 + b];

    free(ph);
}

/* ── version 4: privatized + prefetch ──────────────────────────────────── */
static void histogram_prefetch(
    const uint8_t* img, uint32_t* hist, int n, int T)
{
    uint32_t* ph = (uint32_t*)aligned_alloc(
        64, T * 256 * sizeof(uint32_t));
    memset(ph, 0, T * 256 * sizeof(uint32_t));

    #pragma omp parallel num_threads(T)
    {
        int tid   = omp_get_thread_num();
        uint32_t* lh = ph + tid * 256;
        int chunk = (n + T - 1) / T;
        int start = tid * chunk;
        int end   = start + chunk < n ? start + chunk : n;
        for (int i = start; i < end; i++) {
            if (i + 512 < end)
                __builtin_prefetch(&img[i + 512], 0, 1);
            lh[img[i]]++;
        }
    }

    memset(hist, 0, 256 * sizeof(uint32_t));
    for (int t = 0; t < T; t++)
        for (int b = 0; b < 256; b++)
            hist[b] += ph[t * 256 + b];

    free(ph);
}

/* ── helpers ────────────────────────────────────────────────────────────── */
static int hist_equal(const uint32_t* a, const uint32_t* b) {
    for (int i = 0; i < 256; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static double mean(const double* a, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s / n;
}

static double stddev(const double* a, int n) {
    double m = mean(a, n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s / n);
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(void) {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12};
    int n_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("Logical CPUs available: %d\n\n", omp_get_max_threads());

    uint8_t*  img  = (uint8_t*)malloc(N_PIXELS);
    uint32_t* href = (uint32_t*)malloc(256 * sizeof(uint32_t));
    uint32_t* hist = (uint32_t*)malloc(256 * sizeof(uint32_t));
    double*   times = (double*)malloc(REPEATS * sizeof(double));

    memset(img, 128, N_PIXELS);

    histogram_reference(img, href, N_PIXELS);

    /* kernel table */
    typedef void (*kern_fn)(const uint8_t*, uint32_t*, int, int);
    const char* names[] = {
        "serial", "naive_par(racy)", "privatized", "prefetch"
    };
    /* serial doesn't take T but we wrap it uniformly */
    kern_fn kerns[] = {
        (kern_fn)histogram_serial,
        histogram_naive_parallel,
        histogram_privatized,
        histogram_prefetch
    };
    int n_kerns = 4;

    printf("%-8s  %12s  %12s  %12s  %12s\n",
        "threads",
        "mean(ns)",
        "std(ns)",
        "speedup",
        "eff BW");

    for (int i = 0; i < 70; i++) putchar('-');
    putchar('\n');

    double baseline = 0.0;

    for (int tc = 0; tc < n_tests; tc++) {

        int T = thread_counts[tc];

        for (int k = 0; k < N_IMAGES; k++)
            histogram_privatized(img, hist, N_PIXELS, T);

        #define N_IMAGES 100

        for (int r = 0; r < REPEATS; r++) {

            uint64_t t0 = now_ns();

            for (int k = 0; k < N_IMAGES; k++) {

                memset(hist, 0, 256 * sizeof(uint32_t));
                

                histogram_privatized(img, hist, N_PIXELS, T);
            }

            times[r] = (double)(now_ns() - t0) / N_IMAGES;
        }

        double m = mean(times, REPEATS);
        double s = stddev(times, REPEATS);

        if (T == 1)
            baseline = m;

        double speedup = baseline / m;
        double bw = (double)N_PIXELS / (m * 1e-9) / 1e9;

        printf("%-8d  %12.0f  %12.0f  %12.2fx  %12.2f\n",
            T,
            m,
            s,
            speedup,
            bw);

        if (!hist_equal(hist, href))
            printf("ERROR: incorrect histogram\n");
    }

    free(img); free(href); free(hist); free(times);
    return 0;
}

//doing across 100 images reduces effect of noise and overhead in time measurement

//random image:
/*threads       mean(ns)       std(ns)       speedup        eff BW
----------------------------------------------------------------------
1               606290         12657          1.00x          3.42
2               385849          1181          1.57x          5.37
4               369134          6553          1.64x          5.62
6               249818          9817          2.43x          8.30
8               213026          7135          2.85x          9.73
10              175646          2301          3.45x         11.81
12              181438         55653          3.34x         11.43 */

//single bin image:
/*threads       mean(ns)       std(ns)       speedup        eff BW
----------------------------------------------------------------------
1              3315215         24114          1.00x          0.63
2              1879317         14295          1.76x          1.10
4               932218          2094          3.56x          2.22
6               621535          3102          5.33x          3.34
8               468511          4587          7.08x          4.43
10              451742         18448          7.34x          4.59
12              418982         64438          7.91x          4.95 */
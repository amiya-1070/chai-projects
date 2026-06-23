#define _GNU_SOURCE
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>



void sobel_opencv(
    const uint8_t* img,
    uint8_t* out,
    int rows,
    int cols);

#define M          8192
#define N          8192
#define N_PIXELS   (M * N)
#define REPEATS    50
#define PEAK_BW    32.0
#define PREFETCH_DIST 8

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

/* ── version 1: naive scalar ─────────────────────────────────────────────────
   computes Gx and Gy using the full 3×3 kernel loops
   magnitude = sqrt(Gx^2 + Gy^2), clamped to [0,255]
   boundary: clamp (replicate border pixels)                                  */
static void sobel_naive(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    int m, int n)
{
    /* Gx kernel:  [-1  0 +1]     Gy kernel:  [-1 -2 -1]
                   [-2  0 +2]                 [ 0  0  0]
                   [-1  0 +1]                 [+1 +2 +1] */
    static const int kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int ky[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; dy++) {
                int ii = i + dy;
                if (ii < 0)  ii = 0;
                if (ii >= m) ii = m - 1;
                for (int dx = -1; dx <= 1; dx++) {
                    int jj = j + dx;
                    if (jj < 0)  jj = 0;
                    if (jj >= n) jj = n - 1;
                    int p = img[ii * n + jj];
                    gx += kx[dy+1][dx+1] * p;
                    gy += ky[dy+1][dx+1] * p;
                }
            }
            int mag = abs(gx) + abs(gy);
            if (mag > 255) mag = 255;
            out[i * n + j] = mag;
        }
    }
}

/* ── version 2: optimized scalar + OpenMP ────────────────────────────────────
   key optimizations over naive:
   1. exploit zero-column of Gx — skip 3 multiplications per pixel
   2. sliding 3-row window — rows i-1, i, i+1 loaded once per output row
   3. precompute row pointers to avoid repeated index arithmetic
   4. integer arithmetic throughout (inputs are uint8, intermediates fit int32)
   5. parallelize over rows with OpenMP                                        */
static void sobel_scalar_omp(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    int m, int n, int T)
{
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int i = 0; i < m; i++) {
        /* clamp row indices */
        int i0 = (i > 0)     ? i - 1 : 0;
        int i1 = i;
        int i2 = (i < m - 1) ? i + 1 : m - 1;

        /* row pointers — these rows stay in L1 for the entire inner loop
           row length = 1920 bytes = 30 cache lines, fits in 48KB L1        */
        const uint8_t* r0 = img + i0 * n;
        const uint8_t* r1 = img + i1 * n;
        const uint8_t* r2 = img + i2 * n;

        uint8_t* dst = out + i * n;

        for (int j = 0; j < n; j++) {
            int jl = (j > 0)     ? j - 1 : 0;
            int jr = (j < n - 1) ? j + 1 : n - 1;

            /* Gx = [-1 0 +1; -2 0 +2; -1 0 +1] × neighborhood
               center column is zero — 6 operations instead of 9           */
            int gx = -r0[jl] + r0[jr]
                     - 2*r1[jl] + 2*r1[jr]
                     - r2[jl] + r2[jr];

            /* Gy = [-1 -2 -1; 0 0 0; +1 +2 +1] × neighborhood
               middle row is zero — 6 operations instead of 9              */
            int gy = -r0[jl] - 2*r0[j] - r0[jr]
                     + r2[jl] + 2*r2[j] + r2[jr];

            int mag = abs(gx) + abs(gy);
            if (mag > 255) mag = 255;
            dst[j] = (uint8_t)mag;
        }
    }
}

/* ── version 3: AVX2 + OpenMP ────────────────────────────────────────────────*/
__attribute__((target("avx2")))
static void sobel_avx2_omp(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    int m, int n, int T)
{
    
    int p_w = 160, e_w = 100;
    int total_w = 2 * p_w + (T - 2) * e_w;
    
    int *starts = malloc(T * sizeof(int));
    int *ends   = malloc(T * sizeof(int));
    int row = 0;
    for (int t = 0; t < T; t++) {
        starts[t]  = row;
        int w      = (t < 2) ? p_w : e_w;
        int nrows  = (w * m + total_w / 2) / total_w;
        ends[t]    = starts[t] + nrows;
        row        = ends[t];
    }
    ends[T - 1] = m;

    //NEW    
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int i_start = starts[tid];
        int i_end = ends[tid];

        for (int i = i_start; i < i_end; i++) {
 
            int i0 = (i > 0)     ? i - 1 : 0;
            int i1 = i;
            int i2 = (i < m - 1) ? i + 1 : m - 1;

            const uint8_t* r0 = img + i0 * n;
            const uint8_t* r1 = img + i1 * n;
            const uint8_t* r2 = img + i2 * n;
            uint8_t*       dst = out + i * n;

            
            {
                int gx = -r0[0] + r0[1] - 2*r1[0] + 2*r1[1] - r2[0] + r2[1];
                int gy = -r0[0] - 2*r0[0] - r0[1] + r2[0] + 2*r2[0] + r2[1];
                int mag = (int)sqrtf((float)gx*gx + (float)gy*gy);
                if (mag > 255) mag = 255;
                dst[0] = (uint8_t)mag;
            }

            int j = 1;
            __m256i vmax255 = _mm256_set1_epi16(255);
        

            for (; j <= n - 31; j += 16) {

                __m128i r0a = _mm_loadu_si128((const __m128i*)(r0 + j - 1));
                __m128i r0b = _mm_loadu_si128((const __m128i*)(r0 + j + 15));

                __m128i r0_left8  = r0a;
                __m128i r0_mid8   = _mm_alignr_epi8(r0b, r0a, 1);
                __m128i r0_right8 = _mm_alignr_epi8(r0b, r0a, 2);
                
                __m128i r1a = _mm_loadu_si128((const __m128i*)(r1 + j - 1));
                __m128i r1b = _mm_loadu_si128((const __m128i*)(r1 + j + 15));

                __m128i r1_left8  = r1a;
                __m128i r1_right8 = _mm_alignr_epi8(r1b, r1a, 2);

                __m128i r2a = _mm_loadu_si128((const __m128i*)(r2 + j - 1));
                __m128i r2b = _mm_loadu_si128((const __m128i*)(r2 + j + 15));

                __m128i r2_left8  = r2a;
                __m128i r2_mid8   = _mm_alignr_epi8(r2b, r2a, 1);
                __m128i r2_right8 = _mm_alignr_epi8(r2b, r2a, 2);

                

                __m256i a0l = _mm256_cvtepu8_epi16(r0_left8);
                __m256i a0m = _mm256_cvtepu8_epi16(r0_mid8);
                __m256i a0r = _mm256_cvtepu8_epi16(r0_right8);

                __m256i a1l = _mm256_cvtepu8_epi16(r1_left8);
                __m256i a1r = _mm256_cvtepu8_epi16(r1_right8);

                __m256i a2l = _mm256_cvtepu8_epi16(r2_left8);
                __m256i a2m = _mm256_cvtepu8_epi16(r2_mid8);
                __m256i a2r = _mm256_cvtepu8_epi16(r2_right8);

            

                // gx: independent halves
                __m256i gx_top = _mm256_sub_epi16(a0r, a0l);          // row0 contribution
                __m256i gx_bot = _mm256_sub_epi16(a2r, a2l);          // row2 contribution
                __m256i gx_mid = _mm256_slli_epi16(_mm256_sub_epi16(a1r, a1l), 1);   // row1 contribution
                __m256i gx = _mm256_add_epi16(_mm256_add_epi16(gx_top, gx_bot),gx_mid);

                // same for gy
                __m256i gy_top = _mm256_sub_epi16(a2l, a0l);
                __m256i gy_bot = _mm256_sub_epi16(a2r, a0r);
                __m256i gy_mid = _mm256_slli_epi16(_mm256_sub_epi16(a2m, a0m), 1);
                __m256i gy = _mm256_add_epi16(_mm256_add_epi16(gy_top, gy_bot),gy_mid);



                __m256i abs_gx = _mm256_abs_epi16(gx);
                __m256i abs_gy = _mm256_abs_epi16(gy);

                __m256i mag16 = _mm256_add_epi16(abs_gx, abs_gy);

                
                mag16 = _mm256_min_epu16(mag16, vmax255);

                

                __m128i lo128 = _mm256_castsi256_si128(mag16);
                __m128i hi128 = _mm256_extracti128_si256(mag16, 1);

                __m128i mag8 = _mm_packus_epi16(lo128, hi128);

                _mm_storeu_si128((__m128i*)(dst + j), mag8);
            }


            
            for (; j < n; j++) {
                int jl = j - 1;
                int jr = (j < n - 1) ? j + 1 : n - 1;
                int gx = -r0[jl] + r0[jr] - 2*r1[jl] + 2*r1[jr]
                         - r2[jl] + r2[jr];
                int gy = -r0[jl] - 2*r0[j] - r0[jr]
                         + r2[jl] + 2*r2[j] + r2[jr];
                int mag = abs(gx) + abs(gy);
                if (mag > 255) mag = 255;
                dst[j] = (uint8_t)mag;
            }
        }
        
    }
    free(starts);
    free(ends);
}


/* ── correctness ─────────────────────────────────────────────────────────── */
static int max_diff(const uint8_t* a, const uint8_t* b, int n) {
    int mx = 0;
    for (int i = 0; i < n; i++) {
        int d = abs((int)a[i] - (int)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double dmean(const double* a, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s/n;
}
static double dstd(const double* a, int n) {
    double m = dmean(a,n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s/n);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    int T = omp_get_max_threads();
    printf("Logical CPUs: %d\n\n", T);

    #pragma omp parallel num_threads(T)
    { int x = omp_get_thread_num(); (void)x; }


    int w, h, c;
    uint8_t *img = stbi_load("/media/amiyaun/New Volume/cv algos/rain-forest-tree-view-up_jpg.png",&w,&h,&c,1);      // force grayscale
    uint8_t *out_avx = aligned_alloc(64, w * h);

    size_t pixels = (size_t)w * h;

    uint8_t* out_ref = aligned_alloc(64, pixels);
    uint8_t* out     = aligned_alloc(64, pixels);
    double*  times   = malloc(REPEATS * sizeof(double));


    /* reference */
    
    sobel_naive(img, out_ref, h, w);
    /* correctness */
    sobel_scalar_omp(img, out, h, w, T);
    printf("scalar_omp    maxdiff: %d\n", max_diff(out, out_ref, pixels));

    sobel_avx2_omp(img, out_avx, h, w, T);
    printf("avx2_omp      maxdiff: %d\n", max_diff(out, out_ref, pixels));


    /* timing table */
    printf("\n%-20s  %10s  %9s  %10s  %10s\n",
        "version", "mean(ns)", "std(ns)", "speedup", "BW(GB/s)");
    for (int i = 0; i < 70; i++) putchar('-');
    putchar('\n');

    double naive_mean = 0.0;

    /* naive */
    {
        sobel_naive(img, out, h, w);
        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            sobel_naive(img, out, h, w);
            times[r] = (double)(now_ns() - t0);
        }
        naive_mean = dmean(times, REPEATS);
        /* bytes: 9 reads per pixel (3×3) + 1 write */
        double bw = (double)(pixels * 10) / (naive_mean * 1e-9) / 1e9;
        printf("%-20s  %10.0f  %9.0f  %10.2f  %10.2f\n",
            "naive", naive_mean, dstd(times, REPEATS), 1.0, bw);
    }

    /* scalar_omp */
    {
        sobel_scalar_omp(img, out, h, w, T);
        sobel_scalar_omp(img, out, h, w, T);
        sobel_scalar_omp(img, out, h, w, T);
        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            sobel_scalar_omp(img, out, h, w, T);
            times[r] = (double)(now_ns() - t0);
        }
        double m  = dmean(times, REPEATS);
        double bw = (double)(pixels * 10) / (m * 1e-9) / 1e9;
        printf("%-20s  %10.0f  %9.0f  %10.2f  %10.2f\n",
            "scalar_omp", m, dstd(times, REPEATS), naive_mean/m, bw);
    }

    /* avx2_omp */
    {
        sobel_avx2_omp(img, out, h, w, T);
        sobel_avx2_omp(img, out, h, w, T);
        sobel_avx2_omp(img, out, h, w, T);
        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            sobel_avx2_omp(img, out_avx, h, w, T);
            times[r] = (double)(now_ns() - t0);
        }
        double m  = dmean(times, REPEATS);
        double bw = (double)(pixels * 10) / (m * 1e-9) / 1e9;
        printf("%-20s  %10.0f  %9.0f  %10.2f  %10.2f\n",
            "avx2_omp", m, dstd(times, REPEATS), naive_mean/m, bw);
    }

    
    sobel_avx2_omp(img, out_avx, h, w, T);

    #pragma omp parallel
    {
        printf("thread %d cpu %d\n",
            omp_get_thread_num(),
            sched_getcpu());
    }

    /* profiling loop — avx2_omp is the primary target */
    printf("\nrunning profiling loop (1000 × avx2_omp)...\n");
    for (int i = 0; i < 10; i++)
        sobel_avx2_omp(img, out, h, w, T);
    for (int i = 0; i < 1000; i++)
        sobel_avx2_omp(img, out, h, w, T);

    volatile uint64_t checksum = 0;
    for (int i = 0; i < pixels; i++) checksum += out[i];
    printf("checksum = %lu\n", checksum);


    stbi_image_free(img);
    free(out_ref);
    free(out);
    free(out_avx);
    free(times);
    return 0;
}
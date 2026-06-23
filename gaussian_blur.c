#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define M        1080
#define N        1920
#define N_PIXELS (M * N)
#define RADIUS   2
#define KSIZE    (2 * RADIUS + 1)   /* 5 */
#define SIGMA    1.0f
#define REPEATS  50
#define PEAK_BW  32.0

/* ── timing ─────────────────────────────────────────────────────────────── */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ── image generation ────────────────────────────────────────────────────── */
static void gen_image(uint8_t* img, int n) {
    uint64_t state = 107    ;
    for (int i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        img[i] = (uint8_t)(state >> 56);
    }
}

/* ── kernel precomputation ───────────────────────────────────────────────── */
/* builds a normalized 1D Gaussian kernel of length KSIZE
   kernel[k] = exp(-(k-RADIUS)^2 / (2*sigma^2)), then normalized to sum=1 */
static void build_kernel(float* kernel, float sigma) {
    float sum = 0.0f;
    for (int k = 0; k < KSIZE; k++) {
        float x     = (float)(k - RADIUS);
        kernel[k]   = expf(-(x * x) / (2.0f * sigma * sigma));
        sum        += kernel[k];
    }
    for (int k = 0; k < KSIZE; k++)
        kernel[k] /= sum;
}

/* ── version 1: naive 2D convolution (baseline) ─────────────────────────── */
/* no separability, no SIMD, no parallelism
   computes 2D kernel directly: O((2r+1)^2) per pixel
   every output pixel reads a (2r+1)×(2r+1) neighborhood                   */
static void gaussian_naive(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    const float*   __restrict__ kernel2d,   /* (2r+1)×(2r+1) flattened */
    int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                int ii = i + dy;
                if (ii < 0) ii = 0;
                if (ii >= m) ii = m - 1;
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int jj = j + dx;
                    if (jj < 0) jj = 0;
                    if (jj >= n) jj = n - 1;
                    int kid = (dy + RADIUS) * KSIZE + (dx + RADIUS);
                    acc += kernel2d[kid] * (float)img[ii * n + jj];
                }
            }
            out[i * n + j] = (uint8_t)(acc + 0.5f);
        }
    }
}

/* ── version 2: separable scalar ────────────────────────────────────────── */
/* splits 2D convolution into horizontal pass then vertical pass
   each pass is a 1D convolution with the same 1D kernel
   intermediate buffer tmp holds float32 results of horizontal pass          */
static void gaussian_separable_scalar(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    float*         __restrict__ tmp,        /* M×N float32 scratch */
    const float*   __restrict__ kernel,     /* 1D, length KSIZE */
    int m, int n)
{
    /* ── pass 1: horizontal ── */
    /* for each pixel (i,j): tmp[i,j] = sum_{k=0}^{KSIZE-1} kernel[k] * img[i, clamp(j+k-r, 0, n-1)] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            /* manual unroll for KSIZE=5 */
            for (int k = 0; k < KSIZE; k++) {
                int jj = j + k - RADIUS;
                if (jj < 0) jj = 0;
                if (jj >= n) jj = n - 1;
                acc += kernel[k] * (float)img[i * n + jj];
            }
            tmp[i * n + j] = acc;
        }
    }

    /* ── pass 2: vertical ── */
    /* for each pixel (i,j): out[i,j] = sum_{k=0}^{KSIZE-1} kernel[k] * tmp[clamp(i+k-r,0,m-1), j] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < KSIZE; k++) {
                int ii = i + k - RADIUS;
                if (ii < 0) ii = 0;
                if (ii >= m) ii = m - 1;
                acc += kernel[k] * tmp[ii * n + j];
            }
            float val = acc + 0.5f;
            if (val < 0.0f)   val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            out[i * n + j] = (uint8_t)val;
        }
    }
}

/* ── version 3: separable + OpenMP ──────────────────────────────────────── */
/* parallelizes both passes over rows using OpenMP
   horizontal pass: rows are independent → trivially parallel
   vertical pass: reads from tmp (read-only after horizontal) → also parallel
   each thread processes a contiguous chunk of rows                          */
static void gaussian_separable_omp(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    float*         __restrict__ tmp,
    const float*   __restrict__ kernel,
    int m, int n, int T)
{
    /* pass 1: horizontal — each row independent */
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < KSIZE; k++) {
                int jj = j + k - RADIUS;
                if (jj < 0) jj = 0;
                if (jj >= n) jj = n - 1;
                acc += kernel[k] * (float)img[i * n + jj];
            }
            tmp[i * n + j] = acc;
        }
    }
    /* implicit barrier here — pass 2 must not start until all of tmp is written */

    /* pass 2: vertical — each row independent (reads tmp, writes out) */
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < KSIZE; k++) {
                int ii = i + k - RADIUS;
                if (ii < 0) ii = 0;
                if (ii >= m) ii = m - 1;
                acc += kernel[k] * tmp[ii * n + j];
            }
            float val = acc + 0.5f;
            if (val < 0.0f)   val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            out[i * n + j] = (uint8_t)val;
        }
    }
}

/* ── version 4: separable + OpenMP + AVX2 ───────────────────────────────── */
/* vectorizes both passes using AVX2 256-bit registers
   horizontal pass: processes 8 float32 output pixels per iteration
     load 8 consecutive img bytes → convert to float32 → FMA with kernel weight
     for each of the 5 kernel taps, shift the load window by 1 position
   vertical pass: processes 8 consecutive float32 values per iteration
     for each of 5 rows, load 8 float32s → FMA with kernel weight            */
__attribute__((target("avx2,fma")))
static void gaussian_separable_avx2(
    const float*   __restrict__ imgf,
    uint8_t*       __restrict__ out,
    float*         __restrict__ tmp,
    const float*   __restrict__ kernel,
    int m, int n, int T)
{
    /* ── pass 1: horizontal AVX2 ── */
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int i = 0; i < m; i++) {

        const float* row = imgf + i * n;
        float*       dst = tmp  + i * n;

        __m256 k0 = _mm256_set1_ps(kernel[0]);
        __m256 k1 = _mm256_set1_ps(kernel[1]);
        __m256 k2 = _mm256_set1_ps(kernel[2]);
        __m256 k3 = _mm256_set1_ps(kernel[3]);
        __m256 k4 = _mm256_set1_ps(kernel[4]);

        int j = RADIUS;

        for (; j <= n - RADIUS - 8; j += 8) {

            __m256 p0 = _mm256_loadu_ps(row + j - 2);
            __m256 p1 = _mm256_loadu_ps(row + j - 1);
            __m256 p2 = _mm256_loadu_ps(row + j);
            __m256 p3 = _mm256_loadu_ps(row + j + 1);
            __m256 p4 = _mm256_loadu_ps(row + j + 2);
            
            __m256 acc0 = _mm256_mul_ps(k0, p0);
            __m256 acc1 = _mm256_mul_ps(k1, p1);
            acc0 = _mm256_fmadd_ps(k2, p2, acc0);
            acc1 = _mm256_fmadd_ps(k3, p3, acc1);
            acc0 = _mm256_fmadd_ps(k4, p4, acc0);
            __m256 acc  = _mm256_add_ps(acc0, acc1);


            _mm256_storeu_ps(dst + j, acc);
        }

        /* left border */
        for (int jj = 0; jj < RADIUS; jj++) {
            float a = 0.0f;

            for (int k = 0; k < KSIZE; k++) {
                int jc = jj + k - RADIUS;
                if (jc < 0) jc = 0;
                a += kernel[k] * row[jc];
            }

            dst[jj] = a;
        }

        /* right border + tail */
        for (; j < n; j++) {
            float a = 0.0f;

            for (int k = 0; k < KSIZE; k++) {
                int jc = j + k - RADIUS;
                if (jc >= n) jc = n - 1;
                a += kernel[k] * row[jc];
            }

            dst[j] = a;
        }
    }

    /* ── pass 2: vertical AVX2 ── */
    for (int i = 0; i < m; i++) {
        uint8_t*  dstu8 = out + i * n;

        /* preload kernel weights */
        __m256 k0 = _mm256_set1_ps(kernel[0]);
        __m256 k1 = _mm256_set1_ps(kernel[1]);
        __m256 k2 = _mm256_set1_ps(kernel[2]);
        __m256 k3 = _mm256_set1_ps(kernel[3]);
        __m256 k4 = _mm256_set1_ps(kernel[4]);

        /* clamp row indices for the 5 taps */
        int ii0 = i - 2; if (ii0 < 0) ii0 = 0;
        int ii1 = i - 1; if (ii1 < 0) ii1 = 0;
        int ii2 = i;
        int ii3 = i + 1; if (ii3 >= m) ii3 = m - 1;
        int ii4 = i + 2; if (ii4 >= m) ii4 = m - 1;

        /* pointers to the 5 source rows in tmp */
        const float* r0 = tmp + ii0 * n;
        const float* r1 = tmp + ii1 * n;
        const float* r2 = tmp + ii2 * n;
        const float* r3 = tmp + ii3 * n;
        const float* r4 = tmp + ii4 * n;

        /* constants for clamp [0, 255] and rounding */
        __m256 vhalf  = _mm256_set1_ps(0.5f);
        __m256 vzero  = _mm256_setzero_ps();
        __m256 v255   = _mm256_set1_ps(255.0f);

        int j = 0;
        for (; j <= n - 8; j += 8) {
            /* load 8 float32s from each of the 5 source rows */
            __m256 p0 = _mm256_load_ps(r0 + j);
            __m256 p1 = _mm256_load_ps(r1 + j);
            __m256 p2 = _mm256_load_ps(r2 + j);
            __m256 p3 = _mm256_load_ps(r3 + j);
            __m256 p4 = _mm256_load_ps(r4 + j);

            /* FMA accumulation */
            __m256 acc0 = _mm256_mul_ps(k0, p0);
            __m256 acc1 = _mm256_mul_ps(k1, p1);
            acc0 = _mm256_fmadd_ps(k2, p2, acc0);
            acc1 = _mm256_fmadd_ps(k3, p3, acc1);
            acc0 = _mm256_fmadd_ps(k4, p4, acc0);
            __m256 acc  = _mm256_add_ps(acc0, acc1);

            /* round, clamp, convert float32 → uint8 */
            acc = _mm256_add_ps(acc, vhalf);
            acc = _mm256_max_ps(acc, vzero);
            acc = _mm256_min_ps(acc, v255);

            /* float32 → int32 (truncation after rounding above) */
            __m256i ival = _mm256_cvttps_epi32(acc);

            /* pack int32 → int16 → uint8
               _mm256_packs_epi32: int32×8 → int16×8 (×2 lanes)
               _mm256_packus_epi16: int16×16 → uint8×16 (×2 lanes)
               then extract lower 8 bytes                                    */
            __m128i lo   = _mm256_castsi256_si128(ival);
            __m128i hi   = _mm256_extracti128_si256(ival, 1);
            

            /* AVX2 pack operates within 128-bit lanes independently
               result layout in i8: [b0..b7, b0..b7, b0..b7, b0..b7]
               (the 8 valid bytes are duplicated across 4 positions)
               extract the low 64 bits = 8 bytes we want                    */
            __m128i i16  = _mm_packs_epi32(lo, hi);
            __m128i i8   = _mm_packus_epi16(i16, i16);
            _mm_storel_epi64((__m128i*)(dstu8 + j), i8);
           
        }

        /* scalar tail */
        for (; j < n; j++) {
            float acc = kernel[0]*r0[j] + kernel[1]*r1[j] + kernel[2]*r2[j]
                      + kernel[3]*r3[j] + kernel[4]*r4[j];
            float val = acc + 0.5f;
            if (val < 0.0f)   val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            dstu8[j] = (uint8_t)val;
        }
    }
    //#pragma omp parallel for num_threads(T) schedule(static)
    
}

__attribute__((target("avx2,fma")))
static void gaussian_separable_balanced(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    float*         __restrict__ tmp,
    const float*   __restrict__ kernel,
    int m, int n, int T)
{
    /* chunk assignment — same integer weight calculation */
    int p_w = 160, e_w = 100;
    int total_w = 2 * p_w + (T - 2) * e_w;
    int starts[12], ends[12];
    int row = 0;
    for (int t = 0; t < T; t++) {
        starts[t] = row;
        int w     = (t < 2) ? p_w : e_w;
        int nrows = (w * m + total_w / 2) / total_w;
        ends[t]   = starts[t] + nrows;
        row       = ends[t];
    }
    ends[T - 1] = m;

    /* horizontal pass — AVX2, balanced chunks */
    #pragma omp parallel num_threads(T)
    {
        int tid     = omp_get_thread_num();
        int i_start = starts[tid];
        int i_end   = ends[tid];

        __m256 k0 = _mm256_set1_ps(kernel[0]);
        __m256 k1 = _mm256_set1_ps(kernel[1]);
        __m256 k2 = _mm256_set1_ps(kernel[2]);
        __m256 k3 = _mm256_set1_ps(kernel[3]);
        __m256 k4 = _mm256_set1_ps(kernel[4]);

        /* ── horizontal pass ── */
        for (int i = i_start; i < i_end; i++) {
            const uint8_t* row_ptr = img + i * n;
            float*         dst     = tmp + i * n;

            for (int jj = 0; jj < RADIUS; jj++) {
                float a = 0.0f;
                for (int k = 0; k < KSIZE; k++) {
                    int jc = jj + k - RADIUS;
                    if (jc < 0) jc = 0;
                    a += kernel[k] * (float)row_ptr[jc];
                }
                dst[jj] = a;
            }

            int j = RADIUS;
            for (; j <= n - RADIUS - 8; j +=8) {
                __m128i block = _mm_loadu_si128(
                                    (const __m128i*)(row_ptr + j - RADIUS));
                __m256 p0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(block));
                __m256 p1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block,1)));
                __m256 p2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block,2)));
                __m256 p3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block,3)));
                __m256 p4 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block,4)));

                __m256 acc0 = _mm256_mul_ps(k0, p0);
                __m256 acc1 = _mm256_mul_ps(k1, p1);
                acc0 = _mm256_fmadd_ps(k2, p2, acc0);
                acc1 = _mm256_fmadd_ps(k3, p3, acc1);
                acc0 = _mm256_fmadd_ps(k4, p4, acc0);
                __m256 acc = _mm256_add_ps(acc0, acc1);
                _mm256_storeu_ps(dst + j, acc);
            }

            for (; j < n; j++) {
                float a = 0.0f;
                for (int k = 0; k < KSIZE; k++) {
                    int jc = j + k - RADIUS;
                    if (jc >= n) jc = n - 1;
                    a += kernel[k] * (float)row_ptr[jc];
                }
                dst[j] = a;
            }
        }

        /* barrier — all horizontal writes must complete before any vertical read */
        #pragma omp barrier

        /* ── vertical pass ── */
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vzero = _mm256_setzero_ps();
        __m256 v255  = _mm256_set1_ps(255.0f);

        for (int i = i_start; i < i_end; i++) {
            uint8_t* dstu8 = out + i * n;

            int ii0 = i-2; if (ii0 < 0)  ii0 = 0;
            int ii1 = i-1; if (ii1 < 0)  ii1 = 0;
            int ii2 = i;
            int ii3 = i+1; if (ii3 >= m) ii3 = m-1;
            int ii4 = i+2; if (ii4 >= m) ii4 = m-1;

            const float* r0 = tmp + ii0 * n;
            const float* r1 = tmp + ii1 * n;
            const float* r2 = tmp + ii2 * n;
            const float* r3 = tmp + ii3 * n;
            const float* r4 = tmp + ii4 * n;

            int j = 0;
            for (; j <= n - 8; j += 8) {
                __m256 p0 = _mm256_loadu_ps(r0 + j);
                __m256 p1 = _mm256_loadu_ps(r1 + j);
                __m256 p2 = _mm256_loadu_ps(r2 + j);
                __m256 p3 = _mm256_loadu_ps(r3 + j);
                __m256 p4 = _mm256_loadu_ps(r4 + j);

                __m256 acc0 = _mm256_mul_ps(k0, p0);
                __m256 acc1 = _mm256_mul_ps(k1, p1);
                acc0 = _mm256_fmadd_ps(k2, p2, acc0);
                acc1 = _mm256_fmadd_ps(k3, p3, acc1);
                acc0 = _mm256_fmadd_ps(k4, p4, acc0);
                __m256 acc = _mm256_add_ps(acc0, acc1);

                acc = _mm256_add_ps(acc, vhalf);
                acc = _mm256_max_ps(acc, vzero);
                acc = _mm256_min_ps(acc, v255);

                __m256i ival = _mm256_cvttps_epi32(acc);
                __m128i lo   = _mm256_castsi256_si128(ival);
                __m128i hi   = _mm256_extracti128_si256(ival, 1);
                __m128i i16  = _mm_packs_epi32(lo, hi);
                __m128i i8   = _mm_packus_epi16(i16, i16);
                _mm_storel_epi64((__m128i*)(dstu8 + j), i8);
            }

            for (; j < n; j++) {
                float acc = kernel[0]*r0[j] + kernel[1]*r1[j]
                        + kernel[2]*r2[j] + kernel[3]*r3[j]
                        + kernel[4]*r4[j];
                float val = acc + 0.5f;
                if (val < 0.0f)   val = 0.0f;
                if (val > 255.0f) val = 255.0f;
                dstu8[j] = (uint8_t)val;
            }
        }

    } /* end parallel region */
}

/* ── correctness check ───────────────────────────────────────────────────── */
/* returns max absolute difference between two output images
   Gaussian blur results should match to within ±1 due to float rounding    */
static int max_diff(const uint8_t* a, const uint8_t* b, int n) {
    int mx = 0;
    for (int i = 0; i < n; i++) {
        int d = abs((int)a[i] - (int)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ── benchmark helper ────────────────────────────────────────────────────── */
static double dmean(const double* a, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s/n;
}
static double dstd(const double* a, int n) {
    double m = dmean(a,n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s/n);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    int T = omp_get_max_threads();

    #pragma omp parallel num_threads(T)
    { int x = omp_get_thread_num(); (void)x; }

    uint8_t* img     = aligned_alloc(64, N_PIXELS);
    uint8_t* out     = aligned_alloc(64, N_PIXELS);
    uint8_t* out_ref = aligned_alloc(64, N_PIXELS);
    float*   tmp     = aligned_alloc(64, N_PIXELS * sizeof(float));
    float*   imgf    = aligned_alloc(64, N_PIXELS * sizeof(float));

    float kernel[KSIZE];
    build_kernel(kernel, SIGMA);

    float kernel2d[KSIZE * KSIZE];
    for (int dy = 0; dy < KSIZE; dy++)
        for (int dx = 0; dx < KSIZE; dx++)
            kernel2d[dy * KSIZE + dx] = kernel[dy] * kernel[dx];

    gen_image(img, N_PIXELS);

    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N_PIXELS; i++)
        imgf[i] = (float)img[i];

    /* reference output */
    gaussian_naive(img, out_ref, kernel2d, M, N);
    size_t num_diff = 0;
    /* correctness check — one call each, not timed */
    gaussian_separable_scalar(img, out, tmp, kernel, M, N);
    for (int i = 0; i < N_PIXELS; i++) {
        if (out[i] != out_ref[i])
            num_diff++;
    }
    printf("scalar     maxdiff = %d\n",
       max_diff(out, out_ref, N_PIXELS));

    printf("scalar     mismatches = %zu\n", num_diff);

    gaussian_separable_omp(img, out, tmp, kernel, M, N, T);
    num_diff = 0;
    for (int i = 0; i < N_PIXELS; i++) {
        if (out[i] != out_ref[i])
            num_diff++;
    }

    printf("separable_omp     maxdiff = %d\n",
       max_diff(out, out_ref, N_PIXELS));

    printf("separable_omp     mismatches = %zu\n", num_diff);

    gaussian_separable_avx2(imgf, out, tmp, kernel, M, N, T);
    num_diff = 0;
    for (int i = 0; i < N_PIXELS; i++) {
        if (out[i] != out_ref[i])
            num_diff++;
    }

    printf("separable_avx2_omp     maxdiff = %d\n",
       max_diff(out, out_ref, N_PIXELS));

    printf("separable_avx2_omp     mismatches = %zu\n", num_diff);
    

    gaussian_separable_balanced(img, out, tmp, kernel, M, N, T);
    num_diff = 0;
    for (int i = 0; i < N_PIXELS; i++) {
        if (out[i] != out_ref[i])
            num_diff++;
    }
    printf("separable_balanced     maxdiff = %d\n",
       max_diff(out, out_ref, N_PIXELS));

    printf("separable_balanced     mismatches = %zu\n", num_diff);
    

    /* warmup */
    for (int i = 0; i < 10; i++)
        gaussian_separable_balanced(img, out, tmp, kernel, M, N, T);

    /* profiling loop */
    for (int i = 0; i < 1000; i++)
        gaussian_separable_balanced(img, out, tmp, kernel, M, N, T);

    volatile uint64_t checksum = 0;
    for (int i = 0; i < N_PIXELS; i++)
        checksum += out[i];
    printf("checksum = %lu\n", checksum);

    free(img); free(out); free(out_ref); free(tmp); free(imgf);
    return 0;
}



//sudo sysctl kernel.perf_event_paranoid=-1
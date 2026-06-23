//sobel_min.c

#define _GNU_SOURCE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "sobel_min.h"
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define MAX_THREADS 64

int partition_start[MAX_THREADS];
int partition_end[MAX_THREADS];
#define M          8192
#define N          8192
#define N_PIXELS   (M * N)
#define REPEATS    50
#define PEAK_BW    32.0
#define PREFETCH_DIST 8


static CThreadInfo thread_info[MAX_THREADS];



/* ── version 3: AVX2 + OpenMP ────────────────────────────────────────────────*/
__attribute__((target("avx2")))
void sobel_avx2_omp(
    const uint8_t* __restrict__ img,
    uint8_t*       __restrict__ out,
    int m, int n, int T, double pe_ratio)
{
    int e_w = 100;
    int p_w = (int)(e_w * pe_ratio + 0.5);    // rounds to nearest int
    int total_w = 2 * p_w + (T - 2) * e_w;
    
    int *starts = malloc(T * sizeof(int));
    int *ends   = malloc(T * sizeof(int));
    int row = 0;
    for (int t = 0; t < T; t++) {
        starts[t]  = row;
        int w      = (t < 2) ? p_w : e_w;
        int nrows  = (w * m + total_w / 2) / total_w;
        ends[t]    = starts[t] + nrows;
        partition_start[t] = starts[t];
        partition_end[t]   = ends[t];
        row        = ends[t];
    }
    ends[T - 1] = m;
    partition_end[T - 1] = m;

    //NEW    
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
    
        
        int i_start = starts[tid];
        int i_end = ends[tid];

        thread_info[tid].tid = tid;
        thread_info[tid].cpu = sched_getcpu();
        thread_info[tid].row_start = i_start;
        thread_info[tid].row_end = i_end;


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
}

const CThreadInfo* get_thread_info(void)
{
    return thread_info;
}

int get_thread_count(void)
{
    return omp_get_max_threads();
}
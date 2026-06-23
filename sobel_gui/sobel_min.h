// sobel_min.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void sobel_avx2_omp(const uint8_t* img, uint8_t* out, int m, int n, int T);

// expose the actual partition data filled by sobel_avx2_omp
extern int partition_start[64];
extern int partition_end[64];

typedef struct { int tid; int cpu; int row_start; int row_end; } CThreadInfo;
const CThreadInfo* get_thread_info(void);

int get_thread_count(void);

#ifdef __cplusplus
}
#endif
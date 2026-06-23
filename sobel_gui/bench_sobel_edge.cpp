//bench_sobel_edge.cpp

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <sstream>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define M        8192
#define N        8192
#define N_PIXELS (M * N)
#define REPEATS  50
#define PEAK_BW  32.0

static inline uint64_t now_ns(void) {
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t)(count.QuadPart * 1000000000LL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

static void gen_image(uint8_t* img, int n) {
    uint64_t state = 42;
    for (int i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        img[i] = (uint8_t)(state >> 56);
    }
}

static double dmean(const double* a, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s/n;
}
static double dstd(const double* a, int n) {
    double m = dmean(a,n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s/n);
}

int main(void) {
    int w, h, c;
    uint8_t *img_raw = stbi_load("/media/amiyaun/New Volume/cv algos/rain-forest-tree-view-up_jpg.png", &w, &h, &c, 1);
    cv::Mat img(h, w, CV_8UC1, img_raw);

    size_t pixels = (size_t)w * h;

    double times[REPEATS];

    /* ── method 1: cv::Sobel with CV_16S output, then convertScaleAbs ──────
       This is the standard OpenCV Sobel usage:
       - compute Gx and Gy separately as int16
       - convertScaleAbs: abs + scale + saturate_cast to uint8
       - addWeighted: blend Gx and Gy magnitudes
       This is closest to what we're computing: |Gx| + |Gy| approximation  */
    {
        cv::Mat gx, gy, abs_gx, abs_gy, dst;

        /* warmup */
        for (int i = 0; i < 3; i++) {
            cv::Sobel(img, gx, CV_16S, 1, 0, 3);  /* dx=1, dy=0, ksize=3 */
            cv::Sobel(img, gy, CV_16S, 0, 1, 3);  /* dx=0, dy=1, ksize=3 */
            cv::convertScaleAbs(gx, abs_gx);
            cv::convertScaleAbs(gy, abs_gy);
            cv::addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, dst);
        }

        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            cv::Sobel(img, gx, CV_16S, 1, 0, 3);
            cv::Sobel(img, gy, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(gx, abs_gx);
            cv::convertScaleAbs(gy, abs_gy);
            cv::addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, dst);
            times[r] = (double)(now_ns() - t0);
        }

        double m  = dmean(times, REPEATS);
        double s  = dstd(times, REPEATS);
        double bw = (double)(pixels * 10) / (m * 1e-9) / 1e9;
        printf("%-35s  %10.0f  %9.0f  %10.2f\n",
            "opencv_sobel_gx+gy_separate", m, s, bw);
        cv::imwrite("opencv_sobel_gx+gy_separate.png", dst);
        FILE *f = fopen("opencv_gxgy.json","w");

        fprintf(f,
        "{\n"
        "\"implementation\":\"opencv_gxgy\",\n"
        "\"latency_ms\":%.3f,\n"
        "\"threads\":%d,\n"
        "\"backend\":\"%s\"\n"
        "}\n",
        m/1e6,
        cv::getNumThreads(),
        "OpenCV");

        fclose(f);
    }



    /* ── method 2: single Sobel call outputting float magnitude directly ────
       cv::Sobel with CV_32F output gives float gradients.
       Computing magnitude = sqrt(gx²+gy²) in float — the L2 norm path      */
    {
        cv::Mat gx, gy, mag;

        for (int i = 0; i < 3; i++) {
            cv::Sobel(img, gx, CV_32F, 1, 0, 3);
            cv::Sobel(img, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, mag);
        }

        for (int r = 0; r < REPEATS; r++) {
            uint64_t t0 = now_ns();
            cv::Sobel(img, gx, CV_32F, 1, 0, 3);
            cv::Sobel(img, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, mag);
            times[r] = (double)(now_ns() - t0);
        }

        double m  = dmean(times, REPEATS);
        double s  = dstd(times, REPEATS);
        double bw = (double)(pixels * 10) / (m * 1e-9) / 1e9;
        printf("%-35s  %10.0f  %9.0f  %10.2f\n",
            "opencv_sobel_float_magnitude", m, s, bw);
        cv::Mat mag8;

        cv::normalize(mag, mag8, 0, 255, cv::NORM_MINMAX);
        mag8.convertTo(mag8, CV_8U);

        cv::imwrite("opencv_sobel_float_magnitude.png", mag8);
        FILE *f = fopen("opencv_float.json","w");

        fprintf(f,
        "{\n"
        "\"implementation\":\"opencv_float\",\n"
        "\"latency_ms\":%.3f,\n"
        "\"threads\":%d,\n"
        "\"backend\":\"%s\"\n"
        "}\n",
        m/1e6,
        cv::getNumThreads(),
        "OpenCV");

        fclose(f);
    }

    
    /* ── print build info ── */
    printf("\nOpenCV build info (relevant lines):\n");
    std::string info = cv::getBuildInformation();
    std::istringstream stream(info);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("IPP") != std::string::npos ||
            line.find("TBB") != std::string::npos ||
            line.find("OpenMP") != std::string::npos ||
            line.find("Parallel") != std::string::npos)
            printf("  %s\n", line.c_str());
    }

    printf("TBB threads: %d\n", cv::getNumThreads());   /* ← here */

    
    stbi_image_free(img_raw);
    return 0;
}
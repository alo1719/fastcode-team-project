#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream> 
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define THREADS 4

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

using namespace cv;
using namespace std;

void nn_kernel_8x16(__m256 row_diff, __m256 ruler_col, __m256 ruler_col2, __m256 bdx, Mat image, Mat output_image, int x, int y){

    __m256 col_diff1 = _mm256_mul_ps(ruler_col, bdx);
    __m256 col_diff2 = _mm256_mul_ps(ruler_col2, bdx);
    col_diff1 = _mm256_floor_ps(col_diff1);
    col_diff2 = _mm256_floor_ps(col_diff2);

    __m256 ymm0 = _mm256_broadcast_ss(&row_diff[0]);// latency 5, throuput 0.5, port23
    __m256 ymm1 = _mm256_broadcast_ss(&row_diff[0]);
    __m256 ymm2 = _mm256_broadcast_ss(&row_diff[1]);
    __m256 ymm3 = _mm256_broadcast_ss(&row_diff[1]);
    __m256 ymm4 = _mm256_broadcast_ss(&row_diff[2]);
    __m256 ymm5 = _mm256_broadcast_ss(&row_diff[2]);
    __m256 ymm6 = _mm256_broadcast_ss(&row_diff[3]);
    __m256 ymm7 = _mm256_broadcast_ss(&row_diff[3]);
    ymm0 = _mm256_add_ps(ymm0, col_diff1);// latency 3, throughput 1, port1, the **bottleneck**
    ymm1 = _mm256_add_ps(ymm1, col_diff2);
    ymm2 = _mm256_add_ps(ymm2, col_diff1);
    ymm3 = _mm256_add_ps(ymm3, col_diff2);
    ymm4 = _mm256_add_ps(ymm4, col_diff1);
    ymm5 = _mm256_add_ps(ymm5, col_diff2);
    ymm6 = _mm256_add_ps(ymm6, col_diff1);
    ymm7 = _mm256_add_ps(ymm7, col_diff2);
    for (int i = 0; i < 8; i++) {
        output_image.at<Vec3b>(x, y+i) = image.at<Vec3b>((int)ymm0[i]);
        output_image.at<Vec3b>(x, y+i+8) = image.at<Vec3b>((int)ymm1[i]);
        output_image.at<Vec3b>(x+1, y+i) = image.at<Vec3b>((int)ymm2[i]);
        output_image.at<Vec3b>(x+1, y+i+8) = image.at<Vec3b>((int)ymm3[i]);
        output_image.at<Vec3b>(x+2, y+i) = image.at<Vec3b>((int)ymm4[i]);
        output_image.at<Vec3b>(x+2, y+i+8) = image.at<Vec3b>((int)ymm5[i]);
        output_image.at<Vec3b>(x+3, y+i) = image.at<Vec3b>((int)ymm6[i]);
        output_image.at<Vec3b>(x+3, y+i+8) = image.at<Vec3b>((int)ymm7[i]);
    }

    ymm0 = _mm256_broadcast_ss(&row_diff[4]);
    ymm1 = _mm256_broadcast_ss(&row_diff[4]);
    ymm2 = _mm256_broadcast_ss(&row_diff[5]);
    ymm3 = _mm256_broadcast_ss(&row_diff[5]);
    ymm4 = _mm256_broadcast_ss(&row_diff[6]);
    ymm5 = _mm256_broadcast_ss(&row_diff[6]);
    ymm6 = _mm256_broadcast_ss(&row_diff[7]);
    ymm7 = _mm256_broadcast_ss(&row_diff[7]);
    ymm0 = _mm256_add_ps(ymm0, col_diff1);
    ymm1 = _mm256_add_ps(ymm1, col_diff2);
    ymm2 = _mm256_add_ps(ymm2, col_diff1);
    ymm3 = _mm256_add_ps(ymm3, col_diff2);
    ymm4 = _mm256_add_ps(ymm4, col_diff1);
    ymm5 = _mm256_add_ps(ymm5, col_diff2);
    ymm6 = _mm256_add_ps(ymm6, col_diff1);
    ymm7 = _mm256_add_ps(ymm7, col_diff2);
    for (int i = 0; i < 8; i++) {
        output_image.at<Vec3b>(x+4, y+i) = image.at<Vec3b>((int)ymm0[i]);
        output_image.at<Vec3b>(x+4, y+i+8) = image.at<Vec3b>((int)ymm1[i]);
        output_image.at<Vec3b>(x+5, y+i) = image.at<Vec3b>((int)ymm2[i]);
        output_image.at<Vec3b>(x+5, y+i+8) = image.at<Vec3b>((int)ymm3[i]);
        output_image.at<Vec3b>(x+6, y+i) = image.at<Vec3b>((int)ymm4[i]);
        output_image.at<Vec3b>(x+6, y+i+8) = image.at<Vec3b>((int)ymm5[i]);
        output_image.at<Vec3b>(x+7, y+i) = image.at<Vec3b>((int)ymm6[i]);
        output_image.at<Vec3b>(x+7, y+i+8) = image.at<Vec3b>((int)ymm7[i]);
    }
}

void img_proc(int image_rows, int image_cols, int input_row, int input_col, Mat image,
              Mat output_image, int kernel_m, int kernel_n, float dx, float dy) {
    
    float image_cols_f = (float)image_cols;
    __m256 cols = _mm256_broadcast_ss(&image_cols_f);
    __m256 bdx = _mm256_broadcast_ss(&dx);
    __m256 bdy = _mm256_broadcast_ss(&dy);
    __m256 plus8 = _mm256_set1_ps(8.0);
    __m256 plus16 = _mm256_set1_ps(16.0);
    __m256 ruler_row = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 initial_col = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 initial_col2 = _mm256_set_ps(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0);

    for (int i = 0; i < image_rows / input_row; i++) {
        // printf("thread = %d\n", omp_get_thread_num());
        __m256 row_diff = _mm256_mul_ps(ruler_row, bdy);
        row_diff = _mm256_floor_ps(row_diff);
        row_diff = _mm256_mul_ps(row_diff, cols);
        __m256 ruler_col = initial_col;
        __m256 ruler_col2 = initial_col2;
        for (int j = 0; j < image_cols / input_col; j++){

            nn_kernel_8x16(row_diff, ruler_col, ruler_col2, bdx, image, output_image, i*kernel_m, j*kernel_n);

            ruler_col = _mm256_add_ps(ruler_col, plus16);
            ruler_col2 = _mm256_add_ps(ruler_col2, plus16);
        }
        ruler_row = _mm256_add_ps(ruler_row, plus8);
    }
    
}

void img_proc_parallel(int image_rows, int image_cols, int input_row, int input_col, Mat image,
              Mat output_image, int kernel_m, int kernel_n, float dx, float dy) {
    
    float image_cols_f = (float)image_cols;
    __m256 cols = _mm256_broadcast_ss(&image_cols_f);
    __m256 bdx = _mm256_broadcast_ss(&dx);
    __m256 bdy = _mm256_broadcast_ss(&dy);
    __m256 v8 = _mm256_set1_ps(8.0);
    __m256 v16 = _mm256_set1_ps(16.0);
    __m256 ruler_row = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 initial_col = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 initial_col2 = _mm256_set_ps(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0);

    #pragma omp parallel for schedule(static, image_rows / input_row / THREADS)
    for (int i = 0; i < image_rows / input_row; i++) {
        // printf("thread = %d\n", omp_get_thread_num());
        __m256 vi = _mm256_set1_ps((float)i);
        ruler_row = _mm256_fmadd_ps(vi, v8, initial_col);
        __m256 row_diff = _mm256_mul_ps(ruler_row, bdy);
        row_diff = _mm256_floor_ps(row_diff);
        row_diff = _mm256_mul_ps(row_diff, cols);
        __m256 ruler_col = initial_col;
        __m256 ruler_col2 = initial_col2;
        for (int j = 0; j < image_cols / input_col; j++){
            nn_kernel_8x16(row_diff, ruler_col, ruler_col2, bdx, image, output_image, i*kernel_m, j*kernel_n);

            ruler_col = _mm256_add_ps(ruler_col, v16);
            ruler_col2 = _mm256_add_ps(ruler_col2, v16);
        }
    }
}

void naive(Mat image, int des_m, int des_n, Mat output_image) {
    float divisionH = (float)image.rows / (float)des_m;
    float divisionW = (float)image.cols / (float)des_n;
    for (int i = 0; i < des_m; i++) 
        for (int ii = 0; ii < des_n; ii++) 
            output_image.at<cv::Vec3b>(i, ii) = image.at<cv::Vec3b>((int)((float)i * divisionH), (int)((float)ii * divisionW));
}

void naive_parallel(Mat image, int des_m, int des_n, Mat output_image) {
    float divisionH = (float)image.rows / (float)des_m;
    float divisionW = (float)image.cols / (float)des_n;
    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < des_m; i++) 
        for (int ii = 0; ii < des_n; ii++) 
            output_image.at<cv::Vec3b>(i, ii) = image.at<cv::Vec3b>((int)((float)i * divisionH), (int)((float)ii * divisionW));
}

int main(int argc, char** argv)
{

    int des_m = atoi(argv[1]); // output #row
    int des_n = atoi(argv[2]); // output #col

    Mat image;
    image = imread("./anime.jpg", 1);
    if (!image.data)
    {
        printf("No image data\n");
        return -1;
    }
    int image_rows = image.rows;
    int image_cols = image.cols;
    int graph_size = image.cols * image.rows;

    float dx = 1.0 * image_rows / des_m;
    float dy = 1.0 * image_cols / des_n;

    int kernel_m = 8; // nn kernel row size
    int kernel_n = 16; // nn kernel col size

    int input_row = ceil(dx * kernel_m);
    int input_col = ceil(dy * kernel_n);

    Mat output_image(des_m, des_n, CV_8UC3);

    unsigned long long acc = 0;
    for (int i = 0; i < 20; i++) {
        unsigned long long start = rdtsc();

        // naive(image, des_m, des_n, output_image);
        // naive_parallel(image, des_m, des_n, output_image);
        // img_proc(image_rows, image_cols, input_row, input_col, image, output_image, kernel_m, kernel_n, dx, dy);
        // img_proc_parallel(image_rows, image_cols, input_row, input_col, image, output_image, kernel_m, kernel_n, dx, dy);
        // resize(image, output_image, Size(des_n, des_m), 0, 0, CV_INTER_NN);
        unsigned long long end = rdtsc();
        acc += end - start;
    }

    imwrite("./output.jpg", output_image);

    printf("average cycles: %f\n", (float)acc / 20);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "bilinear.h"

#define MAX_FREQ 4.2
#define BASE_FREQ 1.8

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void bilinear_driver(int ori_h, int ori_w, int new_h, int new_w, float *from, float *to,
        int *row_indices, int *row_indices_plus1, float *parameters){

    float height_division = (float) ori_h / (float) new_h;
    float width_division = (float) ori_w / (float) new_w;

    __m256 ymm0 = _mm256_set1_ps(width_division);

    __m256 ymm1 = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 x = _mm256_mul_ps(ymm1, ymm0); //y
    float *x_array = (float *) calloc(8, sizeof(float));
    memcpy(x_array, &x, 8*sizeof(float));

    __m256 floor = _mm256_floor_ps(x); //floor_y
    __m256 diff = _mm256_sub_ps(x, floor); //diff_y
    ymm1 = _mm256_set1_ps(1);
    __m256 m_diff = _mm256_sub_ps(ymm1, diff); //1 - diff_y

    ymm0 = _mm256_set1_ps(ori_w);
    __m256 row_idx = _mm256_mul_ps(floor, ymm0);
    float *row_indices_float = (float *) calloc(8, sizeof(float));
    memcpy(row_indices_float, &row_idx, 8*sizeof(float));
    for (int i = 0; i != 8; i++) {
        row_indices[i] = (int) row_indices_float[i];
    }

    ymm0 = _mm256_set1_ps(ori_w);
    __m256 row_idx_plus1 = _mm256_add_ps(row_idx, ymm0);
    float *row_indices_plus1_float = (float *) calloc(8, sizeof(float));
    memcpy(row_indices_plus1_float, &row_idx_plus1, 8*sizeof(float));
    for (int i = 0; i != 8; i++) {
        row_indices_plus1[i] = (int) row_indices_plus1_float[i];
    }

    for (int row = 0; row != 8; row ++) {
        float x_idx = x_array[row];

        __m256 ymm0 = _mm256_set1_ps(x_idx);
        __m256 ymm1 = _mm256_floor_ps(ymm0);
        ymm0 = _mm256_sub_ps(ymm0, ymm1); // diff_x
        ymm1 = _mm256_set1_ps(1);
        ymm1 = _mm256_sub_ps(ymm1, ymm0); // 1 - diff_x
        __m256 ymm6 = _mm256_mul_ps(ymm1, m_diff); // (1 - diff_x) * (1 - diff_y)
        int row_offset = row*4*8;
        memcpy(parameters+row_offset, &ymm6, 8*sizeof(float));
        __m256 ymm7 = _mm256_mul_ps(ymm0, m_diff); // diff_x * (1 - diff_y)
        memcpy(parameters+row_offset+8, &ymm7, 8*sizeof(float));
        __m256 ymm8 = _mm256_mul_ps(ymm1, diff); // (1 - diff_x) * diff_y
        memcpy(parameters+row_offset+16, &ymm8, 8*sizeof(float));
        __m256 ymm9 = _mm256_mul_ps(ymm0, diff); // diff_x * diff_y
        memcpy(parameters+row_offset+24, &ymm9, 8*sizeof(float));
    }
#if 0
    for(int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%d %d \n", i, j);
            for (int p = 0; p < 8; p++) {
                printf("%f\t", parameters[i*32+j*8+p]);
            }
            printf("\n");
        }
    }
#endif
    // Input index = floor(output index * input length / output length)

    // compute the permute mask for upscaling
    // because the kernel size is 8*8, so every time we are going to load 8 elements from
    // the input image and then permute them to get the correct order.
    // For example, we are computing the output at 0 1 2 3 4 5 6 7,
    // and for that, we need the input elements at 0 0 1 1 2 3 3 4.
    // _mm256_set_ps(input[0], input[0], input[1], input[1], input[2], input[3], input[3], input[4]) 
    // then do the following computation.

    // output size=8
    // if input size =8 do nothing 
    // then scale ratio is (1-7) / 8
    // 7 possibilities for index mask
 
    free(x_array);
}
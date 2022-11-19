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

int main(){
    int ori_h = 5;
    int ori_w = 5;
    int new_h = 8;
    int new_w = 8;
    float *from; // m*n
    float *to; // 8*8
    posix_memalign((void**)&from, 32, ori_h*ori_w*sizeof(int));
    posix_memalign((void**)&to, 32, 8*8*sizeof(int));
    for (int i = 0; i < ori_h*ori_w; i++) {
        from[i] = i;
    }

    float height_division = (float) ori_h / (float) new_h;
    float width_division = (float) ori_w / (float) new_w;

    __m256 ymm0 = _mm256_set1_ps(width_division);
    // printf("width_division\n");
    // print256_num(ymm0);

    __m256 ymm1 = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 x = _mm256_mul_ps(ymm1, ymm0); //y
    float *x_array = calloc(8, sizeof(float));
    memcpy(x_array, &x, sizeof(x_array));
    // printf("x\n");
    // print256_num(x);
    __m256 floor = _mm256_floor_ps(x); //floor_y
    // printf("floor_y\n");
    // print256_num(floor);
    __m256 diff = _mm256_sub_ps(x, floor); //diff_y
    // printf("diff_y\n");
    // print256_num(diff);
    ymm1 = _mm256_set1_ps(1);
    __m256 m_diff = _mm256_sub_ps(ymm1, diff); //1 - diff_y
    // printf("m_diff\n");
    // print256_num(m_diff);

    ymm0 = _mm256_set1_ps(new_w);
    __m256 row_idx = _mm256_mul_ps(floor, ymm0);
    float *row_indices_float = calloc(8, sizeof(float));
    memcpy(row_indices_float, &row_idx, sizeof(row_indices_float));
    int *row_indices = calloc(8, sizeof(int));
    for (int i = 0; i != 8; i++) {
        row_indices[i] = (int) row_indices_float[i];
    }
    ymm0 = _mm256_set1_ps(new_w);
    __m256 row_idx_plus1 = _mm256_add_ps(row_idx, ymm0);
    float *row_indices_plus1_float = calloc(8, sizeof(float));
    memcpy(row_indices_plus1_float, &row_idx_plus1, sizeof(row_indices_plus1_float));
    int *row_indices_plus1 = calloc(8, sizeof(int));
    for (int i = 0; i != 8; i++) {
        row_indices_plus1[i] = (int) row_indices_plus1_float[i];
    }
    float *parameters = calloc(8*4*8, sizeof(float));
    for (int row = 1; row != 7; row ++) {
        int floor_x_idx = row_indices[row];
        int floor_x_idx_plus1 = row_indices_plus1[row];
        float x_idx = x_array[row];
        // printf("x_array[row] %f\n", x_idx);

        __m256 ymm0 = _mm256_set1_ps(x_idx);
        // printf("x[row]\n");
        // print256_num(ymm0);
        __m256 ymm1 = _mm256_floor_ps(ymm0);
        ymm0 = _mm256_sub_ps(ymm0, ymm1); // diff_x
        // printf("diff_x\n");
        // print256_num(ymm0);
        ymm1 = _mm256_set1_ps(1);
        ymm1 = _mm256_sub_ps(ymm1, ymm0); // 1 - diff_x
        // printf("1 - diff_x\n");
        // print256_num(ymm1);
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
    // permute mask for floor_y
    __m256i mask_floory = _mm256_setzero_si256();
    // permute mask for floor_y + 1
    __m256i mask_flooryp = _mm256_setzero_si256();
    switch(ori_h) {
    case 1:
        mask_floory = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
        break;
    case 2:
        mask_floory = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(2, 2, 2, 2, 1, 1, 1, 1);
        break;
    case 3:
        mask_floory = _mm256_set_epi32(2, 2, 1, 1, 1, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(3, 3, 2, 2, 2, 1, 1, 1);
        break;
    case 4:
        mask_floory = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(4, 4, 3, 3, 2, 2, 1, 1);
        break;
    case 5:
        mask_floory = _mm256_set_epi32(4, 3, 3, 2, 1, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(5, 4, 4, 3, 2, 2, 1, 1);
        break;
    case 6:
        mask_floory = _mm256_set_epi32(5, 4, 3, 3, 2, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(6, 5, 4, 4, 3, 2, 1, 1);
        break;
    case 7:
        mask_floory = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 1);
        break;
    default:
        break;
    }

    unsigned long long acc = 0;
    for (int i = 0; i < 100000; i++) {
        unsigned long long start = rdtsc();
        bilinear_kernel_upscale(
                        row_indices, row_indices_plus1,
                         from, to, new_w,
                         mask_floory,  mask_flooryp, parameters);
        unsigned long long end = rdtsc();
        acc += end - start;
        // bilinear_naive();
    }

    printf("average cycles: %llu\n", acc/100000);
    // 16 instructions per element
    // 20 floating operations per element
    printf("%d\t %d\t %lf\n", ori_h, ori_w, (100000.0*new_h*6*4*2)/((double)(acc)*MAX_FREQ/BASE_FREQ));

    free(from);
    free(to);
}
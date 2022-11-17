#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void print256_num(__m256 var)
{
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %f %f %f %f %f %f %f %f\n", 
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}

//the default parameters are used for testing
int main() {
    int ori_h = 16;
    int ori_w = 16;
    int new_h = 8;
    int new_w = 8;
    int *from; // m*n
    float *to; // 8*8
    posix_memalign((void**)&from, 32, ori_h*ori_w*sizeof(int));
    posix_memalign((void**)&to, 32, 8*8*sizeof(int));
    for (int i = 0; i < ori_h*ori_w; i++) {
        from[i] = i;
    }

    float height_division = (float) ori_h/ new_h;
    float width_division = (float) ori_w / new_w;

    __m256 ymm0 = _mm256_set1_ps(width_division);
    __m256 ymm1 = {0, 1, 2, 3, 4, 5, 6, 7};
    ymm0 = _mm256_mul_ps(ymm1, ymm0); //y

    __m256 ymm2 = _mm256_floor_ps(ymm0); //floor_y
    __m256 ymm3 = _mm256_sub_ps(ymm0, ymm2); //diff_y
    ymm1 = _mm256_set1_ps(1);
    __m256 ymm4 = _mm256_add_ps(ymm2, ymm1); //floor_y + 1
    __m256 ymm5 = _mm256_sub_ps(ymm1, ymm3); //1 - diff_y

    for (int cur_row = 0; cur_row != 8; ++cur_row) {
        float x = cur_row * height_division;
        int floor_x = floor(x);
        int floor_x_idx = floor_x * ori_w;

        int floor_x_plus1 = floor_x + 1;
        int floor_x_plus1_idx = floor_x_plus1 * ori_w;

        float diff_x = x = floor_x;

        __m256 ymm6 = _mm256_set1_ps(diff_x); //diff_x

        ymm1 = _mm256_set1_ps(1);

        __m256 ymm7 = _mm256_sub_ps(ymm1, ymm6); //1 - diff_x

        __m256 ymm8 = _mm256_mul_ps(ymm7, ymm5); //(1 - diff_x) * (1 - diffy)
        __m256 ymm9 = _mm256_mul_ps(ymm6, ymm5); //diff_x * (1 - diffy)

        __m256 ymm10 = _mm256_mul_ps(ymm7, ymm3); // (1 - diff_x) * diffy
        __m256 ymm11 = _mm256_mul_ps(ymm6, ymm3); // diff_x * diffy

        ymm6 = _mm256_set1_ps(floor_x_idx); //floor_x
        ymm7 = _mm256_set1_ps(floor_x_plus1_idx); //floor_x + 1

        // [floor_x, floor_y]
        __m256 ymm12 = _mm256_add_ps(ymm6, ymm2);
        // [floor_x + 1, floor_y]
        __m256 ymm13 = _mm256_add_ps(ymm7, ymm2);
        // [floor_x, floor_y + 1]
        __m256 ymm14 = _mm256_add_ps(ymm6, ymm4);
        // [floor_x + 1, floor_y + 1]
        __m256 ymm15 = _mm256_add_ps(ymm7, ymm4);

        // now we got all the indices of the needed original elements
        // int and float types are not consistent due to floor
        ymm12 = _mm256_set_ps(from[(int)ymm12[7]], from[(int)ymm12[6]], from[(int)ymm12[5]], from[(int)ymm12[4]],
                            from[(int)ymm12[3]], from[(int)ymm12[2]], from[(int)ymm12[1]], from[(int)ymm12[0]]);
        ymm13 = _mm256_set_ps(from[(int)ymm13[7]], from[(int)ymm13[6]], from[(int)ymm13[5]], from[(int)ymm13[4]],
                            from[(int)ymm13[3]], from[(int)ymm13[2]], from[(int)ymm13[1]], from[(int)ymm13[0]]);
        ymm14 = _mm256_set_ps(from[(int)ymm14[7]], from[(int)ymm14[6]], from[(int)ymm14[5]], from[(int)ymm14[4]],
                            from[(int)ymm14[3]], from[(int)ymm14[2]], from[(int)ymm14[1]], from[(int)ymm14[0]]);
        ymm15 = _mm256_set_ps(from[(int)ymm15[7]], from[(int)ymm15[6]], from[(int)ymm15[5]], from[(int)ymm15[4]],
                             from[(int)ymm15[3]], from[(int)ymm15[2]], from[(int)ymm15[1]], from[(int)ymm15[0]]);
#if 0
        printf("[floor_x, floor_y]\n");
        print256_num(ymm12);
        printf("[floor_x + 1, floor_y]\n");
        print256_num(ymm13);
        printf("[floor_x, floor_y + 1]\n");
        print256_num(ymm14);
        printf("[floor_x + 1, floor_y + 1]\n");
        print256_num(ymm15);
#endif
        ymm0 = _mm256_setzero_ps();
        ymm0 = _mm256_fmadd_ps(ymm12, ymm8, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm13, ymm9, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm14, ymm10, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm15, ymm11, ymm0);

        // printf("result \n");
        // print256_num(ymm0);
        _mm256_store_ps(&to[cur_row*new_w], ymm0);
    }
}
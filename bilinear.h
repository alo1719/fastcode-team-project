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
void print256i_num(__m256i var)
{
    int val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %d %d %d %d %d %d %d %d\n", 
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}

// //the default parameters are used for testing
// int bilinear_kernel(__m256 mdiffx_mdiffy, __m256 diffx_mdiffy,
//                     __m256 mdiffx_diffy, __m256 diffx_diffy,
//                     __m256 floorx_floory, __m256 floorxp_floory,
//                     __m256 floorx_flooryp, __m256 floorxp_floory,
//                      int *from, float *to) {
//         // // now we got all the indices of the needed original elements
//         // // int and float types are not consistent due to floor
//         // // Question: is there a better way to load nonconsecutive elements?
//         ymm12 = _mm256_set_ps(from[(int)ymm12[7]], from[(int)ymm12[6]], from[(int)ymm12[5]], from[(int)ymm12[4]],
//                             from[(int)ymm12[3]], from[(int)ymm12[2]], from[(int)ymm12[1]], from[(int)ymm12[0]]);
//         ymm13 = _mm256_set_ps(from[(int)ymm13[7]], from[(int)ymm13[6]], from[(int)ymm13[5]], from[(int)ymm13[4]],
//                             from[(int)ymm13[3]], from[(int)ymm13[2]], from[(int)ymm13[1]], from[(int)ymm13[0]]);
//         ymm14 = _mm256_set_ps(from[(int)ymm14[7]], from[(int)ymm14[6]], from[(int)ymm14[5]], from[(int)ymm14[4]],
//                             from[(int)ymm14[3]], from[(int)ymm14[2]], from[(int)ymm14[1]], from[(int)ymm14[0]]);
//         ymm15 = _mm256_set_ps(from[(int)ymm15[7]], from[(int)ymm15[6]], from[(int)ymm15[5]], from[(int)ymm15[4]],
//                              from[(int)ymm15[3]], from[(int)ymm15[2]], from[(int)ymm15[1]], from[(int)ymm15[0]]);

// #if 0
//         printf("(1 - diff_x) * (1 - diffy)\n");
//         print256_num(ymm8);
//         printf("diff_x * (1 - diffy)\n");
//         print256_num(ymm9);
//         printf("(1 - diff_x) * diffy\n");
//         print256_num(ymm10);
//         printf("diff_x * diffy\n");
//         print256_num(ymm11);
//         printf("[floor_x, floor_y]\n");
//         print256_num(ymm12);
//         printf("[floor_x + 1, floor_y]\n");
//         print256_num(ymm13);
//         printf("[floor_x, floor_y + 1]\n");
//         print256_num(ymm14);
//         printf("[floor_x + 1, floor_y + 1]\n");
//         print256_num(ymm15);
// #endif
//         ymm0 = _mm256_setzero_ps();
//         ymm0 = _mm256_fmadd_ps(ymm12, ymm8, ymm0);
//         ymm0 = _mm256_fmadd_ps(ymm13, ymm9, ymm0);
//         ymm0 = _mm256_fmadd_ps(ymm14, ymm10, ymm0);
//         ymm0 = _mm256_fmadd_ps(ymm15, ymm11, ymm0);

//         // printf("result ==========================================\n");
//         // print256_num(ymm0);
//         _mm256_store_ps(&to[cur_row*new_w], ymm0);
// }


void bilinear_kernel_upscale(__m256 m_diff, __m256 diff, __m256 x,
                        __m256 row_idx, __m256 row_idx_plus1,
                        float *from, float *to, int new_w, int ori_h,
                        __m256i mask_floory, __m256i mask_flooryp) {
    // 7 simd register taken, 9 left
    float row_indices [8];
    memcpy(row_indices, &row_idx, sizeof(row_indices));
    float row_indices_plus1 [8];
    memcpy(row_indices_plus1, &row_idx_plus1, sizeof(row_indices_plus1));
    float x_array [8];
    memcpy(x_array, &x, sizeof(x_array));
    // printf("x\n");
    // print256_num(x);
    // for (int i = 0; i != 8; i++) {
    //         printf("%f\n", x_array[i]);
    // }

    // print256_num(row_idx);
    // print256_num(row_idx_plus1);
    // print256i_num(mask_floory);
    // print256i_num(mask_flooryp);

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
        __m256 ymm7 = _mm256_mul_ps(ymm0, m_diff); // diff_x * (1 - diff_y)
        __m256 ymm8 = _mm256_mul_ps(ymm1, diff); // (1 - diff_x) * diff_y
        ymm1 = _mm256_mul_ps(ymm0, diff); // diff_x * diff_y
#if 0
        printf("(1 - diff_x) * (1 - diff_y)\n");
        print256_num(ymm6);
        printf("diff_x * (1 - diff_y)\n");
        print256_num(ymm7);
        printf("(1 - diff_x) * diff_y\n");
        print256_num(ymm8);
        printf("diff_x * diff_y\n");
        print256_num(ymm1);
#endif
        __m256 ymm2 = _mm256_load_ps(&(from[floor_x_idx]));
        __m256 ymm3 = _mm256_load_ps(&(from[floor_x_idx_plus1]));
        __m256 ymm4 = _mm256_load_ps(&(from[floor_x_idx]));
        __m256 ymm5 = _mm256_load_ps(&(from[floor_x_idx_plus1]));


#if 0
        printf("[floor_x, floor_y]\n");
        print256_num(ymm2);
        printf("[floor_x + 1, floor_y]\n");
        print256_num(ymm3);
        printf("[floor_x, floor_y + 1]\n");
        print256_num(ymm4);
        printf("[floor_x + 1, floor_y + 1]\n");
        print256_num(ymm5);
#endif

        ymm2 = _mm256_permutevar8x32_ps(ymm2, mask_floory);
        ymm3 = _mm256_permutevar8x32_ps(ymm3, mask_floory);
        ymm4 = _mm256_permutevar8x32_ps(ymm4, mask_flooryp);
        ymm5 = _mm256_permutevar8x32_ps(ymm5, mask_flooryp);

#if 0
        printf("permute     [floor_x, floor_y]\n");
        print256_num(ymm2);
        printf("(1 - diff_x) * (1 - diff_y)\n");
        print256_num(ymm6);
        printf("permute     [floor_x + 1, floor_y]\n");
        print256_num(ymm3);
        printf("diff_x * (1 - diff_y) and (1 - diff_x) * diff_y\n");
        print256_num(ymm7);
        printf("permute     [floor_x, floor_y + 1]\n");
        print256_num(ymm4);
        printf("diff_x * (1 - diff_y) and (1 - diff_x) * diff_y\n");
        print256_num(ymm8);
        printf("permute     [floor_x + 1, floor_y + 1]\n");
        print256_num(ymm5);
        printf("diff_x * diff_y\n");
        print256_num(ymm1);
#endif

        ymm0 = _mm256_setzero_ps();
        ymm0 = _mm256_fmadd_ps(ymm2, ymm6, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm3, ymm7, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm4, ymm8, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm5, ymm1, ymm0);

        // printf("result ==========================================\n");
        // print256_num(ymm0);
        _mm256_store_ps(&to[row*new_w], ymm0);
    }
}

void bilinear_naive() {
        int new_h = 8;
        int new_w = 8;
        float divisionH = (float)5 / (float)new_h;
        float divisionW = (float)5 / (float)new_w;

        float from[5][5]; // m*n
        float to[8][8]; // 8*8
        for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                        from[i][j] = i*5 + j;
                }
        }
        for(int i = 1; i < 8; i++){
                for (int j = 1; j < 8; j++) {
                        float x = (float)i * divisionH;
			float y = (float)j * divisionW;
                        printf("%f\n", y);
			float agirlikX = x - (int)x;
			float agirlikY = y - (int)y;
                        printf("%f\t%f\t%f\t%f\t\n", (float)(1 - agirlikX) * (float)(1 - agirlikY), (float)agirlikX * (float)(1 - agirlikY), 
                        (float)(1 - agirlikX) * (float)agirlikY, (float)agirlikX * (float)agirlikY);
                        to[i][j] = from[(int)x][(int)y] * (float)(1 - agirlikX) * (float)(1 - agirlikY)
                                + from[(int)x + 1][(int)y] * (float)agirlikX * (float)(1 - agirlikY)
                                + from[(int)x][(int)y + 1] * (float)(1 - agirlikX) * (float)agirlikY
                                + from[(int)x + 1][(int)y + 1] * (float)agirlikX * (float)agirlikY;
                        printf("%f\n", to[i][j]);
                }
                printf("\n");
        }
}
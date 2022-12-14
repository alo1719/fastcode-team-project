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

void print_row_indices(int *val)
{
    printf("row_indices: %d %d %d %d %d %d %d %d\n", 
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}
void print_matrix(float *val)
{
    printf("print_matrix: %f %f %f %f %f %f %f %f\n", 
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}

void bilinear_kernel_upscale(
                        int *row_indices, int *row_indices_plus1,
                        float *from, float *to, int new_w,
                        __m256i mask_floory, __m256i mask_flooryp, float *parameters) {
        
        __m256 ymm2 = _mm256_loadu_ps(&(from[row_indices[0]]));
        __m256 ymm3 = _mm256_loadu_ps(from+row_indices_plus1[0]);
        __m256 ymm4 = _mm256_loadu_ps(&(from[row_indices[0]]));
        __m256 ymm5 = _mm256_loadu_ps(&(from[row_indices_plus1[0]]));

        ymm2 = _mm256_permutevar8x32_ps(ymm2, mask_floory);
        ymm3 = _mm256_permutevar8x32_ps(ymm3, mask_floory);
        ymm4 = _mm256_permutevar8x32_ps(ymm4, mask_flooryp);
        ymm5 = _mm256_permutevar8x32_ps(ymm5, mask_flooryp);

        int row_offset = 0*4*8;
        __m256 ymm6 = _mm256_load_ps(parameters);
        __m256 ymm7 = _mm256_load_ps(parameters+row_offset+8);
        __m256 ymm8 = _mm256_load_ps(parameters+row_offset+16);
        __m256 ymm9 = _mm256_load_ps(parameters+row_offset+24);

        __m256 ymm1 = _mm256_setzero_ps();
        ymm1 = _mm256_fmadd_ps(ymm2, ymm6, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm3, ymm7, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm4, ymm8, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm5, ymm9, ymm1);
        printf("%d row_indices[1] %d\n", __LINE__, *(row_indices+1));
        print256_num(ymm1);
        // ymm1 = _mm256_floor_ps(ymm1);
        _mm256_store_ps(&to[0*new_w], ymm1);

        ymm2 = _mm256_load_ps(&(from[row_indices[1]]));
        ymm3 = _mm256_loadu_ps(&(from[row_indices_plus1[1]]));
        ymm4 = _mm256_load_ps(&(from[row_indices[1]]));
        ymm5 = _mm256_loadu_ps(&(from[row_indices_plus1[1]]));

        ymm2 = _mm256_permutevar8x32_ps(ymm2, mask_floory);
        ymm3 = _mm256_permutevar8x32_ps(ymm3, mask_floory);
        ymm4 = _mm256_permutevar8x32_ps(ymm4, mask_flooryp);
        ymm5 = _mm256_permutevar8x32_ps(ymm5, mask_flooryp);

        int row_offset_1 = 1*4*8;
        ymm6 = _mm256_load_ps(parameters);
        ymm7 = _mm256_load_ps(parameters+row_offset_1+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_1+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_1+24);

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
        print256_num(ymm9);
#endif
        __m256 ymm0 = _mm256_setzero_ps();
        ymm0 = _mm256_fmadd_ps(ymm2, ymm6, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm3, ymm7, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm4, ymm8, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm5, ymm9, ymm0);

        printf("result ==========================================\n");
        print256_num(ymm0);
        _mm256_store_ps(&to[1*new_w], ymm0);

        __m256 ymm12 = _mm256_loadu_ps(from+row_indices[2]);
        __m256 ymm13 = _mm256_loadu_ps(&(from[row_indices_plus1[2]]));
        __m256 ymm14 = _mm256_loadu_ps(&(from[row_indices[2]]));
        __m256 ymm15 = _mm256_loadu_ps(&(from[row_indices_plus1[2]]));

        ymm12 = _mm256_permutevar8x32_ps(ymm12, mask_floory);
        ymm13 = _mm256_permutevar8x32_ps(ymm13, mask_floory);
        ymm14 = _mm256_permutevar8x32_ps(ymm14, mask_flooryp);
        ymm15 = _mm256_permutevar8x32_ps(ymm15, mask_flooryp);

        int row_offset_2 = 2*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_2);
        ymm7 = _mm256_load_ps(parameters+row_offset_2+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_2+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_2+24);

        ymm1 = _mm256_setzero_ps();
        ymm1 = _mm256_fmadd_ps(ymm12, ymm6, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm13, ymm7, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm14, ymm8, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm15, ymm9, ymm1);

        _mm256_store_ps(&to[2*new_w], ymm1);

        printf("diff_x * diff_y\n");

        ymm2 = _mm256_loadu_ps(from+row_indices[3]);
        ymm3 = _mm256_loadu_ps(from+row_indices_plus1[3]);
        ymm4 = _mm256_loadu_ps(from+row_indices[3]);
        ymm5 = _mm256_loadu_ps(from+row_indices_plus1[3]);

        ymm2 = _mm256_permutevar8x32_ps(ymm2, mask_floory);
        ymm3 = _mm256_permutevar8x32_ps(ymm3, mask_floory);
        ymm4 = _mm256_permutevar8x32_ps(ymm4, mask_flooryp);
        ymm5 = _mm256_permutevar8x32_ps(ymm5, mask_flooryp);

        int row_offset_3 = 3*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_3);
        ymm7 = _mm256_load_ps(parameters+row_offset_3+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_3+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_3+24);

        __m256 ymm10 = _mm256_setzero_ps();
        ymm10 = _mm256_fmadd_ps(ymm2, ymm6, ymm10);
        ymm10 = _mm256_fmadd_ps(ymm3, ymm7, ymm10);
        ymm10 = _mm256_fmadd_ps(ymm4, ymm8, ymm10);
        ymm10 = _mm256_fmadd_ps(ymm5, ymm9, ymm10);

        // printf("result ==========================================\n");
        // print256_num(ymm0);
        _mm256_store_ps(&to[3*new_w], ymm10);

        ymm12 = _mm256_loadu_ps(&(from[row_indices[4]]));
        ymm13 = _mm256_loadu_ps(&(from[row_indices_plus1[4]]));
        ymm14 = _mm256_loadu_ps(&(from[row_indices[4]]));
        ymm15 = _mm256_loadu_ps(&(from[row_indices_plus1[4]]));

        ymm12 = _mm256_permutevar8x32_ps(ymm12, mask_floory);
        ymm13 = _mm256_permutevar8x32_ps(ymm13, mask_floory);
        ymm14 = _mm256_permutevar8x32_ps(ymm14, mask_flooryp);
        ymm15 = _mm256_permutevar8x32_ps(ymm15, mask_flooryp);

        int row_offset_4 = 4*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_4);
        ymm7 = _mm256_load_ps(parameters+row_offset_4+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_4+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_4+24);

        __m256 ymm11 = _mm256_setzero_ps();
        ymm11 = _mm256_fmadd_ps(ymm12, ymm6, ymm11);
        ymm11 = _mm256_fmadd_ps(ymm13, ymm7, ymm11);
        ymm11 = _mm256_fmadd_ps(ymm14, ymm8, ymm11);
        ymm11 = _mm256_fmadd_ps(ymm15, ymm9, ymm11);

        // printf("result ==========================================\n");
        // print256_num(ymm11);
        _mm256_store_ps(&to[4*new_w], ymm11);


        ymm2 = _mm256_loadu_ps(&(from[row_indices[5]]));
        ymm3 = _mm256_loadu_ps(&(from[row_indices_plus1[5]]));
        ymm4 = _mm256_loadu_ps(&(from[row_indices[5]]));
        ymm5 = _mm256_loadu_ps(&(from[row_indices_plus1[5]]));

        ymm2 = _mm256_permutevar8x32_ps(ymm2, mask_floory);
        ymm3 = _mm256_permutevar8x32_ps(ymm3, mask_floory);
        ymm4 = _mm256_permutevar8x32_ps(ymm4, mask_flooryp);
        ymm5 = _mm256_permutevar8x32_ps(ymm5, mask_flooryp);

        int row_offset_5 = 5*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_5);
        ymm7 = _mm256_load_ps(parameters+row_offset_5+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_5+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_5+24);

        ymm0 = _mm256_setzero_ps();
        ymm0 = _mm256_fmadd_ps(ymm2, ymm6, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm3, ymm7, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm4, ymm8, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm5, ymm9, ymm0);

        // printf("result ==========================================\n");
        // print256_num(ymm0);
        _mm256_store_ps(&to[5*new_w], ymm0);

        ymm12 = _mm256_loadu_ps(&(from[row_indices[6]]));
        ymm13 = _mm256_loadu_ps(&(from[row_indices_plus1[6]]));
        ymm14 = _mm256_loadu_ps(&(from[row_indices[6]]));
        ymm15 = _mm256_loadu_ps(&(from[row_indices_plus1[6]]));

        ymm12 = _mm256_permutevar8x32_ps(ymm12, mask_floory);
        ymm13 = _mm256_permutevar8x32_ps(ymm13, mask_floory);
        ymm14 = _mm256_permutevar8x32_ps(ymm14, mask_flooryp);
        ymm15 = _mm256_permutevar8x32_ps(ymm15, mask_flooryp);

        int row_offset_6 = 6*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_6);
        ymm7 = _mm256_load_ps(parameters+row_offset_6+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_6+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_6+24);

        ymm1 = _mm256_setzero_ps();
        ymm1 = _mm256_fmadd_ps(ymm12, ymm6, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm13, ymm7, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm14, ymm8, ymm1);
        ymm1 = _mm256_fmadd_ps(ymm15, ymm9, ymm1);
        _mm256_store_ps(&to[6*new_w], ymm1);

        ymm12 = _mm256_loadu_ps(&(from[row_indices[7]]));
        ymm13 = _mm256_loadu_ps(&(from[row_indices_plus1[7]]));
        ymm14 = _mm256_loadu_ps(&(from[row_indices[7]]));
        ymm15 = _mm256_loadu_ps(&(from[row_indices_plus1[7]]));

        ymm12 = _mm256_permutevar8x32_ps(ymm12, mask_floory);
        ymm13 = _mm256_permutevar8x32_ps(ymm13, mask_floory);
        ymm14 = _mm256_permutevar8x32_ps(ymm14, mask_flooryp);
        ymm15 = _mm256_permutevar8x32_ps(ymm15, mask_flooryp);

        int row_offset_7 = 7*4*8;
        ymm6 = _mm256_load_ps(parameters+row_offset_7);
        ymm7 = _mm256_load_ps(parameters+row_offset_7+8);
        ymm8 = _mm256_load_ps(parameters+row_offset_7+16);
        ymm9 = _mm256_load_ps(parameters+row_offset_7+24);

        ymm0 = _mm256_setzero_ps();
        ymm0 = _mm256_fmadd_ps(ymm12, ymm6, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm13, ymm7, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm14, ymm8, ymm0);
        ymm0 = _mm256_fmadd_ps(ymm15, ymm9, ymm0);

        _mm256_store_ps(&to[7*new_w], ymm0);
        for (int i = 0; i != 64; i++) {
                printf("%f\t", to[i]);
                if (i % 8 == 7) {
                        printf("\n");
                }
        }
        printf("-------end of kernel result---------\n");
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
        for(int i = 1; i < 7; i++){
                for (int j = 1; j < 7; j++) {
                        float x = (float)i * divisionH;
			float y = (float)j * divisionW;
			float agirlikX = x - (int)x;
			float agirlikY = y - (int)y;
                        // printf("%f\t%f\t%f\t%f\t\n", (float)(1 - agirlikX) * (float)(1 - agirlikY), (float)agirlikX * (float)(1 - agirlikY), 
                        //         (float)(1 - agirlikX) * (float)agirlikY, (float)agirlikX * (float)agirlikY);
                        to[i][j] = from[(int)x][(int)y] * (float)(1 - agirlikX) * (float)(1 - agirlikY)
                                + from[(int)x + 1][(int)y] * (float)agirlikX * (float)(1 - agirlikY)
                                + from[(int)x][(int)y + 1] * (float)(1 - agirlikX) * (float)agirlikY
                                + from[(int)x + 1][(int)y + 1] * (float)agirlikX * (float)agirlikY;
                }
        }
        for(int i = 0; i < 8; i++){
                for (int j = 0; j < 8; j++) {
                        printf("%f\t", to[i][j]);
                }
                printf("\n");
        }
}
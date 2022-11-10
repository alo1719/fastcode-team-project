#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

int main(){
    int m=4;
    int n=4;
    int *from; // m*n
    int *to;// 8*8
    float *to_address; // 8*8
    posix_memalign((void**)&from, 32, m*n*sizeof(int));
    posix_memalign((void**)&to, 32, 8*8*sizeof(int));
    posix_memalign((void**)&to_address, 32, 8*8*sizeof(float));
    for (int i = 0; i < m*n; i++) {
        from[i] = i;
    }
    // ---------------------------------- kernel start -----------------------------
    float dx = n/8.0;
    float dy = m/8.0;
    __m256 bdx = _mm256_set1_ps(dx);
    __m256 col = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 col_diff = _mm256_mul_ps(col, bdx);
    col_diff = _mm256_floor_ps(col_diff);
    // __m256 size = _mm256_set1_ps(32.0);
    // col_diff = _mm256_mul_ps(col_diff, size);
    printf("%f %f %f %f %f %f %f %f\n", col_diff[0], col_diff[1], col_diff[2], col_diff[3], col_diff[4], col_diff[5], col_diff[6], col_diff[7]);
    __m256 bdy = _mm256_set1_ps(dy);
    __m256 row = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 row_diff = _mm256_mul_ps(row, bdy);
    row_diff = _mm256_floor_ps(row_diff);
    // row_diff = _mm256_mul_ps(row_diff, size);
    printf("%f %f %f %f %f %f %f %f\n", row_diff[0], row_diff[1], row_diff[2], row_diff[3], row_diff[4], row_diff[5], row_diff[6], row_diff[7]);
    __m256 orig1 = _mm256_set1_ps(row_diff[0]*n);
    __m256 rst1 = _mm256_add_ps(orig1, col_diff);
    _mm256_store_ps(to_address, rst1);
    __m256 orig2 = _mm256_set1_ps(row_diff[1]*n);
    __m256 rst2 = _mm256_add_ps(orig2, col_diff);
    _mm256_store_ps(to_address+8, rst2);
    __m256 orig3 = _mm256_set1_ps(row_diff[2]*n);
    __m256 rst3 = _mm256_add_ps(orig3, col_diff);
    _mm256_store_ps(to_address+2*8, rst3);
    __m256 orig4 = _mm256_set1_ps(row_diff[3]*n);
    __m256 rst4 = _mm256_add_ps(orig4, col_diff);
    _mm256_store_ps(to_address+3*8, rst4);
    __m256 orig5 = _mm256_set1_ps(row_diff[4]*n);
    __m256 rst5 = _mm256_add_ps(orig5, col_diff);
    _mm256_store_ps(to_address+4*8, rst5);
    __m256 orig6 = _mm256_set1_ps(row_diff[5]*n);
    __m256 rst6 = _mm256_add_ps(orig6, col_diff);
    _mm256_store_ps(to_address+5*8, rst6);
    __m256 orig7 = _mm256_set1_ps(row_diff[6]*n);
    __m256 rst7 = _mm256_add_ps(orig7, col_diff);
    _mm256_store_ps(to_address+6*8, rst7);
    __m256 orig8 = _mm256_set1_ps(row_diff[7]*n);
    __m256 rst8 = _mm256_add_ps(orig8, col_diff);
    _mm256_store_ps(to_address+7*8, rst8);
    // ---------------------------------- kernel post-processing start -----------------------------
    for (int i = 0; i < 8*8; i++) {
        to[i] = from[(int)to_address[i]];
    }
    // ---------------------------------- kernel post_processing end -----------------------------

    // ---------------------------------- kernel end -----------------------------
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", from[i*n+j]);
        }
        printf("\n");
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%f ", to_address[i*8+j]);
        }
        printf("\n");
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%d ", to[i*8+j]);
        }
        printf("\n");
    }
    free(from);
    free(to);
}
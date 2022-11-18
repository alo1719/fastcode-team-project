#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define MAX_FREQ 4.2
#define BASE_FREQ 1.8

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void kernel8x8(int m, int n, float* to_address, float dx, float dy, __m256 ruler, __m256 col_diff, __m256 row_diff_n) {

    __m256 orig1 = _mm256_set1_ps(row_diff_n[0]);
    __m256 rst1 = _mm256_add_ps(orig1, col_diff);//latency 3 through 1
    _mm256_store_ps(to_address, rst1);
    __m256 orig2 = _mm256_set1_ps(row_diff_n[1]);
    __m256 rst2 = _mm256_add_ps(orig2, col_diff);
    _mm256_store_ps(to_address+8, rst2);
    __m256 orig3 = _mm256_set1_ps(row_diff_n[2]);
    __m256 rst3 = _mm256_add_ps(orig3, col_diff);
    _mm256_store_ps(to_address+2*8, rst3);
    __m256 orig4 = _mm256_set1_ps(row_diff_n[3]);
    __m256 rst4 = _mm256_add_ps(orig4, col_diff);
    _mm256_store_ps(to_address+3*8, rst4);
    __m256 orig5 = _mm256_set1_ps(row_diff_n[4]);
    __m256 rst5 = _mm256_add_ps(orig5, col_diff);
    _mm256_store_ps(to_address+4*8, rst5);
    __m256 orig6 = _mm256_set1_ps(row_diff_n[5]);
    __m256 rst6 = _mm256_add_ps(orig6, col_diff);
    _mm256_store_ps(to_address+5*8, rst6);
    __m256 orig7 = _mm256_set1_ps(row_diff_n[6]);
    __m256 rst7 = _mm256_add_ps(orig7, col_diff);
    _mm256_store_ps(to_address+6*8, rst7);
    orig1 = _mm256_set1_ps(row_diff_n[7]);
    rst1 = _mm256_add_ps(orig1, col_diff);
    _mm256_store_ps(to_address+7*8, rst1);
}


int main(){
    int m=3;
    int n=4;
    int kernel_m=8;
    int kernel_n=8;
    int *from; // m*n
    int *to;// 8*8: no bubble in add, maximum 40*40 
    float *to_address;
    posix_memalign((void**)&from, 32, m*n*sizeof(int));
    posix_memalign((void**)&to, 32, kernel_m*kernel_n*sizeof(int));
    posix_memalign((void**)&to_address, 32, kernel_m*kernel_n*sizeof(float));
    for (int i = 0; i < m*n; i++) {
        from[i] = i;
    }
    // ---------------------------------- kernel start -----------------------------
    float dx = n/8.0;
    float dy = m/8.0;
    __m256 ruler = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 bdx = _mm256_broadcast_ss(&dx);
    __m256 col_diff = _mm256_mul_ps(ruler, bdx);
    col_diff = _mm256_floor_ps(col_diff);//latenty 6 throughput 2
    __m256 bdy = _mm256_broadcast_ss(&dy);
    __m256 row_diff = _mm256_mul_ps(ruler, bdy);
    row_diff = _mm256_floor_ps(row_diff);
    __m256 mul_n = _mm256_set1_ps(n);
    __m256 row_diff_n = _mm256_mul_ps(row_diff, mul_n);
    unsigned long long acc = 0;
    for (int i = 0; i < 100000; i++) {
        unsigned long long start = rdtsc();
        kernel8x8(m, n, to_address, dx, dy, ruler, col_diff, row_diff_n);
        unsigned long long end = rdtsc();
        acc += end - start;
    }
    
    // ---------------------------------- kernel post-processing start -----------------------------
    for (int i = 0; i < kernel_m*kernel_n; i++) {
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
    for (int i = 0; i < kernel_m; i++) {
        for (int j = 0; j < kernel_n; j++) {
            printf("%f ", to_address[i*kernel_n+j]);
        }
        printf("\n");
    }
    for (int i = 0; i < kernel_m; i++) {
        for (int j = 0; j < kernel_n; j++) {
            printf("%d ", to[i*kernel_n+j]);
        }
        printf("\n");
    }

    printf("average cycles: %llu\n", acc/100000);
    printf("%d\t %d\t %lf\n", m, n, (100000.0*kernel_m*kernel_n)/((double)(acc)*MAX_FREQ/BASE_FREQ));

    free(from);
    free(to);
    free(to_address);
}
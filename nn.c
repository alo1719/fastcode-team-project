#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void kernel8x16(float* to_address, __m256 col_diff1, __m256 col_diff2, __m256 row_diff) {
    //1
    __m256 ymm0 = _mm256_broadcast_ss(&row_diff[0]);// latency 5, throuput 0.5, port23
    __m256 ymm1 = _mm256_broadcast_ss(&row_diff[0]);
    //2
    __m256 ymm2 = _mm256_broadcast_ss(&row_diff[1]);
    __m256 ymm3 = _mm256_broadcast_ss(&row_diff[1]);
    //3
    __m256 ymm4 = _mm256_broadcast_ss(&row_diff[2]);
    __m256 ymm5 = _mm256_broadcast_ss(&row_diff[2]);
    //4
    __m256 ymm6 = _mm256_broadcast_ss(&row_diff[3]);
    __m256 ymm7 = _mm256_broadcast_ss(&row_diff[3]);
    //5
    __m256 ymm8 = _mm256_broadcast_ss(&row_diff[4]);
    __m256 ymm9 = _mm256_broadcast_ss(&row_diff[4]);
    //6
    __m256 ymm10 = _mm256_broadcast_ss(&row_diff[5]);
    __m256 ymm11 = _mm256_broadcast_ss(&row_diff[5]);
    ymm0 = _mm256_add_ps(ymm0, col_diff1);// latency 3, throughput 1, port1, the **bottleneck**
    //7
    __m256 ymm12 = _mm256_broadcast_ss(&row_diff[6]);
    __m256 ymm13 = _mm256_broadcast_ss(&row_diff[6]);
    ymm1 = _mm256_add_ps(ymm1, col_diff2);
    //8
    __m256 ymm14 = _mm256_broadcast_ss(&row_diff[7]);
    __m256 ymm15 = _mm256_broadcast_ss(&row_diff[7]);
    ymm2 = _mm256_add_ps(ymm2, col_diff1);
    //9
    ymm3 = _mm256_add_ps(ymm3, col_diff2);
    _mm256_store_ps(to_address, ymm0);// latency 4, throughput 1, port237+port4
    //10
    ymm4 = _mm256_add_ps(ymm4, col_diff1);
    _mm256_store_ps(to_address+8, ymm1);
    //11
    ymm5 = _mm256_add_ps(ymm5, col_diff2);
    _mm256_store_ps(to_address+16, ymm2);
    //12
    ymm6 = _mm256_add_ps(ymm6, col_diff1);
    _mm256_store_ps(to_address+16+8, ymm3);
    //13
    ymm7 = _mm256_add_ps(ymm7, col_diff2);
    _mm256_store_ps(to_address+16*2, ymm4);
    //14
    ymm8 = _mm256_add_ps(ymm8, col_diff1);
    _mm256_store_ps(to_address+16*2+8, ymm5);
    //15
    ymm9 = _mm256_add_ps(ymm9, col_diff2);
    _mm256_store_ps(to_address+16*3, ymm6);
    //16
    ymm10 = _mm256_add_ps(ymm10, col_diff1);
    _mm256_store_ps(to_address+16*3+8, ymm7);
    //17
    ymm11 = _mm256_add_ps(ymm11, col_diff2);
    _mm256_store_ps(to_address+16*4, ymm8);
    //18
    ymm12 = _mm256_add_ps(ymm12, col_diff1);
    _mm256_store_ps(to_address+16*4+8, ymm9);
    //19
    ymm13 = _mm256_add_ps(ymm13, col_diff2);
    _mm256_store_ps(to_address+16*5, ymm10);
    //20
    ymm14 = _mm256_add_ps(ymm14, col_diff1);
    _mm256_store_ps(to_address+16*5+8, ymm11);
    //21
    ymm15 = _mm256_add_ps(ymm15, col_diff2);
    _mm256_store_ps(to_address+16*6, ymm12);
    //22
    _mm256_store_ps(to_address+16*6+8, ymm13);
    //23
    _mm256_store_ps(to_address+16*7, ymm14);
    //24, but can be faster than this!
    _mm256_store_ps(to_address+16*7+8, ymm15);
}


int main(){
    int m=8;
    int n=16;
    int kernel_m=8;
    int kernel_n=16;
    int *from; // m*n
    int *to;// kernel_m*kernel_n
    float *to_address;
    posix_memalign((void**)&from, 32, m*n*sizeof(int));
    posix_memalign((void**)&to, 32, kernel_m*kernel_n*sizeof(int));
    posix_memalign((void**)&to_address, 32, kernel_m*kernel_n*sizeof(float));
    for (int i = 0; i < m*n; i++) {
        from[i] = i;
    }

    // ---------------------------------- kernel start -----------------------------
    float dx = 1.0*n/kernel_n;
    float dy = 1.0*m/kernel_m;
    __m256 ruler1 = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m256 ruler2 = _mm256_set_ps(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0);
    __m256 bdx = _mm256_broadcast_ss(&dx);
    // printf("%f %f %f %f %f %f %f %f", bdx[0], bdx[1], bdx[2], bdx[3], bdx[4], bdx[5], bdx[6], bdx[7]);
    __m256 col_diff1 = _mm256_mul_ps(ruler1, bdx);
    __m256 col_diff2 = _mm256_mul_ps(ruler2, bdx);
    col_diff1 = _mm256_floor_ps(col_diff1);
    col_diff2 = _mm256_floor_ps(col_diff2);
    __m256 bdy = _mm256_broadcast_ss(&dy);
    __m256 row_diff = _mm256_mul_ps(ruler1, bdy);
    row_diff = _mm256_floor_ps(row_diff);
    __m256 mul_n = _mm256_set1_ps(n);
    row_diff = _mm256_mul_ps(row_diff, mul_n);

    unsigned long long acc = 0;

    for (int i = 0; i < 1000000; i++) {
        unsigned long long start = rdtsc();

        kernel8x16(to_address, col_diff1, col_diff2, row_diff);

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

    printf("average cycles: %f\n", (float)acc/1000000);
    // theoretical peak: considering add is the bottleneck, SIMD length=8, throughput=1, TP is 8 FLOPS
    printf("%d\t %d\t %lf\n", kernel_m, kernel_n, (1000000.0*kernel_m*kernel_n)/((double)(acc)*MAX_FREQ/BASE_FREQ));

    free(from);
    free(to);
    free(to_address);
}
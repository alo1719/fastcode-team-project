#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

using namespace cv;
using namespace std;

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

void pre_processing(int *source, int *dest, int row, int col){
    int des = 0;
    int k;
    int p;
    int n;
    int m;
    for (k = 0; k < row; k += 8){
        for (p = 0; p < col; p += 8){
            for (n = 0; n < 8; ++n){
                for (m = 0; m < 8; ++m){
                    dest[des] = source[(k + n) * col + p + m];
                    des += 1;
                }
            }
        }
    }
}

void post_processing(unsigned char *dest, unsigned char *source, int row, int col, int kernel_m, int kernel_n){
    int des = 0;
    int k;
    int p;
    int n;
    int m;
    for (k = 0; k < row; k += kernel_m){
        for (p = 0; p < col; p += kernel_n){
            for (n = 0; n < kernel_m; ++n){
                for (m = 0; m < kernel_n; ++m){
                    dest[(k + n) * col + p + m] = source[des];
                    des += 1;
                }
            }
        }
    }
}

void nn(int *from, int *to, float * to_address, int m, int n, int kernel_m, int kernel_n, float dx, float dy){
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

    kernel8x16(to_address, col_diff1, col_diff2, row_diff);

    for (int i = 0; i < kernel_m*kernel_n; i++) {
        to[i] = from[(int)to_address[i]];
    }
}

void verify(int *a, int m, int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            printf("%d ", a[i*n+j]);
        }
        printf("\n");
    }
}

void verify(unsigned char *a, int m, int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            printf("%d ", int(a[i*n+j]));
        }
        printf("\n");
    }
}

int main(int argc, char** argv )
{
    Mat image;
    image = imread( "./test.jpeg", 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    int image_rows = image.rows, image_cols = image.cols;
    int graph_size = image.cols * image.rows;
    unsigned char *B, *G, *R;
    B = new unsigned char[graph_size];
    G = new unsigned char[graph_size];
    R = new unsigned char[graph_size];
    // cout<<image.cols<<" "<<image.rows<<" "<<image.channels()<<endl;
    int idx = 0;
    for (size_t y = 0; y < image.rows; ++y) {

        unsigned char* row_ptr= image.ptr<unsigned char>(y);
        for (size_t x = 0; x < image.cols; ++x) {

            unsigned char* data_ptr = &row_ptr[x*image.channels()];


            B[idx] = data_ptr[0];
            G[idx] = data_ptr[1];
            R[idx] = data_ptr[2];
            idx++;

        }
    }

    // L1 size: 7168

    cout<<int(B[0])<<" "<<int(G[0])<<" "<<int(R[0])<<endl;

    // int image_rows = 16, image_cols = 32;

    // int B[512]= {
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    // };

    // int B[32]= {
    //     1,2,3,4,5,6,7,8,
    //     1,2,3,4,5,6,7,8,
    //     1,2,3,4,5,6,7,8,
    //     1,2,3,4,5,6,7,8
    // };

    int des_m=80, des_n=160;

    unsigned char *desB, *desG, *desR;
    desB = new unsigned char[des_m*des_n];
    desG = new unsigned char[des_m*des_n];
    desR = new unsigned char[des_m*des_n];
    

    float dx = 1.0*image_rows/des_m;
    float dy = 1.0*image_cols/des_n;

    int kernel_m=8;
    int kernel_n=16;

    int input_row = int(dx*kernel_m);
    int input_col = int(dy*kernel_n);


    
    
    int *from; // m*n
    int *to;// kernel_m*kernel_n
    float *to_address;
    posix_memalign((void**)&from, 32, input_row*input_col*sizeof(int));
    posix_memalign((void**)&to, 32, kernel_m*kernel_n*sizeof(int));
    posix_memalign((void**)&to_address, 32, kernel_m*kernel_n*sizeof(float));

    int des = 0;

    // cout<<image_rows<<" "<<image_cols<<" "<<input_row<<" "<<input_col<<" "<<dx<<" "<<dy<<endl; 
    // just color B   

    for(int i=0; i<image_rows-input_row; i+=input_row){
        for(int j=0; j<image_cols-input_col; j+=input_col){
            int index=0;
            for(int m=0; m<input_row; ++m){
                for(int n=0; n<input_col; ++n){
                    from[index] = int(B[(i+m)*image_cols + j + n]);
                    index++;
                }
            }
            // cout<<des<<endl;
            nn(from, to, to_address, input_row, input_col, kernel_m, kernel_n, dx, dy);
            for(int n=0; n<kernel_m*kernel_n; ++n){
                desB[des] = (unsigned char)(to[n]);
                des++;
            }
            // cout<<i<<" "<<j<<" "<<des<<endl;
        }
    }

    delete B;
    delete G;
    delete R;

    unsigned char *outB, *outG, *outR;
    outB = new unsigned char[des_m*des_n];
    outG = new unsigned char[des_m*des_n];
    outR = new unsigned char[des_m*des_n];

    post_processing(outB, desB, des_m, des_n, kernel_m, kernel_n);

    // verify(B, image_rows, image_cols);

    // printf("\n");

    verify(desB, des_m, des_n);

    printf("\n");

    verify(outB, des_m, des_n);

    delete desB;
    delete desG;
    delete desR;
    delete outB;
    delete outG;
    delete outR;

    free(from);
    free(to);
    free(to_address);
    
    return 0;
}
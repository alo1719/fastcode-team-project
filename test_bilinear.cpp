#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream> 
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include "kernel_driver.h"

using namespace cv;
using namespace std;

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

void eachColorNaive(int image_rows, int image_cols, int des_m, int des_n, unsigned char* color,
               unsigned char* colorDes) {
    float divisionH = 1.0*image_rows/des_m;
    float divisionW = 1.0*image_cols/des_n;
    #pragma omp parallel for
    for(int i=0; i<des_m; i+=1){
        for(int j=0; j<des_n; j+=1){
            float x = (float)i * divisionH;
            float y = (float)j * divisionW;
            float agirlikX = x - (int)x;
            float agirlikY = y - (int)y;
            // printf("%f\t%f\t%f\t%f\t\n", (float)(1 - agirlikX) * (float)(1 - agirlikY), (float)agirlikX * (float)(1 - agirlikY), 
            //         (float)(1 - agirlikX) * (float)agirlikY, (float)agirlikX * (float)agirlikY);
            colorDes[i*des_n + j] = color[(int)x*des_n + (int)y] * (float)(1 - agirlikX) * (float)(1 - agirlikY)
                    + color[((int)x + 1)*des_n + (int)y] * (float)agirlikX * (float)(1 - agirlikY)
                    + color[(int)x*des_n + (int)y + 1] * (float)(1 - agirlikX) * (float)agirlikY
                    + color[((int)x + 1)*des_n + (int)y + 1] * (float)agirlikX * (float)agirlikY;
        }
    }
}

void eachColor(int image_rows, int image_cols, int input_row, int input_col, float* from, unsigned char* color,
               unsigned char* colorDes, float* to, int kernel_m, int kernel_n) {
    int des = 0;
    // #pragma omp parallel for
    int *row_indices = (int *) calloc(8, sizeof(int));
    int *row_indices_plus1 = (int *) calloc(8, sizeof(int));
    float *parameters;
    posix_memalign((void**)&parameters, 32, 8*4*8*sizeof(float));

    bilinear_driver(input_row, input_col, kernel_m, kernel_n, from, to,
            row_indices, row_indices_plus1, parameters);

    // permute mask for floor_y
    __m256i mask_floory = _mm256_setzero_si256();
    // permute mask for floor_y + 1
    __m256i mask_flooryp = _mm256_setzero_si256();
    switch(input_row) {
    case 1:
        mask_floory = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        break;
    case 2:
        mask_floory = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
        break;
    case 3:
        mask_floory = _mm256_set_epi32(2, 2, 1, 1, 1, 0, 0, 0);
        mask_flooryp = _mm256_set_epi32(2, 2, 2, 2, 2, 1, 1, 1);
        break;
    case 4:
        mask_floory = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 1, 1);
        break;
    case 5:
        mask_floory = _mm256_set_epi32(4, 3, 3, 2, 1, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(4, 4, 4, 3, 2, 2, 1, 1);
        break;
    case 6:
        mask_floory = _mm256_set_epi32(5, 4, 3, 3, 2, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(5, 5, 4, 4, 3, 2, 1, 1);
        break;
    case 7:
        mask_floory = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
        mask_flooryp = _mm256_set_epi32(6, 6, 5, 4, 3, 2, 1, 1);
        break;
    default:
        break;
    }
    // print256i_num(mask_floory);
    // print256i_num(mask_flooryp);
    unsigned long long start = rdtsc();
    for(int i=1; i<image_rows+1; i+=input_row){
        for(int j=1; j<image_cols+1; j+=input_col){
            // pack the input matrix needed for one kernel
            int index=0;
            for(int m=0; m<input_row; ++m){
                for(int n=0; n<input_col; ++n){
                    from[index] = float(color[(i+m)*(image_cols+2) + j + n]);
                    index++;
                }
            }
            bilinear_kernel_upscale(row_indices, row_indices_plus1,
                        from, to, kernel_n,
                        mask_floory,  mask_flooryp, parameters);
            // unpack the output matrix
            for(int n=0; n<kernel_m*kernel_n; ++n){
                colorDes[des] = (unsigned char)(int)(to[n]);
                des++;
            }
        }
    }

}

int main(int argc, char** argv)
{

    int des_m = atoi(argv[1]);
    int des_n = atoi(argv[2]);

    Mat image;
    image = imread("./Wechat.jpeg", 1);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }
    int image_rows = image.rows, image_cols = image.cols;
    int graph_size = (image.cols+2) * (image.rows+2);
    unsigned char *B, *G, *R;
    B = new unsigned char[graph_size];
    G = new unsigned char[graph_size];
    R = new unsigned char[graph_size];
    // cout<<image.cols<<" "<<image.rows<<" "<<image.channels()<<endl;
    int idx = image.cols + 2;
    for (size_t y = 0; y < image.rows; ++y) {
        B[idx] = 0; G[idx] = 0; R[idx] = 0; idx++;
        unsigned char* row_ptr= image.ptr<unsigned char>(y);
        for (size_t x = 0; x < image.cols; ++x) {
            unsigned char* data_ptr = &row_ptr[x*image.channels()];
            B[idx] = data_ptr[0];
            G[idx] = data_ptr[1];
            R[idx] = data_ptr[2];
            idx++;
        }
        B[idx] = 0; G[idx] = 0; R[idx] = 0; idx++;
    }
    // idx = 0;
    // for (size_t y = 0; y < image.rows + 2; ++y) {
    //     for (size_t x = 0; x < image.cols + 2; ++x) {
    //         cout<<(int)G[idx++]<<"\t";
    //     }
    //     cout<<endl;
    // }

    unsigned char *desB, *desG, *desR;
    desB = new unsigned char[des_m*des_n];
    desG = new unsigned char[des_m*des_n];
    desR = new unsigned char[des_m*des_n];

    float dx = 1.0*image_rows/des_m;
    float dy = 1.0*image_cols/des_n;

    int kernel_m=8;
    int kernel_n=8;

    int input_row = ceil(dx*kernel_m);
    int input_col = ceil(dy*kernel_n);

    float *from; // des_m*des_n original img
    float *to;// kernel_m*kernel_n img
    posix_memalign((void**)&from, 32, input_row*input_col*sizeof(float));
    posix_memalign((void**)&to, 32, kernel_m*kernel_n*sizeof(float));

    int des = 0;
    unsigned long long start = rdtsc();

    eachColor(image_rows, image_cols, input_row, input_col, from, B, desB, to, kernel_m, kernel_n);
    eachColor(image_rows, image_cols, input_row, input_col, from, G, desG, to, kernel_m, kernel_n);
    eachColor(image_rows, image_cols, input_row, input_col, from, R, desR, to, kernel_m, kernel_n);
    unsigned long long end = rdtsc();
    printf("needed cycles: %f\n", (float)(end - start));

    // unsigned long long start_naive = rdtsc();

    // // performance baseline
    // eachColorNaive(image_rows, image_cols, des_m, des_n, B, desB);
    // eachColorNaive(image_rows, image_cols, des_m, des_n, B, desB);
    // eachColorNaive(image_rows, image_cols, des_m, des_n, B, desB);
    // unsigned long long end_naive = rdtsc();
    // printf("needed cycles: %f\n", (float)(end_naive - start_naive));

    delete B;
    delete G;
    delete R;

    unsigned char *outB, *outG, *outR;
    outB = new unsigned char[des_m*des_n];
    outG = new unsigned char[des_m*des_n];
    outR = new unsigned char[des_m*des_n];

    post_processing(outB, desB, des_m, des_n, kernel_m, kernel_n);
    post_processing(outG, desG, des_m, des_n, kernel_m, kernel_n);
    post_processing(outR, desR, des_m, des_n, kernel_m, kernel_n);

    idx = 0;

    Mat output_image(des_m, des_n, CV_8UC3);

    for (size_t y = 0; y < des_m; ++y) {
        unsigned char* row_ptr= output_image.ptr<unsigned char>(y);
        for (size_t x = 0; x < des_n; ++x) {

            unsigned char* data_ptr = &row_ptr[x*image.channels()];

            data_ptr[0] = outB[idx];
            data_ptr[1] = outG[idx];
            data_ptr[2] = outR[idx];
            idx++;
        }
    }

    imwrite("./output.jpeg", output_image);


    delete desB;
    delete desG;
    delete desR;
    delete outB;
    delete outG;
    delete outR;

    free(from);
    free(to);

    return 0;
}
#include <stdio.h>


void pre_processing(int *a, int *b, int i, int j){
    int des = 0;
    int k;
    int p;
    int n;
    int m;
    for (k = 0; k < i; k += 8){
        for (p = 0; p < j; p += 8){
            for (n = 0; n < 8; ++n){
                for (m = 0; m < 8; ++m){
                    b[des] = a[(k + n) * j + p + m];
                    des += 1;
                }
            }
        }
    }
}

void post_processing(int *a, int *b, int i, int j){
    int des = 0;
    int k;
    int p;
    int n;
    int m;
    for (k = 0; k < i; k += 8){
        for (p = 0; p < j; p += 8){
            for (n = 0; n < 8; ++n){
                for (m = 0; m < 8; ++m){
                    a[(k + n) * j + p + m] = b[des];
                    des += 1;
                }
            }
        }
    }
}

int main(){
    int a[128]= {
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    };
    int b[128] = {};
    int c[128] = {};
    pre_processing(&a, &b, 8, 16);
    int i;
    for (i=0; i<128; ++i){
        printf("%d ", b[i]);
        if ((i+1)%8==0){
            printf("\n");
        }
    }
    printf("\n");
    post_processing(&c, &b, 8, 16);
    for (i=0; i<128; ++i){
        printf("%d ", c[i]);
        if ((i+1)%8==0){
            printf("\n");
        }
    }
    printf("\n");
    return 0;
}
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>

using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    Mat image;
    image = imread( "./test.jpeg", 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    int graph_size = image.cols * image.rows;
    unsigned char B[graph_size], G[graph_size], R[graph_size];
    cout<<image.cols<<" "<<image.rows<<" "<<image.channels()<<endl;
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

    cout<<int(B[0])<<" "<<int(G[0])<<" "<<int(R[0])<<endl;
    
    return 0;
}
#include <immintrin.h>
int main() {
    int new_h = 5;
    int ori_h = 10;
    int new_w = 5;
    int ori_w = 10;
    int *from; // m*n
    int *to;// 8*8

    float height_division = (float) new_h / ori_h;
    float width_division = new_w / ori_w;

    int cur_row = 0;
    float x = cur_row * height_division;
    int floor_x = floor(x);
    int floor_x_idx = floor_x * ori_w;

    int floor_x_plus1 = floor_x + 1;
    int floor_x_plus1_idx = floor_x_plus1 * ori_w;

    float diff_x = x = floor_x;


    __m256 ymm0 = _mm256_broadcast_ps(&width_division);
    __m256 ymm1 = {0, 1, 2, 3, 4, 5, 6, 7};
    ymm0 = _mm256_mul_ps(ymm1, ymm0); //y
    // ymm0 is occupied

    __m256 ymm2 = _mm256_floor_ps(ymm0); //floory
    __m256 ymm3 = _mm256_sub_ps(ymm0, ymm2); //diffy
    __m256 ymm4 = _mm256_set1_ps(diff_x); //diff_x
    // ymm0 ymm2 ymm3 ymm4 are occupied

    ymm1 = _mm256_set1_ps(1);
    __m256 ymm5 = _mm256_sub_ps(ymm1, ymm3); //1 - diffy
    __m256 ymm6 = _mm256_sub_ps(ymm1, ymm4); //1 - diff_x
    // ymm0 ymm1 ymm2 ymm3 ymm4 ymm5 ymm6 are occupied

    __m256 ymm7 = _mm256_mul_ps(ymm6, ymm5); //(1 - diff_x) * (1 - diffy)
    __m256 ymm8 = _mm256_mul_ps(ymm4, ymm5); //diff_x * (1 - diffy)
    // ymm0 ymm1 ymm2 ymm3 ymm4 ymm6 ymm7 ymm8 are occupied

    __m256 ymm9 = _mm256_mul_ps(ymm6, ymm3); // (1 - diff_x) * diffy
    __m256 ymm10 = _mm256_mul_ps(ymm4, ymm3); // diff_x * diffy
    // ymm0 ymm1 ymm2 ymm7 ymm8 ymm9 ymm10 are occupied

    // _mm256_set1_ei32
    ymm3 = _mm256_set1_ps(floor_x_idx); //floor_x
    ymm4 = _mm256_set1_ps(floor_x_plus1_idx); //floor_x + 1
    ymm5 = _mm256_add_ps(ymm2, ymm1); //floory + 1
    // ymm0 ymm1 ymm2 ymm3 ymm4 ymm5 ymm7 ymm8 ymm9 ymm10 are occupied

    // [floor_x, floor_y]
    __m256 ymm11 = _mm256_add_ps(ymm3, ymm2);
    // [floor_x + 1, floor_y]
    __m256 ymm12 = _mm256_add_ps(ymm4, ymm2);
    // [floor_x, floor_y + 1]
    __m256 ymm13 = _mm256_add_ps(ymm3, ymm5);
    // [floor_x + 1, floor_y + 1]
    __m256 ymm14 = _mm256_add_ps(ymm4, ymm5);
    // ymm0 ymm7 ymm8 ymm9 ymm10 ymm11 ymm12 ymm13 ymm14 are occupied

    // now we got all the indices of the needed original elements
    // TODO: int and float types are not consistent due to floor
    ymm1 = _mm256_set_epi8(from[ymm11[0]], from[ymm11[1]], from[ymm11[2]], from[ymm11[3]],
                         from[(int)ymm11[4]], from[ymm11[5]], from[ymm11[6]], from[ymm11[7]]);
    ymm2 = _mm256_set_ps(from[ymm12[0]], from[ymm12[1]], from[ymm12[2]], from[ymm12[3]],
                         from[ymm12[4]], from[ymm12[5]], from[ymm12[6]], from[ymm12[7]]);
    ymm3 = _mm256_set_ps(from[ymm13[0]], from[ymm13[1]], from[ymm13[2]], from[ymm13[3]],
                         from[ymm13[4]], from[ymm13[5]], from[ymm13[6]], from[ymm13[7]]);
    ymm4 = _mm256_set_ps(from[ymm14[0]], from[ymm14[1]], from[ymm14[2]], from[ymm14[3]],
                         from[ymm14[4]], from[ymm14[5]], from[ymm14[6]], from[ymm14[7]]);

    _mm256_fmadd_ps(ymm1, ymm7, ymm1);
    _mm256_fmadd_ps(ymm1, ymm8, ymm1);
    _mm256_fmadd_ps(ymm1, ymm9, ymm1);
    _mm256_fmadd_ps(ymm1, ymm10, ymm1);

    _mm256_fmadd_ps(ymm2, ymm7, ymm2);
    _mm256_fmadd_ps(ymm2, ymm8, ymm2);
    _mm256_fmadd_ps(ymm2, ymm9, ymm2);
    _mm256_fmadd_ps(ymm2, ymm10, ymm2);

    _mm256_fmadd_ps(ymm3, ymm7, ymm3);
    _mm256_fmadd_ps(ymm3, ymm8, ymm3);
    _mm256_fmadd_ps(ymm3, ymm9, ymm3);
    _mm256_fmadd_ps(ymm3, ymm10, ymm3);

    _mm256_fmadd_ps(ymm4, ymm7, ymm4);
    _mm256_fmadd_ps(ymm4, ymm8, ymm4);
    _mm256_fmadd_ps(ymm4, ymm9, ymm4);
    _mm256_fmadd_ps(ymm4, ymm10, ymm4);

    // TODO
    // _mm256_store_ps(to[], ymm7);
    // _mm256_store_ps(to[], ymm7);
    // _mm256_store_ps(to[], ymm7);
    // _mm256_store_ps(to[], ymm7);

}
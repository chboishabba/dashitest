#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vnni_tile_i8(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc
) {
    for (int i = 0; i < M; i++) {
        const int8_t *rowA = A + i * lda;
        int32_t *rowC = C + i * ldc;
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)rowA[k] * (int32_t)B[k * ldb + j];
            }
            rowC[j] += acc;
        }
    }
}

#ifdef __cplusplus
}
#endif

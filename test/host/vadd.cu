
#include <cuda.h>
#include <stdint.h>

extern "C" __global__ void vectorAdd(int *A, int *B, int *C, uint64_t N)
{
    uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

#include <cuda_runtime.h>

#define a(i, j) ((i)*y+(j))
#define b(i, j) ((i)*z+(j))
#define c(i, j) ((i)*z+(j))

extern "C" __global__ void matrixMul(int *A, int *B, int *C, int x, int y, int z)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cul = blockIdx.x*blockDim.x + threadIdx.x;
	int k, val;

	if( row<x && cul<z )
	{
		val = 0;
		for(k=0; k<y; k++)
			val += A[a(row, k)]*B[b(k, cul)];
		C[c(row, cul)] = val;
	}
}

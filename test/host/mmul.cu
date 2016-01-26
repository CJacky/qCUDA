#include <cuda_runtime.h>

#define a(i, j) ((i)*s+(j))
#define b(i, j) ((i)*s+(j))
#define c(i, j) ((i)*s+(j))

extern "C"{

__global__ void matrixMul(int *A, int *B, int *C, int s)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cul = blockIdx.x*blockDim.x + threadIdx.x;
	int k, val;

	if( row<s && cul<s )
	{
		val = 0;
		for(k=0; k<s; k++)
			val += A[a(row, k)]*B[b(k, cul)];
		C[c(row, cul)] = val;
	}
}
}

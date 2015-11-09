#include <cuda_runtime.h>

/*
#include <sys/time.h>
#define time_define() struct timeval timeval_begin
#define time_begin() gettimeofday (&timeval_begin, NULL);
#define time_end() ({ \
	struct timeval timeval_end; \
	gettimeofday (&timeval_end, NULL); \
	(double)((timeval_end.tv_sec-timeval_begin.tv_sec)+((timeval_end.tv_usec-timeval_begin.tv_usec)/1000000.0)); \
	})
*/

#define m(i, j) ((i)*y+(j))

extern "C" __global__ void matrixAdd(int *A, int *B, int x, int y)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cul = blockIdx.x*blockDim.x + threadIdx.x;

	if( row<x && cul<y )
	{
		A[m(row, cul)] += B[m(row, cul)];
	}
}

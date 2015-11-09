#include <stdio.h>
#include <stdlib.h>
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

#ifndef DUMP_FILE
#define DUMP_FILE 0
#endif

#define m(i, j) ((i)*y+(j))

__global__ void matrixAdd(int *A, int *B, int x, int y)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cul = blockIdx.x*blockDim.x + threadIdx.x;

	if( row<x && cul<y )
	{
		A[m(row, cul)] += B[m(row, cul)];
	}
}

void print_matrix(int *M, int x, int y, FILE *f)
{
	int i, j;

	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
		{
			fprintf(f, "%2d ", M[m(i,j)]);
		}
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
}

int main(int argc, char* argv[])
{
	int x, y, n; // matrix dim  xy, yz, xz
	size_t i, j;
	int *A_h, *A_d;
	int *B_h, *B_d;

	n = (argc>=2)? atoi(argv[1]):2;
	x = (argc>=3)? atoi(argv[2]):3;
	y = (argc>=4)? atoi(argv[3]):3;

	if( x<=0 || y<=0 || n<2){
		printf("dim error, n= %d, x= %d, y= %d\n", n, x, y);
		return -1;
	}

	// malloc matrix memory in host
	A_h = (int*)malloc( x*y*sizeof(int));
	B_h = (int*)malloc( x*y*sizeof(int));

	// malloc matrix memory in device
	cudaMalloc(&A_d, x*y*sizeof(int));
	cudaMalloc(&B_d, x*y*sizeof(int));

	// copy matrix from host to device
	for(j=0; j<x*y; j++) A_h[j] = rand()%10;
	cudaMemcpy(A_d, A_h, x*y*sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads(32, 32);
	dim3 blocks( (x%32)?x/32+1:x/32, (y%32)?y/32+1:y/32);
	
	for(i=1; i<n; i++)
	{
		for(j=0; j<x*y; j++) B_h[j] = rand()%10;
		cudaMemcpy(B_d, B_h, x*y*sizeof(int), cudaMemcpyHostToDevice);
		matrixAdd <<< blocks, threads >>>(A_d, B_d, x, y);
	}

	cudaMemcpy(A_h, A_d, x*y*sizeof(int), cudaMemcpyDeviceToHost);
#if DUMP_FILE
	FILE *f = fopen("madd_gpu_out", "w");
	print_matrix(A_h, x, y, f);
	fclose(f);
#endif

	cudaFree(A_d);
	cudaFree(B_d);


	free(A_h);
	free(B_h);

	return 0;
}

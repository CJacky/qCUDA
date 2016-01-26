#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"

#define a(i, j) ((i)*s+(j))
#define b(i, j) ((i)*s+(j))
#define c(i, j) ((i)*s+(j))

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

int main(int argc, char* argv[])
{
	int i;
	int *A_h, *A_d;
	int *B_h, *B_d;
	int *C_h, *C_d;
	uint64_t s = 0;

	for(i=0; i<strlen(argv[1]); i++)
		s = s*10 + (argv[1][i]-'0');

	// malloc matrix memory in host
	A_h = (int*)malloc( s*s*sizeof(int));
	B_h = (int*)malloc( s*s*sizeof(int));
	C_h = (int*)malloc( s*s*sizeof(int));

	// malloc matrix memory in device
	time_begin();
	cudaMalloc((void**)&A_d, s*s*sizeof(int));
	cudaMalloc((void**)&B_d, s*s*sizeof(int));
	cudaMalloc((void**)&C_d, s*s*sizeof(int));
	time_end();

	// init matrix 
	for(i=0; i<s*s; i++) A_h[i] = rand()%10;
	for(i=0; i<s*s; i++) B_h[i] = rand()%10;
	//for(int i=0; i<x*z; i++) C_h[i] = 0;

	// copy matrix from host to device
	time_begin();
	cudaMemcpy(A_d, A_h, s*s*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, s*s*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, x*z*sizeof(int), cudaMemcpyHostToDevice);
	time_end();

	dim3 blocks( ((s%32)?s/32+1:s/32), ((s%32)?s/32+1:s/32), 1);
	dim3 thread(32, 32, 1);

	time_begin();
	matrixMul<<<blocks, thread>>>(A_d, B_d, C_d, s);
	cudaDeviceSynchronize();
	time_end();

	time_begin();
	cudaMemcpy(C_h, C_d, s*s*sizeof(int), cudaMemcpyDeviceToHost);
	time_end();

	//********************************************************************
	/*
	int j;
	printf("\n");
	for(i=0; i<s; i++){
		for(j=0; j<s; j++)
			printf("%3d ", A_h[a(i,j)]);
		
		printf("  ");
		
		for(j=0; j<s; j++)
			printf("%3d ", B_h[a(i,j)]);
		
		printf("  ");
		
		for(j=0; j<s; j++)
			printf("%3d ", C_h[a(i,j)]);
		
		printf("\n");
	}
	*/
	time_begin();
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	time_end();

	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}

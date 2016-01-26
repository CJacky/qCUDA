#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "common.h"

__global__ void vectorAdd(int *A, int *B, int *C, uint64_t N)
{
    uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
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
	A_h = (int*)malloc( s*sizeof(int));
	B_h = (int*)malloc( s*sizeof(int));
	C_h = (int*)malloc( s*sizeof(int));

	// malloc matrix memory in device
	time_begin();
	cudaMalloc((void**)&A_d, s*sizeof(int));
	cudaMalloc((void**)&B_d, s*sizeof(int));
	cudaMalloc((void**)&C_d, s*sizeof(int));
	time_end();

	// init matrix 
	for(i=0; i<s; i++) A_h[i] = rand()%10;
	for(i=0; i<s; i++) B_h[i] = rand()%10;
	//for(int i=0; i<x*z; i++) C_h[i] = 0;

	// copy matrix from host to device
	time_begin();
	cudaMemcpy(A_d, A_h, s*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, s*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, x*z*sizeof(int), cudaMemcpyHostToDevice);
	time_end();

	dim3 blocks( ((s%1024)?s/1024+1:s/1024), 1, 1);
	dim3 thread( 1024, 1, 1);
	
	time_begin();
	vectorAdd<<<blocks, thread>>>(A_d, B_d, C_d, s);
	cudaDeviceSynchronize();
	time_end();

	time_begin();
	cudaMemcpy(C_h, C_d, s*sizeof(int), cudaMemcpyDeviceToHost);
	time_end();

	//********************************************************************
	/*
	printf("\n");
	
	for(i=0; i<s; i++) printf("%3d ", A_h[i]);
	printf("\n");
		
	for(i=0; i<s; i++) printf("%3d ", B_h[i]);
	printf("\n");
		
	for(i=0; i<s; i++) printf("%3d ", C_h[i]);
	printf("\n");
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "aio.h"

void vadd(uint64_t s)
{
	int i, j;
	int *A_h, *A_d;
	int *B_h, *B_d;
	int *C_h, *C_d;
	void **args;

	printf("%"PRIu64" ", s);
	
	cudaInit();

	cudaRegFunc("vadd.fatbin", "vectorAdd");
	
	// malloc matrix memory in host
	A_h = (int*)malloc( s*sizeof(int));
	B_h = (int*)malloc( s*sizeof(int));
	C_h = (int*)malloc( s*sizeof(int));


	// malloc matrix memory in device
	time_begin();
	cudaMalloc((void**)&A_d, s*sizeof(int));
	cudaMalloc((void**)&B_d, s*sizeof(int));
	cudaMalloc((void**)&C_d, s*sizeof(int));
	printf("%u ", time_end());

	// init matrix 
	for(i=0; i<s; i++) A_h[i] = rand()%10;
	for(i=0; i<s; i++) B_h[i] = rand()%10;
	//for(int i=0; i<x*z; i++) C_h[i] = 0;

	// copy matrix from host to device
	time_begin();
	cudaMemcpy(A_d, A_h, s*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, s*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, x*z*sizeof(int), cudaMemcpyHostToDevice);
	printf("%u ", time_end());

	args = (void**)malloc(4*sizeof(void*));
	args[0] = &A_d;
	args[1] = &B_d;
	args[2] = &C_d;
	args[3] = &s;


	cudaExecFunc(((s%1024)?s/1024+1:s/1024), 1, 1, 1024, 1, 1, 0, args);

	time_begin();
	cudaMemcpy(C_h, C_d, s*sizeof(int), cudaMemcpyDeviceToHost);
	printf("%u ", time_end());

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
	printf("%u ", time_end());

	free(A_h);
	free(B_h);
	free(C_h);
	
	cudaFini();
	printf("\n");
}

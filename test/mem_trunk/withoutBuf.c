// includes
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "common.h"

void main(int argc, char* argv[])
{
	uint64_t size, size2, idx, len;
	int **T, *D, *D2;
	int i, j;

	size = 0;
	if(argc==2)
	{
		for(i=0; i<strlen(argv[1]); i++)
			size = size*10 + (argv[1][i]-'0');
	}
	else
	{
		size = 10;
	}

	printf("%"PRIu64" ", size);

	T = (int**)malloc( ((size%TRUNK_SIZE)? (size/TRUNK_SIZE+1):(size/TRUNK_SIZE)) *sizeof(int*));
	size2 = size;
	idx = 0;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		T[i] = (int*)malloc(len*sizeof(int));
		for(j=0; j<len; j++)
		{
			T[i][j] = hash(idx);
			idx++;
		}
		size2 -= len;
	}
	
	cudaMalloc((void**)&D, sizeof(int)*size);

	/***************************************************************/
	// H2D
	time_begin();

	D2 = D;
	size2 = size;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		cudaMemcpy(D2, T[i], len*sizeof(int), cudaMemcpyHostToDevice);
		size2 -= len;
		D2 += len;
	}

	printf("%u ", time_end());

	/***************************************************************/
	// reset
	size2 = size;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		for(j=0; j<len; j++)
		{
			T[i][j] = 0;
		}
		size2 -= len;
	}
	/***************************************************************/
	// D2H
	time_begin();
	
	D2 = D;
	size2 = size;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		cudaMemcpy(T[i], D2, len*sizeof(int), cudaMemcpyDeviceToHost);
		size2 -= len;
		D2 += len;
	}
	
	printf("%u ", time_end());

	/***************************************************************/
	// check
	size2 = size;
	idx = 0;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		for(j=0; j<len; j++)
		{
			if(T[i][j]!=hash(idx))
				printf("error %"PRIu64"\n", idx);
			idx++;
		}
		size2 -= len;
	}

	size2 = size;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		free(T[i]);
		size2 -= len;
	}
	free(T);

	cudaFree(D);
	
	printf("\n");
}

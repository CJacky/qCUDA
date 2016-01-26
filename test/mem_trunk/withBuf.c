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
	uint8_t **T, *D, *B, *B2;
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

	T = (uint8_t**)malloc( ((size%TRUNK_SIZE)? (size/TRUNK_SIZE+1):(size/TRUNK_SIZE)) *sizeof(uint8_t*));
	size2 = size;
	idx = 0;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		T[i] = (uint8_t*)malloc(len*sizeof(uint8_t));
		for(j=0; j<len; j++)
		{
			T[i][j] = hash(idx);
			idx++;
		}
		size2 -= len;
	}
	
	cudaMalloc((void**)&D, sizeof(uint8_t)*size);

	/***************************************************************/
	// H2D
	time_begin();

	size2 = size;
	B = (uint8_t*)malloc(size*sizeof(uint8_t));
	B2 = B;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		memcpy(B2, T[i], len*sizeof(uint8_t));
		B2 += len;
		size2 -= len;
	}
	
	cudaMemcpy(D, B, size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	free(B);

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
	
	B = (uint8_t*)malloc(size*sizeof(uint8_t));
	cudaMemcpy(B, D, size*sizeof(uint8_t), cudaMemcpyDeviceToHost);

	size2 = size;
	B2 = B;
	for(i=0; size2>0; i++)
	{
		len = MIN(size2, TRUNK_SIZE);
		memcpy(T[i], B2, len*sizeof(uint8_t));
		size2 -= len;
		B2 += len;
	}
	free(B);
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
			{
				printf("error %"PRIu64", %d, %d\n", idx, T[i][j], (int)hash(idx));
			}
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

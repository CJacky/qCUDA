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
	uint64_t size;
	uint8_t *H, *D;
	int i;

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
	
	H = (uint8_t*)malloc(sizeof(uint8_t)*size);
	
	cudaMalloc((void**)&D, sizeof(uint8_t)*size);

	for(i=0; i<size; i++)
	{
		H[i]=hash(i);
	}

	/***************************************************************/
	// H2D
	time_begin();
	cudaMemcpy(D, H, size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	printf("%u ", time_end());

	/***************************************************************/
	// reset
	for(i=0; i<size; i++)
	{
		H[i]=0;
	}

	/***************************************************************/
	// D2H
	time_begin();
	cudaMemcpy(H, D, size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	printf("%u ", time_end());

	/***************************************************************/
	// check
	for(i=0; i<size; i++)
	{
		if(H[i]!=hash(i))
			printf("error %d\n", i);
	}

	free(H);
	cudaFree(D);
	
	printf("\n");
}

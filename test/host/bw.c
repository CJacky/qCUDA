// includes
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>
#include <sys/time.h>
#include "aio.h"


void bw(uint64_t size)
{
	uint8_t *H, *D;
	int i;

	printf("%"PRIu64" ", size);
	
	cudaInit();

	printf("0 "); // reg func

	H = (uint8_t*)malloc(sizeof(uint8_t)*size);
	
	time_begin();
	cudaMalloc((void**)&D, sizeof(uint8_t)*size);
	printf("%u ", time_end());

	for(i=0; i<size; i++)
	{
		H[i]=i%255;
	}

	time_begin();
	cudaMemcpy(D, H, size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	printf("%u ", time_end());

	printf("0 "); // exec kernel

	for(i=0; i<size; i++)
	{
		H[i]=0;
	}

	time_begin();
	cudaMemcpy(H, D, size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	printf("%u ", time_end());


	for(i=0; i<size; i++)
	{
		if(H[i]!=i%255)
			printf("error %d\n", i);
	}

	free(H);
	
	time_begin();
	cudaFree(D);
	printf("%u ", time_end());
	
	cudaFini();
	printf("\n");
}

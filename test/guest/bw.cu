#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"

int main(int argc, char* argv[])
{
	uint8_t *H, *D;
	uint64_t size = 0, err = 0;
	int i;

	printf("0 "); //reg

	for(i=0; i<strlen(argv[1]); i++)
		size = size*10 + (argv[1][i]-'0');

	H = (uint8_t*)malloc(sizeof(uint8_t)*size);
	
	time_begin();
	cudaMalloc((void**)&D, sizeof(uint8_t)*size);
	time_end();

	for(i=0; i<size; i++)
	{
		H[i] = i%255;
	}

	time_begin();
	cudaMemcpy(D, H, size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	time_end();

	for(i=0; i<size; i++)
	{
		H[i]=0;
	}

	printf("0 "); //launch

	time_begin();
	cudaMemcpy(H, D, size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	time_end();

	for(i=0; i<size; i++)
	{
		if(H[i] != i%255)
			err++;
	}

	if(err)
		fprintf(stderr, "bw %"PRIu64" error %"PRIu64"\n", size, err);

	free(H);
	
	time_begin();
	cudaFree(D);
	time_end();

	return 0;
}

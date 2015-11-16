#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
	int devNum;

	cudaGetDeviceCount(&devNum);

	printf("there are %d gpus\n", devNum);
	cudaSetDevice(0);
	cudaDeviceReset();
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define m(i, j) ((i)*y+(j))

#ifndef DUMP_FILE
#define DUMP_FILE 0
#endif

CUdevice cuda_device;
CUcontext cuda_context;
CUmodule cuda_module;
CUfunction cuda_function;

#if 1
#define checkCudaErrors(err)  __checkCudaErrors(err, __LINE__) 
void __checkCudaErrors(CUresult err, const int line)
{
	char *str;
	if ( err != CUDA_SUCCESS )
	{   
		cuGetErrorName(err, (const char**)&str);
		printf("CUDA Driver API error = %04d \"%s\" line %d\n", err, str, line);
		cuCtxDestroy(cuda_context);
		exit(-1);
	}   
}
#else
#define checkCudaErrors(err) 
#endif

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

void initCUDA()
{	
	/*
	int major = 0, minor = 0;
	char name[100];
	size_t     totalGlobalMem;
	*/

	checkCudaErrors( cuInit(0) );
	checkCudaErrors( cuDeviceGet(&cuda_device, 0) );
/*
	cuDeviceGetName(name, 100, cuda_device);
	printf("Using device 0: %s\n", name);

	checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuda_device) );
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, cuda_device) );
	printf("  Total amount of global memory:   %llu bytes\n", 
			(unsigned long long)totalGlobalMem);
	printf("  64-bit Memory Address:       %s\n",
			(totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?				              "YES" : "NO");
*/
	checkCudaErrors( cuCtxCreate(&cuda_context, 0, cuda_device) );
	checkCudaErrors( cuModuleLoad( &cuda_module, "madd_gpu_drv.cubin" ));
	checkCudaErrors( cuModuleGetFunction(&cuda_function, cuda_module, "matrixAdd") );
}

void finiCUDA()
{
	checkCudaErrors( cuCtxDestroy(cuda_context));
}

int main(int argc, char* argv[])
{
	CUdeviceptr A_d,  B_d;
	int        *A_h, *B_h;
	int x, y, n; // matrix dim  xy, yz, xz
	size_t i, j;
	unsigned int threads[3], blocks[3], sharedMem;
	void* para[4];

	initCUDA();
	n = (argc>=2)? atoi(argv[1]):2;
	x = (argc>=3)? atoi(argv[2]):3;
	y = (argc>=4)? atoi(argv[3]):3;

	if( x<=0 || y<=0 || n<2){
		printf("dim error, n= %d, x= %d, y= %d\n", n, x, y);
		return -1;
	}

	A_h = (int*)malloc( x*y*sizeof(int));
	B_h = (int*)malloc( x*y*sizeof(int));
	
	checkCudaErrors( cuMemAlloc( &A_d, x*y*sizeof(int)));	
	checkCudaErrors( cuMemAlloc( &B_d, x*y*sizeof(int)));	
	
	for(j=0; j<x*y; j++) A_h[j] = rand()%10;
	checkCudaErrors( cuMemcpyHtoD( A_d, A_h, x*y*sizeof(int)) );
	
	threads[0] = 32;
	threads[1] = 32;
	threads[2] = 1;

	blocks[0] = (x%32)? x/32+1 : x/32;
	blocks[1] = (y%32)? y/32+1 : y/32;
	blocks[2] = 1;

	sharedMem = 0;

	para[0] = &A_d;
	para[1] = &B_d;
	para[2] = &x;
	para[3] = &y;
/*
	printf("%d %d %d %d %d %d\n",
			threads[0], threads[1], threads[2],
			blocks [0], blocks [1], blocks [2]);
*/

	for(i=1; i<n; i++)
	{
		for(j=0; j<x*y; j++) B_h[j] = rand()%10;
		checkCudaErrors( cuMemcpyHtoD( B_d, B_h, x*y*sizeof(int)) );
		checkCudaErrors( cuLaunchKernel(cuda_function, 
					blocks [0], blocks [1], blocks [2], 
					threads[0], threads[1], threads[2],
					sharedMem, NULL, para, NULL));
	}
	checkCudaErrors( cuMemcpyDtoH( A_h, A_d, x*y*sizeof(int)) );
#if DUMP_FILE
	FILE *f = fopen("madd_gpu_drv_out", "w");
	print_matrix(A_h, x, y, f);
	fclose(f);
#endif
	
	free(A_h);
	free(B_h);

	checkCudaErrors( cuMemFree( A_d ));
	checkCudaErrors( cuMemFree( B_d ));

	finiCUDA();

	return 0;
}

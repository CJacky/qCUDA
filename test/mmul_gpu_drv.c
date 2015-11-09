#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

typedef int elem_t;
#define a(i, j) ((i)*y+(j))
#define b(i, j) ((i)*z+(j))
#define c(i, j) ((i)*z+(j))

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
	checkCudaErrors( cuModuleLoad( &cuda_module, "mmul_gpu_drv.cubin" ));
	checkCudaErrors( cuModuleGetFunction(&cuda_function, cuda_module, "matrixMul") );
}

void finiCUDA()
{
	checkCudaErrors( cuCtxDestroy(cuda_context));
}

int main(int argc, char* argv[])
{
	CUdeviceptr A_d,  B_d,  C_d;
	elem_t     *A_h, *B_h, *C_h;
	int x, y, z; // matrix dim  xy, yz, xz
	int i;
	unsigned int threads[3], blocks[3], sharedMem;
	void* para[6];

	initCUDA();

	x = (argc>=2)? atoi(argv[1]):3;
	y = (argc>=3)? atoi(argv[2]):3;
	z = (argc>=4)? atoi(argv[3]):3;


	if( x<=0 || y<=0 || z<=0){
		printf("dim error, x= %d, y= %d, z= %d\n", x, y, z);
		return -1;
	}

	A_h = (elem_t*)malloc( x*y*sizeof(elem_t));
	B_h = (elem_t*)malloc( y*z*sizeof(elem_t));
	C_h = (elem_t*)malloc( x*z*sizeof(elem_t));
	
	for(i=0; i<x*y; i++) A_h[i] = rand()%10;
	for(i=0; i<y*z; i++) B_h[i] = rand()%10;
	

	checkCudaErrors( cuMemAlloc( &A_d, x*y*sizeof(elem_t)));	
	checkCudaErrors( cuMemAlloc( &B_d, y*z*sizeof(elem_t)));	
	checkCudaErrors( cuMemAlloc( &C_d, x*z*sizeof(elem_t)));	

	checkCudaErrors( cuMemcpyHtoD( A_d, A_h, x*y*sizeof(elem_t)) );
	checkCudaErrors( cuMemcpyHtoD( B_d, B_h, y*z*sizeof(elem_t)) );

	threads[0] = 32;
	threads[1] = 32;
	threads[2] = 1;

	blocks[0] = (x%32)? x/32+1 : x/32;
	blocks[1] = (z%32)? z/32+1 : z/32;
	blocks[2] = 1;

	sharedMem = 0;

	para[0] = &A_d;
	para[1] = &B_d;
	para[2] = &C_d;
	para[3] = &x;
	para[4] = &y;
	para[5] = &z;
/*
	printf("%d %d %d %d %d %d\n",
			threads[0], threads[1], threads[2],
			blocks [0], blocks [1], blocks [2]);
*/
	checkCudaErrors( cuLaunchKernel(cuda_function, 
				blocks [0], blocks [1], blocks [2], 
				threads[0], threads[1], threads[2],
				sharedMem, NULL, para, NULL)
			);

	checkCudaErrors( cuMemcpyDtoH( C_h, C_d, x*z*sizeof(elem_t)) );

//********************************************************************
#if DUMP_FILE
	FILE *f = fopen("mmul_gpu_drv_out", "w");
	int j;

	fprintf(f, "%d %d %d\n", x, y, z);
	for(i=0; i<((x>y)?x:y); i++)
	{
		for(j=0; j<y; j++){
			if(i<x)	fprintf(f, "%2d ", A_h[a(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "  ");

		for(j=0; j<z; j++){
			if(i<y) fprintf(f, "%2d ", B_h[b(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "    ");

		for(j=0; j<z; j++){
			if(i<x) fprintf(f, "%4d ", C_h[c(i,j)]);
			else    fprintf(f, "%4c ", ' ');
		}
		fprintf(f, "\n");
	}
	fclose(f);
#endif
//********************************************************************
	
	free(A_h);
	free(B_h);
	free(C_h);

	checkCudaErrors( cuMemFree( A_d ));
	checkCudaErrors( cuMemFree( B_d ));
	checkCudaErrors( cuMemFree( C_d ));

	finiCUDA();

	return 0;
}

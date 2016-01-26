#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "aio.h"

CUdevice cudaDevice;
CUcontext cudaContext;
CUmodule cudaModule;

#define cudaFunctionMaxNum 8
CUfunction cudaFunction[cudaFunctionMaxNum];
uint32_t cudaFunctionId[cudaFunctionMaxNum];
uint32_t cudaFunctionNum;

#define cudaEventMaxNum 16
cudaEvent_t cudaEvent[cudaEventMaxNum];
uint32_t cudaEventNum;

#define cudaStreamMaxNum 32
cudaStream_t cudaStream[cudaStreamMaxNum];
uint32_t cudaStreamNum;


#define CEE(err) __cudaErrorCheck(err, __LINE__, __FILE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line, const char* file)
{
	char *str;
	if ( err != cudaSuccess )
	{
		str = (char*)cudaGetErrorString(err);
		printf("CUDA Runtime API error = %04d \"%s\" line %d, file %s\n", err, str, line, file);
		exit(1);
	}
}

void cudaInit()
{
	uint32_t i;

	for(i=0; i<cudaFunctionMaxNum; i++)
		memset(&cudaFunction[i], 0, sizeof(CUfunction));

	for(i=0; i<cudaEventMaxNum; i++)
		memset(&cudaEvent[i], 0, sizeof(cudaEvent_t));

	for(i=0; i<cudaStreamMaxNum; i++)
		memset(&cudaStream[i], 0, sizeof(cudaStream_t));

	time_begin();
	CEE(cuInit(0));
	CEE(cuDeviceGet(&cudaDevice, 0));
	CEE(cuCtxCreate(&cudaContext, 0, cudaDevice));
	printf("%u ", time_end());

	cudaFunctionNum = 0;
	cudaEventNum = 0;
}

void cudaFini()
{
	uint32_t i;

	for(i=0; i<cudaEventMaxNum; i++)
	{
		if( cudaEvent[i] != 0 ){
			cudaEventDestroy(cudaEvent[i]);
		}
	}
	time_begin();
	CEE(cuCtxDestroy(cudaContext));
	printf("%u ", time_end());
}

void cudaRegFunc(char *path, char *name)
{
	FILE *fp = fopen(path, "rb");
	int file_size;
	char *buf;

	if(fp==NULL)
		printf("file open fail\n");

	fseek(fp, 0, SEEK_END);
	file_size=ftell(fp);
	buf = (char*)malloc(file_size+1);
	fseek(fp, 0, SEEK_SET);

	fread(buf, sizeof(char), file_size, fp);
	fclose(fp);
	buf[file_size]='\0';

	time_begin();
	CEE(cuModuleLoadData(&cudaModule, buf));
	CEE(cuModuleGetFunction(&cudaFunction[cudaFunctionNum], cudaModule, name));
	printf("%u ", time_end());
	cudaFunctionNum++;
}

void cudaExecFunc(int bx, int by, int bz, int tx, int ty, int tz, int sm, void **args)
{
	time_begin();
	CEE(cuLaunchKernel(cudaFunction[0], bx, by, bz, tx, ty, tz, sm, NULL, args, NULL));
	CEE(cuCtxSynchronize());
	printf("%u ", time_end());
}

void aio(void)
{
	uint64_t i, j;
	
	printf("bw\n");
	for(i=0; i<10; i++)
		for(j=1024; j<=1073741824; j*=2)
		{
			fprintf(stderr, "bw %"PRIu64" %"PRIu64"\n", i, j);
			bw(j);
		}
	
	printf("\nmmul\n");
	for(i=0; i<10; i++)
		for(j=32; j<=4096; j*=2)
		{
			fprintf(stderr, "mmul %"PRIu64" %"PRIu64"\n", i, j);
			mmul(j);
		}
	
	printf("\nvadd\n");
	for(i=0; i<10; i++)
		for(j=1048576; j<=268435456; j*=2)
		{
			fprintf(stderr, "vadd %"PRIu64" %"PRIu64"\n", i, j);
			vadd(j);
		}
}

void rcuda(void)
{
	uint64_t i, j;
	struct timeval rb, re;
/*	
	printf("### bw\n");
	for(i=0; i<10; i++)
		for(j=1024; j<=1073741824; j*=2)
		{
			fprintf(stderr, "bw %"PRIu64" %"PRIu64"\n", i, j);
	
			gettimeofday (&rb, NULL);
			bw(j);
			gettimeofday (&re, NULL);
			printf("### %"PRIu64" %u\n", j, (unsigned int)((re.tv_sec  - rb.tv_sec)*1000000 + 
						(re.tv_usec - rb.tv_usec)) );
		}
*/	
	printf("\n### mmul\n");
	for(i=0; i<10; i++)
		for(j=32; j<=4096; j*=2)
		{
			fprintf(stderr, "mmul %"PRIu64" %"PRIu64"\n", i, j);
	
			gettimeofday (&rb, NULL);
			mmul(j);
			gettimeofday (&re, NULL);
			printf("### %"PRIu64" %u\n", j, (unsigned int)((re.tv_sec  - rb.tv_sec)*1000000 + 
						(re.tv_usec - rb.tv_usec)) );
		}
	
	printf("\n### vadd\n");
	for(i=0; i<10; i++)
		for(j=1048576; j<=268435456; j*=2)
		{
			fprintf(stderr, "vadd %"PRIu64" %"PRIu64"\n", i, j);
	
			gettimeofday (&rb, NULL);
			vadd(j);
			gettimeofday (&re, NULL);
			printf("### %"PRIu64" %u\n", j, (unsigned int)((re.tv_sec  - rb.tv_sec)*1000000 + 
						(re.tv_usec - rb.tv_usec)) );
		}
}
int main(int argc, char* argv[])
{
	int i;
	uint64_t size=0; 
	
	printf("####### ");
	bw(10);

	if( !strcmp(argv[1], "single") )
	{
		aio();
	}
	else if( !strcmp(argv[1], "rcuda") )
	{
	
		rcuda();
	}
	else 
	{
		for(i=0; i<strlen(argv[2]); i++)
			size = size*10 + (argv[2][i]-'0');

		//printf("%s %"PRIu64"\n", argv[1], size);
		if( !strcmp(argv[1], "bw") )
		{
			for(i=0; i<10; i++)
				bw(size);
		}
		else if( !strcmp(argv[1], "mmul") )
		{
			for(i=0; i<10; i++)
				mmul(size);
		}
		else if( !strcmp(argv[1], "vadd") )
		{
			for(i=0; i<10; i++)
				vadd(size);
		}
	}
	return 0;
}


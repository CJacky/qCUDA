#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <elf.h>

#include <cuda.h>
#include <builtin_types.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

#include "../vhm-driver/virtio_hm.h"

#define Errchk(ans) { DrvAssert((ans), __FILE__, __LINE__); }
inline void DrvAssert( CUresult code, const char *file, int line)
{
	char *str;
	if (code != CUDA_SUCCESS) {
		cuGetErrorName(code, (const char**)&str);
		printf("Error: %s , %s@%d\n", str, file, line);
		exit(code);
	}/* else {
		std::cout << "Success: " << file << "@" << line << std::endl;
		}*/
}

void *lib = NULL;
CUmodule mod;
CUdevice dev;
CUcontext ctx;
CUfunction fun;
void *paraBuf[1024];
size_t paraSize=0;
dim3 gridDim;
dim3 blockDim;
size_t sharedMem;
cudaStream_t stream;

void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic = *(unsigned int*)fatCubin;
	void **fatCubinHandle = malloc(sizeof(void*));

	printf("### %s ###\n", __FILE__);
	printf("%s\n", __func__);
	printf("    fatCubin= %p\n", fatCubin);

	if( magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		printf("    FATBINC_MAGIC\n");
		printf("    magic= %x\n", binary->magic);
		printf("    version= %x\n", binary->version);
		printf("    data= %p\n", binary->data);
		printf("    filename_or_fatbins= %p\n", binary->filename_or_fatbins);

		// TODO
		// add 0x50 is result of researching hexdump of binary file
		// it should be  some data struct element
		*fatCubinHandle = (void*)binary->data;// + 0x50;
	}
	else 
	{
#if 0	
magic: __cudaFatFormat.h
		   header: __cudaFatMAGIC)
		   __cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;

magic: FATBIN_MAGIC
		   header: fatbinary.h
		   computeFatBinaryFormat_t binary = (computeFatBinaryFormat_t)fatCubin;
#endif	 
	   printf("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
	   exit(1);
	}

	Errchk( cuInit(0));
	Errchk( cuDeviceGet(&dev, 0));
	Errchk( cuCtxCreate(&ctx, 0, dev));

	// the pointer value is cubin ELF entry point
	return fatCubinHandle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{	
	printf("%s\n", __func__);
	printf("    handle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

	cuCtxDestroy(ctx);
	free(fatCubinHandle);
}

void __cudaRegisterFunction(
		void   **fatCubinHandle,
		const char    *hostFun,
		char    *deviceFun,
		const char    *deviceName,
		int      thread_limit,
		uint3   *tid,
		uint3   *bid,
		dim3    *bDim,
		dim3    *gDim,
		int     *wSize
		)
{

	printf("%s\n", __func__);
	printf("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	printf("    hostFun= %s (%p)\n", hostFun, hostFun);
	printf("    deviceFun= %s (%p)\n", deviceFun, deviceFun);
	printf("    deviceName= %s\n", deviceName);
	printf("    thread_limit= %d\n", thread_limit);

	if(tid) printf("    tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	printf("    tid is NULL\n");

	if(bid)	printf("    bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	printf("    bid is NULL\n");

	if(bDim)printf("    bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	printf("    bDim is NULL\n");

	if(gDim)printf("    gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	printf("    gDim is NULL\n");

	if(wSize)printf("    wSize= %d\n", *wSize);
	else	 printf("    wSize is NULL\n");

	
	computeFatBinaryFormat_t fatBinHeader;
	unsigned long long int fatSize;
	char *fatBin;

	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	printf("    magic= %x\n", fatBinHeader->magic);
	printf("    version= %x\n", fatBinHeader->version);
	printf("    headerSize= %x\n", fatBinHeader->headerSize);
	printf("    fatSize= %llx\n", fatBinHeader->fatSize);

	fatSize = fatBinHeader->fatSize;
	fatBin = (char*)malloc(fatSize);
	memcpy(fatBin, fatBinHeader, fatSize);

	Errchk( cuModuleLoadData( &mod, fatBin ));
	Errchk( cuModuleGetFunction(&fun, mod, deviceName) );

	free(fatBin);
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	Errchk( cuMemAlloc( (CUdeviceptr*)devPtr, size));	
	return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr)
{
	Errchk( cuMemFree( (CUdeviceptr)devPtr));
	return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,  enum cudaMemcpyKind kind)
{
	if( kind == cudaMemcpyHostToDevice)
	{
		Errchk( cuMemcpyHtoD((CUdeviceptr)dst, src, count) );
	}
	else if( kind == cudaMemcpyDeviceToHost)
	{
		Errchk( cuMemcpyDtoH(dst, (CUdeviceptr)src, count) );
	}
	return cudaSuccess;
}


cudaError_t cudaConfigureCall(
		dim3 _gridDim, 
		dim3 _blockDim, 
		size_t _sharedMem, 
		cudaStream_t _stream)
{
	printf("%s\n", __func__);
	printf("    gridDim= %d %d %d\n", _gridDim.x, _gridDim.y, _gridDim.z);
	printf("    blockDim= %d %d %d\n", _blockDim.x, _blockDim.y, _blockDim.z);
	printf("    sharedMem= %lu\n", _sharedMem);
	printf("    stream= %p\n", (void*)_stream);
	//printf("    size= %lu\n", sizeof(cudaStream_t));

	memcpy(  &gridDim,   &_gridDim, sizeof(dim3));
	memcpy( &blockDim,  &_blockDim, sizeof(dim3));
	memcpy(&sharedMem, &_sharedMem, sizeof(size_t));
	memcpy(   &stream,    &_stream, sizeof(cudaStream_t));

	return cudaSuccess;
}

cudaError_t cudaSetupArgument(
		const void *arg, 
		size_t size, 
		size_t offset)
{
	printf("%s\n", __func__);
	switch(size)
	{
		case 4:
			printf("    arg= %p, value= %u\n", arg, *(unsigned int*)arg);
			break;
		case 8:
			printf("    arg= %p, value= %llx\n", arg, *(unsigned long long*)arg);
			break;
	}
	printf("    size= %lu\n", size);
	printf("    offset= %lu\n", offset);


	/*
	memcpy(paraBuf+offset, arg, size);
	paraSize += size;
	*/

	paraBuf[ paraSize ] = (void*)arg;
	paraSize++;

	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
	/*
	void *config[] = 
	{
		CU_LAUNCH_PARAM_BUFFER_POINTER, paraBuf,
		CU_LAUNCH_PARAM_BUFFER_SIZE,	&paraSize,
		CU_LAUNCH_PARAM_END
	};
*/
	printf("%s\n", __func__);
	printf("    func= %p\n", func);
	printf("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	printf("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	printf("    sharedMem= %lu\n", sharedMem);
	printf("    stream= %p\n", stream);

	Errchk( cuLaunchKernel(fun, 
				 gridDim.x,  gridDim.y,  gridDim.z,
				blockDim.x, blockDim.y, blockDim.z, 
				sharedMem, NULL, paraBuf, NULL));

	return cudaSuccess;
}

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

#include "../qcu-driver/qcuda_common.h"


#if 1
#define pfunc() printf("### %s\n", __func__)
#else
#define pfunc() 
#endif

#if 1
#define ptrace(fmt, arg...) 	printf("###    "fmt, ##arg)
#else
#define ptrace(fmt, arg...)
#endif

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

#define Errchk(ans) { DrvAssert((ans), __FILE__, __LINE__); }
inline void DrvAssert( CUresult code, const char *file, int line)
{
	char *str;
	if (code != CUDA_SUCCESS) {
		cuGetErrorName(code, (const char**)&str);
		printf("Error: %s at %s:%d\n", str, file, line);
		cuCtxDestroy(ctx);
		exit(code);
	}/* else {
		std::cout << "Success: " << file << "@" << line << std::endl;
		}*/
}

void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic = *(unsigned int*)fatCubin;
	void **fatCubinHandle = malloc(sizeof(void*));

	pfunc();
	ptrace("    fatCubin= %p\n", fatCubin);

	if( magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		ptrace("    FATBINC_MAGIC\n");
		ptrace("    magic= %x\n", binary->magic);
		ptrace("    version= %x\n", binary->version);
		ptrace("    data= %p\n", binary->data);
		ptrace("    filename_or_fatbins= %p\n", binary->filename_or_fatbins);

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
	   ptrace("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
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
	ptrace("%s\n", __func__);
	ptrace("    handle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

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
	pfunc();
	ptrace("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	ptrace("    hostFun= %s (%p)\n", hostFun, hostFun);
	ptrace("    deviceFun= %s (%p)\n", deviceFun, deviceFun);
	ptrace("    deviceName= %s\n", deviceName);
	ptrace("    thread_limit= %d\n", thread_limit);

	if(tid) ptrace("    tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	ptrace("    tid is NULL\n");

	if(bid)	ptrace("    bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	ptrace("    bid is NULL\n");

	if(bDim)ptrace("    bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	ptrace("    bDim is NULL\n");

	if(gDim)ptrace("    gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	ptrace("    gDim is NULL\n");

	if(wSize)ptrace("    wSize= %d\n", *wSize);
	else	 ptrace("    wSize is NULL\n");

	
	computeFatBinaryFormat_t fatBinHeader;
	unsigned long long int fatSize;
	char *fatBin;

	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptrace("    magic= %x\n", fatBinHeader->magic);
	ptrace("    version= %x\n", fatBinHeader->version);
	ptrace("    headerSize= %x\n", fatBinHeader->headerSize);
	ptrace("    fatSize= %llx\n", fatBinHeader->fatSize);

	fatSize = fatBinHeader->fatSize;
	fatBin = (char*)malloc(fatSize);
	memcpy(fatBin, fatBinHeader, fatSize);

	Errchk( cuModuleLoadData( &mod, fatBin ));
	Errchk( cuModuleGetFunction(&fun, mod, deviceName) );

	free(fatBin);
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	pfunc();
	Errchk( cuMemAlloc( (CUdeviceptr*)devPtr, size));	
	return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr)
{
	pfunc();
	Errchk( cuMemFree( (CUdeviceptr)devPtr));
	return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,  enum cudaMemcpyKind kind)
{
	pfunc();
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
	pfunc();
	ptrace("    gridDim= %d %d %d\n", _gridDim.x, _gridDim.y, _gridDim.z);
	ptrace("    blockDim= %d %d %d\n", _blockDim.x, _blockDim.y, _blockDim.z);
	ptrace("    sharedMem= %lu\n", _sharedMem);
	ptrace("    stream= %p\n", (void*)_stream);
	//ptrace("    size= %lu\n", sizeof(cudaStream_t));

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
	pfunc();
	switch(size)
	{
		case 4:
			ptrace("    arg= %p, value= %u\n", arg, *(unsigned int*)arg);
			break;
		case 8:
			ptrace("    arg= %p, value= %llx\n", arg, *(unsigned long long*)arg);
			break;
	}
	ptrace("    size= %lu\n", size);
	ptrace("    offset= %lu\n", offset);


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
	pfunc();
	ptrace("    func= %p\n", func);
	ptrace("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	ptrace("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	ptrace("    sharedMem= %lu\n", sharedMem);
	ptrace("    stream= %p\n", stream);

	Errchk( cuLaunchKernel(fun, 
				 gridDim.x,  gridDim.y,  gridDim.z,
				blockDim.x, blockDim.y, blockDim.z, 
				sharedMem, stream, paraBuf, NULL));
	paraSize = 0;
	return cudaSuccess;
}

//#####################################################################

cudaError_t cudaGetDevice(int *device)
{
	pfunc();
	Errchk( cuDeviceGet((CUdevice*)device, 0));
	return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{	/*
	char name[256];
	size_t totalGlobalMem;
	int major, minor;
	pfunc();
	
	Errchk( cuDeviceGetName(name, 256, device));
	Errchk( cuDeviceTotalMem(&totalGlobalMem, device));
	Errchk( cuDeviceComputeCapability(&major, &minor, device));

	memset(prop, 0, sizeof(struct cudaDeviceProp));
	strcpy(prop->name, name);
	prop->totalGlobalMem = totalGlobalMem;
	prop->major = major;
	prop->minor = minor;

	return cudaSuccess;*/

	pfunc();
	return cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceSynchronize(void) 	
{
	pfunc();
	return cudaSuccess;
}

// typedef struct CUevent_st* cudaEvent_t
// typedef struct CUevent_st* CUevent
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	pfunc();
	Errchk(cuEventCreate(event, 0));
	return cudaSuccess;
}

cudaError_t cudaEventRecord	(cudaEvent_t event,	cudaStream_t stream)
{
	pfunc();
	Errchk(cuEventRecord(event, stream));
	return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) 
{
	pfunc();
	Errchk(cuEventSynchronize(event));
	return cudaSuccess;
}







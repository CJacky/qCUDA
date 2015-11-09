#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <dlfcn.h>

#include "../vhm-driver/virtio_hm.h"

#include <cuda_runtime.h>

#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

void *lib = NULL;

void open_library()
{
	printf("open library\n");
	lib = dlopen("/usr/local/cuda/lib64/libcudart.so.7.5.18", RTLD_LAZY);
	if( !lib )
	{
		printf("open library failed, %s (%d)\n", strerror(errno), errno);
		exit (EXIT_FAILURE);
	}
}

void close_library()
{
	printf("close library\n");
	dlclose(lib);
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	cudaError_t (*func)(void**, size_t);
	cudaError_t err;

	printf("%s\n", __func__);
	printf("    devPtr= %p, size= %lu\n", *devPtr, size);

	func = dlsym(lib, "cudaMalloc");
	err = (*func)(devPtr, size);

	printf("    devPtr= %p, size= %lu\n", *devPtr, size);

	return err;
}

cudaError_t cudaFree(void* devPtr)
{
	cudaError_t (*func)(void*);
	cudaError_t err;

	printf("%s\n", __func__);
	printf("    devPtr= %p\n", devPtr);

	func = dlsym(lib, "cudaFree");
	err = (*func)(devPtr);

	return err;

	//return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,  enum cudaMemcpyKind kind)
{
	cudaError_t (*func)(void*, const void*, size_t, enum cudaMemcpyKind);
	cudaError_t err;

	printf("%s\n", __func__);
	printf("    dst= %p, src= %p, count= %lu, kind %d\n", dst, src, count, kind);

	func = dlsym(lib, "cudaMemcpy");
	err = (*func)(dst, src, count, kind);

	return err;
}

void** __cudaRegisterFatBinary(void *fatCubin)
{
	void** (*func)(void*);
	void **handle;

	open_library();

	printf("%s\n", __func__);
	printf("    fatCubin= %p\n", fatCubin);

	func = dlsym(lib, "__cudaRegisterFatBinary");
	handle = (*func)(fatCubin);

	printf("    handle= %p, value= %p\n", handle, *handle);

	return handle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{	
	void (*func)(void**);

	func = dlsym(lib, "__cudaUnregisterFatBinary");
	(*func)(fatCubinHandle);

	printf("%s\n", __func__);
	printf("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

	close_library();
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
	void (*func)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);

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

	func = dlsym(lib, "__cudaRegisterFunction");
	(*func)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

}



cudaError_t cudaConfigureCall(
		dim3 gridDim, 
		dim3 blockDim, 
		size_t sharedMem, 
		cudaStream_t stream)
{
	cudaError_t (*func)(dim3, dim3, size_t, cudaStream_t);
	cudaError_t err;


	func = dlsym(lib ,"cudaConfigureCall");
	err = (*func)(gridDim, blockDim, sharedMem, stream);

	printf("%s\n", __func__);
	printf("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	printf("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	printf("    sharedMem= %lu\n", sharedMem);
	printf("    stream= %p\n", (void*)stream);
	//printf("    size= %lu\n", sizeof(cudaStream_t));

	if( err != cudaSuccess )
		printf("    FAILED\n");

	return err;
}

cudaError_t cudaSetupArgument(
		const void *arg, 
		size_t size, 
		size_t offset)
{
	cudaError_t (*func)(const void*, size_t, size_t);
	cudaError_t err;

	func = dlsym(lib ,"cudaSetupArgument");
	err = (*func)(arg, size, offset);

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

	if( err != cudaSuccess )
		printf("    FAILED\n");

	return err;
}

cudaError_t cudaLaunch(const void *func)
{
	cudaError_t (*func_L)(const void*);
	cudaError_t err;

	func_L = dlsym(lib, "cudaLaunch");
	err = (*func_L)(func);

	printf("%s\n", __func__);
	printf("    func= %p\n", func);

	if( err != cudaSuccess )
		printf("    FAILED\n");

	return err;
}

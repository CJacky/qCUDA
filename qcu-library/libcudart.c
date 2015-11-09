#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/ioctl.h> // ioclt
#include "../qcu-driver/qcuda_common.h"

#include <builtin_types.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

#define dev_path "/dev/qcuda"

#if 0
#define cjPrint(fmt, arg...) printf(fmt, ##arg)
#else
#define cjPrint(fmt, arg...)
#endif

#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)

#define zalloc(n) ({\
	void *addr = malloc(n); \
	memset(addr, 0, n); \
	addr; })


#define ptr( p , v, s)\
	p = (uint64_t)v; \
	p##Size = (uint32_t)s;

int fd;
uint32_t cudaKernelConf[7];
uint64_t cudaKernelPara[16];
size_t cudaParaNum = 0;

void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic;
	void **fatCubinHandle;

	cjPrint("%s\n", __func__);
	
	fd = open(dev_path, O_RDWR);
	if( fd < 0 )
	{
		error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
		exit (EXIT_FAILURE);
	}

	magic = *(unsigned int*)fatCubin;
	fatCubinHandle = malloc(sizeof(void*));

	if( magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		cjPrint("    FATBINC_MAGIC\n");
		cjPrint("    magic= %x\n", binary->magic);
		cjPrint("    version= %x\n", binary->version);
		cjPrint("    data= %p\n", binary->data);
		cjPrint("    filename_or_fatbins= %p\n", binary->filename_or_fatbins);
		
		*fatCubinHandle = (void*)binary->data;
	}
	else 
	{
		/*
		magic: __cudaFatFormat.h
		header: __cudaFatMAGIC)
		__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;

		magic: FATBIN_MAGIC
		header: fatbinary.h
		computeFatBinaryFormat_t binary = (computeFatBinaryFormat_t)fatCubin;
		*/
		cjPrint("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
		exit(EXIT_FAILURE);
	}

	ioctl(fd, VIRTQC_cudaRegisterFatBinary, NULL);
	
	// the pointer value is cubin ELF entry point
	return fatCubinHandle;
}


void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	cjPrint("%s\n", __func__);
	cjPrint("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

	ioctl(fd, VIRTQC_cudaUnregisterFatBinary, NULL);
	free(fatCubinHandle);
	close(fd);
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
	VirtioQCArg *arg;
	computeFatBinaryFormat_t fatBinHeader;

	cjPrint("%s\n", __func__);
	cjPrint("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);
	cjPrint("    hostFun= %s (%p)\n", hostFun, hostFun);
	cjPrint("    deviceFun= %s (%p)\n", deviceFun, deviceFun);
	cjPrint("    deviceName= %s\n", deviceName);
	cjPrint("    thread_limit= %d\n", thread_limit);

	if(tid) cjPrint("    tid= %u %u %u\n", tid->x, tid->y, tid->z);
	else	cjPrint("    tid is NULL\n");

	if(bid)	cjPrint("    bid= %u %u %u\n", bid->x, bid->y, bid->z);
	else	cjPrint("    bid is NULL\n");

	if(bDim)cjPrint("    bDim= %u %u %u\n", bDim->x, bDim->y, bDim->z);
	else	cjPrint("    bDim is NULL\n");

	if(gDim)cjPrint("    gDim= %u %u %u\n", gDim->x, gDim->y, gDim->z);
	else	cjPrint("    gDim is NULL\n");

	if(wSize)cjPrint("    wSize= %d\n", *wSize);
	else	 cjPrint("    wSize is NULL\n");

	arg = zalloc(sizeof(VirtioQCArg));
	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptr( arg->pA , fatBinHeader, fatBinHeader->fatSize);
	ptr( arg->pB , deviceName  , strlen(deviceName)+1 );

//	cjPrint("pA= %p, pASize= %u, pB= %p, pBSize= %u\n", 
//			(void*)arg->pA, arg->pASize, (void*)arg->pB, arg->pBSize);

	ioctl(fd, VIRTQC_cudaRegisterFunction, arg);

	free(arg);
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	VirtioQCArg *arg;

	cjPrint("%s\n", __func__);

	arg = zalloc(sizeof(VirtioQCArg));

	ptr( arg->pA, 0,  0);
	arg->flag = size;

	ioctl(fd, VIRTQC_cudaMalloc, arg);

	*devPtr = (void*)arg->pA;
	
	cjPrint("    devPtr= %p\n", (void*)arg->pA);

	free(arg);
	return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr)
{
	VirtioQCArg *arg;

	arg = zalloc(sizeof(VirtioQCArg));

	cjPrint("%s\n", __func__);

	ptr( arg->pA, devPtr, 0);

	ioctl(fd, VIRTQC_cudaFree, arg);

	cjPrint("    devPtr= %p\n", (void*)arg->pA);

	free(arg);
	return cudaSuccess;
}

cudaError_t cudaMemcpy(
		void* dst, 
		const void* src, 
		size_t count,  
		enum cudaMemcpyKind kind)
{
	VirtioQCArg *arg;
	arg = zalloc(sizeof(VirtioQCArg));

	cjPrint("%s\n", __func__);
//	cjPrint("dst= %p , src= %p ,size= %u\n", dst, src, size);

	if( kind == cudaMemcpyHostToDevice)
	{
		ptr( arg->pA, dst, 0);
		ptr( arg->pB, src, count);
		arg->flag   = 1;
	}
	else if( kind == cudaMemcpyDeviceToHost )
	{
		ptr( arg->pA, dst, count);
		ptr( arg->pB, src, 0);
		arg->flag   = 2;
	}
	else
	{
		error("Not impletment cudaMemcpyKind\n");
		free(arg);
		return cudaErrorInvalidValue;
	}

	ioctl(fd, VIRTQC_cudaMemcpy, arg);

	free(arg);
	return cudaSuccess;
}

cudaError_t cudaConfigureCall(
		dim3 gridDim, 
		dim3 blockDim, 
		size_t sharedMem, 
		cudaStream_t stream)
{
	cjPrint("%s\n", __func__);
	cjPrint("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	cjPrint("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	cjPrint("    sharedMem= %lu\n", sharedMem);
	cjPrint("    stream= %p\n", (void*)stream);
	//cjPrint("    size= %lu\n", sizeof(cudaStream_t));

	cudaKernelConf[0] = gridDim.x;
	cudaKernelConf[1] = gridDim.y;
	cudaKernelConf[2] = gridDim.z;

	cudaKernelConf[3] = blockDim.x;
	cudaKernelConf[4] = blockDim.y;
	cudaKernelConf[5] = blockDim.z;

	cudaKernelConf[6] = sharedMem;

	return cudaSuccess;
}

cudaError_t cudaSetupArgument(
		const void *arg, 
		size_t size, 
		size_t offset)
{
	cjPrint("%s\n", __func__);
	/*
	switch(size)
	{
		case 4:
			cjPrint("    arg= %p, value= %u\n", arg, *(unsigned int*)arg);
			break;
		case 8:
			cjPrint("    arg= %p, value= %llx\n", arg, *(unsigned long long*)arg);
			break;
	}

	cjPrint("    size= %lu\n", size);
	cjPrint("    offset= %lu\n", offset);
*/
//	uint64_t *buf = (uint64_t*)&cudaKernelConf[7];
	/*
	switch(size)
	{
		case 4: buf[ cudaParaNum ] = *(uint32_t*)arg;
		case 8: buf[ cudaParaNum ] = *(uint64_t*)arg;
		default:
			cjPrint("    unknow data size %lu\n", size);
	}*/

	if( size == 4 )
	{
		cudaKernelPara[ cudaParaNum ] = *(uint32_t*)arg;
	}
	else if( size == 8 )
	{
		cudaKernelPara[ cudaParaNum ] = *(uint64_t*)arg;
	}
	else
	{
		cjPrint("    unknow data size %lu\n", size);
	}

	cjPrint("    para %lu = %llx\n", cudaParaNum, 
			(unsigned long long)cudaKernelPara[cudaParaNum]);

	cudaParaNum++;

	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
	VirtioQCArg *arg;
	arg = zalloc(sizeof(VirtioQCArg));

	cjPrint("%s\n", __func__);

	ptr( arg->pA, cudaKernelConf, 7*sizeof(uint32_t));
	ptr( arg->pB, cudaKernelPara, cudaParaNum*sizeof(uint64_t));

	ioctl(fd, VIRTQC_cudaLaunch, arg);
	
	cudaParaNum = 0;

	free(arg);
	return cudaSuccess;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/ioctl.h> // ioclt

#include <builtin_types.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>

#define	PFUNC	1
#define PTRACE	1
#include "../qcu-driver/qcuda_common.h"

#define dev_path "/dev/qcuda"

#define error(fmt, arg...) printf("ERROR: "fmt, ##arg)


#define ptr( p , v, s)\
	p = (uint64_t)v; \
p##Size = (uint32_t)s;

int fd = -1;
uint32_t cudaKernelConf[7];
uint64_t cudaKernelPara[16];
size_t cudaParaNum = 0;

////////////////////////////////////////////////////////////////////////////////
/// General Function
////////////////////////////////////////////////////////////////////////////////

void open_device()
{
	if( fd == -1)
	{
		fd = open(dev_path, O_RDWR);
		if( fd < 0 )
		{
			error("open device %s faild, %s (%d)\n", dev_path, strerror(errno), errno);
			exit (EXIT_FAILURE);
		}
	}
}

void close_device()
{
	close(fd);
}

////////////////////////////////////////////////////////////////////////////////
/// Module & Execution control
////////////////////////////////////////////////////////////////////////////////

void** __cudaRegisterFatBinary(void *fatCubin)
{
	unsigned int magic;
	void **fatCubinHandle;

	pfunc();
	open_device();

	magic = *(unsigned int*)fatCubin;
	fatCubinHandle = malloc(sizeof(void*));

	if( magic == FATBINC_MAGIC)
	{// fatBinaryCtl.h
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		ptrace("    FATBINC_MAGIC\n");
		ptrace("    magic= %x\n", binary->magic);
		ptrace("    version= %x\n", binary->version);
		ptrace("    data= %p\n", binary->data);
		ptrace("    filename_or_fatbins= %p\n", binary->filename_or_fatbins);

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
		ptrace("Unrecognized CUDA FAT MAGIC 0x%x\n", magic);
		exit(EXIT_FAILURE);
	}

	ioctl(fd, VIRTQC_cudaRegisterFatBinary, NULL);

	// the pointer value is cubin ELF entry point
	return fatCubinHandle;
}


void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	ptrace("%s\n", __func__);
	ptrace("    fatCubinHandle= %p, value= %p\n", fatCubinHandle, *fatCubinHandle);

	ioctl(fd, VIRTQC_cudaUnregisterFatBinary, NULL);
	free(fatCubinHandle);
	close_device();
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
	VirtioQCArg arg;
	computeFatBinaryFormat_t fatBinHeader;

	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

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

	
	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);

	ptr( arg.pA , fatBinHeader, fatBinHeader->fatSize);
	ptr( arg.pB , deviceName  , strlen(deviceName)+1 );

	//	ptrace("pA= %p, pASize= %u, pB= %p, pBSize= %u\n", 
	//			(void*)arg.pA, arg.pASize, (void*)arg.pB, arg.pBSize);

	ioctl(fd, VIRTQC_cudaRegisterFunction, &arg);
}

cudaError_t cudaConfigureCall(
		dim3 gridDim, 
		dim3 blockDim, 
		size_t sharedMem, 
		cudaStream_t stream)
{
	pfunc();
	ptrace("    gridDim= %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
	ptrace("    blockDim= %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
	ptrace("    sharedMem= %lu\n", sharedMem);
	ptrace("    stream= %p\n", (void*)stream);
	//ptrace("    size= %lu\n", sizeof(cudaStream_t));

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
	pfunc();
	/*
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
	 */
	//	uint64_t *buf = (uint64_t*)&cudaKernelConf[7];
	/*
	   switch(size)
	   {
	   case 4: buf[ cudaParaNum ] = *(uint32_t*)arg;
	   case 8: buf[ cudaParaNum ] = *(uint64_t*)arg;
	   default:
	   ptrace("    unknow data size %lu\n", size);
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
		ptrace("    unknow data size %lu\n", size);
	}

	ptrace("    para %lu = %llx\n", cudaParaNum, 
			(unsigned long long)cudaKernelPara[cudaParaNum]);

	cudaParaNum++;

	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ptr( arg.pA, cudaKernelConf, 7*sizeof(uint32_t));
	ptr( arg.pB, cudaKernelPara, cudaParaNum*sizeof(uint64_t));

	ioctl(fd, VIRTQC_cudaLaunch, &arg);

	cudaParaNum = 0;
	
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	VirtioQCArg arg;

	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	open_device();

	ptr( arg.pA, 0,  0);
	arg.flag = size;

	ioctl(fd, VIRTQC_cudaMalloc, &arg);

	*devPtr = (void*)arg.pA;

	ptrace("    devPtr= %p\n", (void*)arg.pA);

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaFree(void* devPtr)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, devPtr, 0);

	ioctl(fd, VIRTQC_cudaFree, &arg);

	ptrace("    devPtr= %p\n", (void*)arg.pA);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaMemcpy(
		void* dst, 
		const void* src, 
		size_t count,  
		enum cudaMemcpyKind kind)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptrace("dst= %p , src= %p ,size= %lu\n", (void*)dst, (void*)src, count);

	if( kind == cudaMemcpyHostToDevice)
	{
		ptr( arg.pA, dst, 0);
		ptr( arg.pB, src, count);
		arg.flag   = 1;
	}
	else if( kind == cudaMemcpyDeviceToHost )
	{
		ptr( arg.pA, dst, count);
		ptr( arg.pB, src, 0);
		arg.flag   = 2;
	}
	else
	{
		error("Not impletment cudaMemcpyKind\n");
		
		return cudaErrorInvalidValue;
	}

	ioctl(fd, VIRTQC_cudaMemcpy, &arg);

	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Device Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetDevice(int *device)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ioctl(fd, VIRTQC_cudaGetDevice, &arg);
	*device = (int)arg.pA;

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaGetDeviceCount, &arg);
	*count = (int)arg.pA;

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaSetDevice(int device)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, device, 0);
	ioctl(fd, VIRTQC_cudaGetDeviceProperties, &arg);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, prop, sizeof(struct cudaDeviceProp));
	ptr( arg.pB, device, 0);
	ioctl(fd, VIRTQC_cudaSetDevice, &arg);

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaDeviceSynchronize(void)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ioctl(fd, VIRTQC_cudaDeviceSynchronize, &arg);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaDeviceReset(void)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaDeviceReset, &arg);

	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Version Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaDriverGetVersion, &arg);
	*driverVersion = (int)arg.pA;
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaRuntimeGetVersion, &arg);
	*runtimeVersion = (uint64_t)arg.pA;
	
	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Event Management
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaEventCreate, &arg);
	*event = (void*)arg.pA;
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventRecord	(cudaEvent_t event,	cudaStream_t stream)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, event, 0);
	ptr( arg.pB, stream, 0);
	ioctl(fd, VIRTQC_cudaEventRecord, &arg);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ptr( arg.pA, event, 0);
	ioctl(fd, VIRTQC_cudaEventSynchronize, &arg);
	
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventElapsedTime(float *ms,	cudaEvent_t start, cudaEvent_t end)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ptr( arg.pA, start, 0);
	ptr( arg.pB, end, 0);
	ioctl(fd, VIRTQC_cudaEventElapsedTime, &arg);
	memcpy(ms, &arg.flag, sizeof(float));

	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));
	
	ptr( arg.pA, event, 0);
	ioctl(fd, VIRTQC_cudaEventDestroy, &arg);

	return (cudaError_t)arg.cmd;
}

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetLastError(void)
{
	VirtioQCArg arg;
	pfunc();
	memset(&arg, 0, sizeof(VirtioQCArg));

	ioctl(fd, VIRTQC_cudaGetLastError, &arg);
	
	return (cudaError_t)arg.cmd;
}


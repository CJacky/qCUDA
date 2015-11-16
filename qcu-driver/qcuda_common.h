
#ifndef QCUDA_COMMON_H
#define QCUDA_COMMON_H

////////////////////////////////////////////////////////////////////////////////
///	common variables
////////////////////////////////////////////////////////////////////////////////

#define QCU_KMALLOC_SHIFT_BIT 22
#define QCU_KMALLOC_MAX_SIZE (1UL<<QCU_KMALLOC_SHIFT_BIT)
// don't bigger than 1<<22

#define VIRTIO_ID_QC 69

enum
{
	VIRTQC_CMD_WRITE = 100,
	VIRTQC_CMD_READ,
};

enum
{
	// Module & Execution control (driver API)
	VIRTQC_cudaRegisterFatBinary = 200,
	VIRTQC_cudaUnregisterFatBinary,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaLaunch,

	// Memory Management (runtime API)
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaFree,

	// Device Management (runtime API)
	VIRTQC_cudaGetDevice,
	VIRTQC_cudaGetDeviceCount,
	VIRTQC_cudaGetDeviceProperties,
	VIRTQC_cudaSetDevice,
	VIRTQC_cudaDeviceSynchronize,
	VIRTQC_cudaDeviceReset,

	// Version Management (runtime API)
	VIRTQC_cudaDriverGetVersion,
	VIRTQC_cudaRuntimeGetVersion,

	// Event Management (runtime API)
	VIRTQC_cudaEventCreate,
	VIRTQC_cudaEventRecord,
	VIRTQC_cudaEventSynchronize,
	VIRTQC_cudaEventElapsedTime,
	VIRTQC_cudaEventDestroy,

	// Error Handling (runtime API)
	VIRTQC_cudaGetLastError,

};

typedef struct VirtioQCArg   VirtioQCArg;

// function args
struct VirtioQCArg
{
	int32_t cmd;
	int32_t rnd;
	
	uint64_t pA;
	uint32_t pASize;

	uint64_t pB;
	uint32_t pBSize;

	uint32_t flag;
};

#endif

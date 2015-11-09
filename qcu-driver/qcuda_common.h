
#ifndef QCUDA_COMMON_H
#define QCUDA_COMMON_H


#if 1

#ifdef __KERNEL__
	#include <linux/timekeeping.h>
	#define qcu_gettime(t) do_gettimeofday(t)
	#define print printk
#else
	#include <sys/time.h>
	#define qcu_gettime(t) gettimeofday (t, NULL)
	#define print printf
#endif

uint64_t qcu_TimeRegFatbin;
uint64_t qcu_TimeUnFatbin;
uint64_t qcu_TimeRegFunc;
uint64_t qcu_TimeLaunch;
uint64_t qcu_TimeMalloc;
uint64_t qcu_TimeFree;
uint64_t qcu_TimeMemcpyH2D;
uint64_t qcu_TimeMemcpyD2H;

#define time_reset() \
	qcu_TimeRegFatbin = 0; \
	qcu_TimeRegFunc = 0; \
	qcu_TimeMalloc = 0; \
	qcu_TimeLaunch = 0; \
	qcu_TimeFree = 0; \
	qcu_TimeUnFatbin = 0; \
	qcu_TimeMemcpyH2D = 0; \
	qcu_TimeMemcpyD2H = 0;

#define time_print() \
	print("%s\t%llu\n", "cudaRegisterFatBinary", (unsigned long long)qcu_TimeRegFatbin); \
	print("%s\t%llu\n", "cudaRegisterFunction", (unsigned long long)qcu_TimeRegFunc); \
	print("%s\t%llu\n", "cudaMalloc", (unsigned long long)qcu_TimeMalloc); \
	print("%s\t%llu\n", "cudaMemcpyH2D", (unsigned long long)qcu_TimeMemcpyH2D); \
	print("%s\t%llu\n", "cudaMemcpyD2H", (unsigned long long)qcu_TimeMemcpyD2H); \
	print("%s\t%llu\n", "cudaLaunch", (unsigned long long)qcu_TimeLaunch); \
	print("%s\t%llu\n", "cudaFree", (unsigned long long)qcu_TimeFree); \
	print("%s\t%llu\n", "cudaUnregisterFatBinary", (unsigned long long)qcu_TimeUnFatbin); \
	print("\n\n");

#define time_begin() \
	struct timeval timeval_begin; \
	qcu_gettime(&timeval_begin);

#define time_end() ({ \
		struct timeval timeval_end; \
		uint64_t time_usec; \
		qcu_gettime( &timeval_end); \
		time_usec = (timeval_end.tv_sec  - timeval_begin.tv_sec)*1000000; \
		time_usec+= (timeval_end.tv_usec - timeval_begin.tv_usec); \
		time_usec; })

#define time_add(v, t) \
	v += t

#else // MEAS_TIME

#define time_reset()
#define time_begin()
#define time_end()
#define time_print()
#define time_add(v, t) 

#endif //MEAS_TIME


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
	VIRTQC_cudaRegisterFatBinary = 200,
	VIRTQC_cudaUnregisterFatBinary,
	VIRTQC_cudaRegisterFunction,
	VIRTQC_cudaLaunch,
	VIRTQC_cudaMalloc,
	VIRTQC_cudaMemcpy,
	VIRTQC_cudaFree,
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

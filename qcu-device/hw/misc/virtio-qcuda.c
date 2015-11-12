#include "qemu-common.h"
#include "qemu/iov.h"
#include "qemu/error-report.h"
#include "hw/virtio/virtio.h"
#include "hw/virtio/virtio-bus.h"
#include "hw/virtio/virtio-qcuda.h"

#ifdef CONFIG_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#endif

#define PFUNC	1
#define PTRACE	1
#include "../../../qcu-driver/qcuda_common.h"

#define error(fmt, arg...) \
	error_report("file %s ,line %d ,ERROR: "fmt, __FILE__, __LINE__, ##arg)

#ifndef MIN
#define MIN(a,b) ({ ((a)<(b))? (a):(b) })
#endif

char *deviceSpace = NULL;
uint32_t deviceSpaceSize = 0;

static void* gpa_to_hva(uint64_t pa) 
{
	MemoryRegionSection section;

	section = memory_region_find(get_system_memory(), (ram_addr_t)pa, 1);
	if ( !int128_nz(section.size) || !memory_region_is_ram(section.mr)){
		error("addr %p in rom\n", (void*)pa); 
		return 0;
	}

	return (memory_region_get_ram_ptr(section.mr) + section.offset_within_region);
}

#ifdef CONFIG_CUDA
CUdevice cudaDevice;
CUcontext cudaContext;
CUmodule cudaModule;

#define cudaFunctionMaxNum 8
CUfunction cudaFunction[cudaFunctionMaxNum];
uint64_t cudaFunctionId[cudaFunctionMaxNum];
uint32_t cudaFunctionNum;

#define cudaEventMaxNum 16
cudaEvent_t cudaEvent[cudaEventMaxNum];
uint32_t cudaEventNum;

#define cudaStreamMaxNum 32
cudaStream_t cudaStream[cudaStreamMaxNum];
uint32_t cudaStreamNum;

#define cudaError(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line)
{
	char *str;
	if ( err != cudaSuccess )
	{
		str = (char*)cudaGetErrorString(err);
		error_report("CUDA Runtime API error = %04d \"%s\" line %d\n", err, str, line);
	}
}


#define cuError(err)  __cuErrorCheck(err, __LINE__) 
static inline void __cuErrorCheck(CUresult err, const int line)
{
	char *str;
	if ( err != CUDA_SUCCESS )
	{   
		cuGetErrorName(err, (const char**)&str);
		error_report("CUDA Runtime API error = %04d \"%s\" line %d\n", err, str, line);
	}   
}

////////////////////////////////////////////////////////////////////////////////
///	Module & Execution control (driver API)
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaRegisterFatBinary(VirtioQCArg *arg)
{
	uint32_t i;
	time_begin();
	time_reset();
	pfunc();

	for(i=0; i<cudaFunctionMaxNum; i++)
		memset(&cudaFunction[i], 0, sizeof(CUfunction));

	for(i=0; i<cudaEventMaxNum; i++)
		memset(&cudaEvent[i], 0, sizeof(cudaEvent_t));

	for(i=0; i<cudaStreamMaxNum; i++)
		memset(&cudaStream[i], 0, sizeof(cudaStream_t));

	cuError( cuInit(0) );
	cuError( cuDeviceGet(&cudaDevice, 0) );
	cuError( cuCtxCreate(&cudaContext, 0, cudaDevice) );

	cudaFunctionNum = 0;
	cudaEventNum = 0;

	time_add( qcu_TimeRegFatbin , time_end() );
}

static void qcu_cudaUnregisterFatBinary(VirtioQCArg *arg)
{
	uint32_t i;
	time_begin();	
	pfunc();

	for(i=0; i<cudaEventMaxNum; i++)
	{
		if( cudaEvent[i] != 0 ){
			cudaError( cudaEventDestroy(cudaEvent[i]));
		}
	}

	cuCtxDestroy(cudaContext);
	
	time_add( qcu_TimeUnFatbin , time_end() );
	time_print();
}

static void qcu_cudaRegisterFunction(VirtioQCArg *arg)
{
	void *fatBin;
	char *functionName;
	uint32_t funcId;
	time_begin();
	
	pfunc();

	// assume fatbin size is less equal 4MB
	fatBin       = gpa_to_hva(arg->pA);
	functionName = gpa_to_hva(arg->pB);
	funcId		 = arg->flag;

	ptrace("fatBin= %16p ,name= '%s'\n", fatBin, functionName);
	cuError( cuModuleLoadData( &cudaModule, fatBin ));
	cuError( cuModuleGetFunction(&cudaFunction[cudaFunctionNum], 
				cudaModule, functionName) );
	cudaFunctionId[cudaFunctionNum] = funcId;
	cudaFunctionNum++;
	
	time_add( qcu_TimeRegFunc ,  time_end() );
}

static void qcu_cudaLaunch(VirtioQCArg *arg)
{
	unsigned int *conf;
	uint64_t *para, funcId;
	uint32_t paraSize, funcIdx;
	void **paraBuf;
	int i;
	time_begin();
	pfunc();

	conf = gpa_to_hva(arg->pA);
	para = gpa_to_hva(arg->pB);
	paraSize = (arg->pBSize)/sizeof(uint64_t);
	funcId = arg->flag;

	paraBuf = malloc(sizeof(void*)*paraSize);
	for(i=0; i<paraSize; i++)
		paraBuf[i] = &para[i];

	for(funcIdx=0; funcIdx<cudaFunctionNum; funcIdx++)
	{
		if( cudaFunctionId[funcIdx] == funcId )
			break;
	}

	ptrace("grid (%u %u %u) block(%u %u %u) sharedMem(%u)\n", 
			conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]);

	/*
	for(i=0; i<paraSize; i++)
		ptrace("    para %d = %llu\n", i, *(unsigned long long*)paraBuf[i]);
	*/	

	cuError( cuLaunchKernel(cudaFunction[funcIdx],
				conf[0], conf[1], conf[2],
				conf[3], conf[4], conf[5], 
				conf[6], NULL, paraBuf, NULL)); // not suppoer stream yeat
	
	free(paraBuf);
	time_add( qcu_TimeLaunch , time_end() );
}

////////////////////////////////////////////////////////////////////////////////
/// Memory Management (runtime API)
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaMalloc(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t count;
	void* devPtr;
	time_begin();
	pfunc();

	count = arg->flag;;
	cudaError((err = cudaMalloc( &devPtr, count )));
	arg->cmd = err;
	arg->pA = (uint64_t)devPtr;

	ptrace("ptr= %p ,count= %u\n", (void*)arg->pA, count);
	time_add( qcu_TimeMalloc , time_end() );
}

static void qcu_cudaMemcpy(VirtioQCArg *arg)
{
	cudaError_t err;
	void *dst, *src;
	uint64_t *gpa_array;
	uint32_t size, len, i;
	time_begin();
	pfunc();

	if( arg->flag == cudaMemcpyHostToDevice )
	{
		dst = (void*)arg->pA;
		size = arg->pBSize;

		if( size > QCU_KMALLOC_MAX_SIZE)
		{
			gpa_array = gpa_to_hva(arg->pB);
			for(i=0; size>0; i++)
			{
				src = gpa_to_hva(gpa_array[i]);
				len = MIN(size, QCU_KMALLOC_MAX_SIZE);
				cudaError(( err = cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice)));
				dst  += len;
				size -= len;
			}
		}
		else
		{
			src = gpa_to_hva(arg->pB);
			cudaError(( err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)));
		}
		arg->cmd = err;
	}
	else if(arg->flag == cudaMemcpyDeviceToHost )
	{
		src = (void*)arg->pB;
		size = arg->pASize;

		if( size > QCU_KMALLOC_MAX_SIZE)
		{
			gpa_array = gpa_to_hva(arg->pA);
			for(i=0; size>0; i++)
			{
				dst = gpa_to_hva(gpa_array[i]);
				len = MIN(size, QCU_KMALLOC_MAX_SIZE);
				cudaError(( err = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost)));
				src  += len;
				size -= len;
			}
		}
		else
		{
			dst = gpa_to_hva(arg->pA);
			cudaError(( err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)));
		}
		arg->cmd = err;
	}
	else if( arg->flag == cudaMemcpyDeviceToDevice )
	{
		dst = (void*)arg->pA;
		src = (void*)arg->pB;
		size = arg->pBSize;
		cudaError(( err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)));
		arg->cmd = err;
	}

	ptrace("size= %u\n", size);
	time_add( qcu_TimeMemcpyD2H , time_end() );
}

static void qcu_cudaFree(VirtioQCArg *arg)
{
	cudaError_t err;
	void* dst;
	time_begin();
	pfunc();

	dst = (void*)arg->pA;
	cudaError((err = cudaFree(dst)));
	arg->cmd = err;

	ptrace("ptr= %16p\n", dst);
	time_add( qcu_TimeFree , time_end() );
}

////////////////////////////////////////////////////////////////////////////////
///	Device Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaGetDevice(VirtioQCArg *arg)
{
	cudaError_t err;
	int device;
	pfunc();

	cudaError((err = cudaGetDevice( &device )));
	arg->cmd = err;
	arg->pA = (uint64_t)device;

	ptrace("device= %d\n", device);
}

static void qcu_cudaGetDeviceCount(VirtioQCArg *arg)
{
	cudaError_t err;
	int device;
	pfunc();

	cudaError((err = cudaGetDeviceCount( &device )));
	arg->cmd = err;
	arg->pA = (uint64_t)device;

	ptrace("device count=%d\n", device);
}

static void qcu_cudaSetDevice(VirtioQCArg *arg)
{
	cudaError_t err;
	int device;
	pfunc();

	device = (int)arg->pA;
	cudaError((err = cudaSetDevice( device )));
	arg->cmd = err;

	ptrace("set device= %d\n", device);
}

static void qcu_cudaGetDeviceProperties(VirtioQCArg *arg)
{
	cudaError_t err;
	struct cudaDeviceProp *prop;
	int device;
	pfunc();

	prop = gpa_to_hva(arg->pA);
	device = (int)arg->pB;

	cudaError((err = cudaGetDeviceProperties( prop, device )));
	arg->cmd = err;

	ptrace("get prop for device %d\n", device);
}

static void qcu_cudaDeviceSynchronize(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();
	cudaError((err = cudaDeviceSynchronize()));
	arg->cmd = err;
}

static void qcu_cudaDeviceReset(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();
	cudaError((err = cudaDeviceReset()));
	arg->cmd = err;
}

////////////////////////////////////////////////////////////////////////////////
///	Version Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaDriverGetVersion(VirtioQCArg *arg)
{
	cudaError_t err;
	int version;
	pfunc();

	cudaError((err = cudaDriverGetVersion( &version )));
	arg->cmd = err;
	arg->pA = (uint64_t)version;

	ptrace("driver version= %d\n", version);
}

static void qcu_cudaRuntimeGetVersion(VirtioQCArg *arg)
{
	cudaError_t err;
	int version;
	pfunc();

	cudaError((err = cudaRuntimeGetVersion( &version )));
	arg->cmd = err;
	arg->pA = (uint64_t)version;

	ptrace("runtime driver= %d\n", version);
}

////////////////////////////////////////////////////////////////////////////////
///	Event Management
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaEventCreate(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = cudaEventNum;
	cudaError((err = cudaEventCreate(&cudaEvent[idx])));
	arg->cmd = err;
	arg->pA = (uint64_t)idx;

	cudaEventNum++;
	ptrace("create event %u\n", idx);
}

static void qcu_cudaEventRecord(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t eventIdx;
	uint32_t streamIdx;
	pfunc();

	eventIdx  = arg->pA;
	streamIdx = arg->pB;
	cudaError((err = cudaEventRecord(cudaEvent[eventIdx], cudaStream[streamIdx])));
	arg->cmd = err;

	ptrace("event record %u\n", eventIdx);
}

static void qcu_cudaEventSynchronize(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = arg->pA;
	cudaError((err = cudaEventSynchronize( cudaEvent[idx] )));
	arg->cmd = err;

	ptrace("sync event %u\n", idx);
}

static void qcu_cudaEventElapsedTime(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t startIdx;
	uint32_t endIdx;
	float ms;
	pfunc();

	startIdx = arg->pA;
	endIdx   = arg->pB;
	cudaError((err = cudaEventElapsedTime(&ms, cudaEvent[startIdx], cudaEvent[endIdx])));
	arg->cmd = err;
	memcpy(&arg->flag, &ms, sizeof(float));

	ptrace("event elapse time= %f, start= %u, end= %u\n", 
			ms, startIdx, endIdx);
}

static void qcu_cudaEventDestroy(VirtioQCArg *arg)
{
	cudaError_t err;
	uint32_t idx;
	pfunc();

	idx = arg->pA;
	cudaError((err = cudaEventDestroy(cudaEvent[idx])));
	arg->cmd = err;
	memset(&cudaEvent[idx], 0, sizeof(cudaEvent_t));

	ptrace("destroy event %u\n", idx);
}

////////////////////////////////////////////////////////////////////////////////
///	Error Handling
////////////////////////////////////////////////////////////////////////////////

static void qcu_cudaGetLastError(VirtioQCArg *arg)
{
	cudaError_t err;
	pfunc();

	err =  cudaGetLastError();
	arg->cmd = err;
	ptrace("lasr cudaError %d\n", err);
}

#endif // CONFIG_CUDA

static int qcu_cmd_write(VirtioQCArg *arg)
{
	void   *src, *dst;
	uint64_t *gpa_array;
	uint32_t size, len, i;

	size = arg->pASize;

	ptrace("szie= %u\n", size);

	if(deviceSpace!=NULL)
	{
		free(deviceSpace);	
	}

	deviceSpaceSize = size;;
	deviceSpace = (char*)malloc(deviceSpaceSize);

	if( size > deviceSpaceSize )
	{
		gpa_array = gpa_to_hva(arg->pA);
		dst = deviceSpace;
		for(i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			src = gpa_to_hva(gpa_array[i]);
			memcpy(dst, src, len);
			size -= len;
			dst  += len;
		}
	}
	else
	{
		src = gpa_to_hva(arg->pA);
		memcpy(deviceSpace, src, size);
	}
	// checker ------------------------------------------------------------
/*
	uint64_t err;
	if( deviceSpaceSize<32 )
	{
		for(i=0; i<deviceSpaceSize; i++)
		{
			ptrace("deviceSpace[%lu]= %d\n", i, deviceSpace[i]);
		}
	}
	else
	{	
		err = 0;
		for(i=0; i<deviceSpaceSize; i++)
		{
			if( deviceSpace[i] != (i%17)*7 ) err++;
		}
		ptrace("error= %llu\n", (unsigned long long)err);
	}
	ptrace("\n\n");
	//---------------------------------------------------------------------
*/
	return 0;
}

static int qcu_cmd_read(VirtioQCArg *arg)
{
	void   *src, *dst;
	uint64_t *gpa_array;
	uint32_t size, len, i;

	if(deviceSpace==NULL)
	{
		return -1;
	}

	size = arg->pASize;

	ptrace("szie= %u\n", size);
	
	if( size > deviceSpaceSize )
	{
		gpa_array = gpa_to_hva(arg->pA);
		src = deviceSpace;
		for(i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			dst = gpa_to_hva(gpa_array[i]);
			memcpy(dst, src, len);
			size -= len;
			src  += len;
		}
	}
	else
	{
		dst = gpa_to_hva(arg->pA);
		memcpy(dst, deviceSpace, size);
	}

	return 0;
}

static void virtio_qcuda_cmd_handle(VirtIODevice *vdev, VirtQueue *vq)
{
	VirtQueueElement elem;
	VirtioQCArg *arg;

	struct iovec *iov;
	unsigned int iov_num;

	while( virtqueue_pop(vq, &elem) )
	{
		arg = malloc( sizeof(VirtioQCArg));

		iov_num = elem.out_num;
		iov = g_memdup(elem.out_sg, sizeof(struct iovec) * elem.out_num);
		iov_to_buf(iov, iov_num, 0, arg, sizeof(VirtioQCArg));

		switch( arg->cmd )
		{
			case VIRTQC_CMD_WRITE:
				qcu_cmd_write(arg);
				break;

			case VIRTQC_CMD_READ:
				qcu_cmd_read(arg); 
				break;

#ifdef CONFIG_CUDA
			// Module & Execution control (driver API)
			case VIRTQC_cudaRegisterFatBinary:
				qcu_cudaRegisterFatBinary(arg);
				break;

			case VIRTQC_cudaUnregisterFatBinary:
				qcu_cudaUnregisterFatBinary(arg); 
				break;

			case VIRTQC_cudaRegisterFunction:
				qcu_cudaRegisterFunction(arg);
				break;

			case VIRTQC_cudaLaunch:
				qcu_cudaLaunch(arg);
				break;

			// Memory Management (runtime API)
			case VIRTQC_cudaMalloc:
				qcu_cudaMalloc(arg);
				break;

			case VIRTQC_cudaMemcpy:
				qcu_cudaMemcpy(arg);
				break;

			case VIRTQC_cudaFree:
				qcu_cudaFree(arg);
				break;

			// Device Management (runtime API)
			case VIRTQC_cudaGetDevice:
				qcu_cudaGetDevice(arg);
				break;

			case VIRTQC_cudaGetDeviceCount:
				qcu_cudaGetDeviceCount(arg);
				break;

			case VIRTQC_cudaSetDevice:
				qcu_cudaSetDevice(arg);
				break;

			case VIRTQC_cudaGetDeviceProperties:
				qcu_cudaGetDeviceProperties(arg);
				break;

			case VIRTQC_cudaDeviceSynchronize:
				qcu_cudaDeviceSynchronize(arg);
				break;

			case VIRTQC_cudaDeviceReset:
				qcu_cudaDeviceReset(arg);
				break;

			// Version Management (runtime API)
			case VIRTQC_cudaDriverGetVersion:
				qcu_cudaDriverGetVersion(arg);
				break;

			case VIRTQC_cudaRuntimeGetVersion:
				qcu_cudaRuntimeGetVersion(arg);
				break;

			// Event Management (runtime API)
			case VIRTQC_cudaEventCreate:
				qcu_cudaEventCreate(arg);
				break;

			case VIRTQC_cudaEventRecord:
				qcu_cudaEventRecord(arg);
				break;

			case VIRTQC_cudaEventSynchronize:
				qcu_cudaEventSynchronize(arg);
				break;

			case VIRTQC_cudaEventElapsedTime:
				qcu_cudaEventElapsedTime(arg);
				break;

			case VIRTQC_cudaEventDestroy:
				qcu_cudaEventDestroy(arg);
				break;

			// Error Handling (runtime API)
			case VIRTQC_cudaGetLastError:
				qcu_cudaGetLastError(arg);
				break;
#endif
			default:
				error("unknow cmd= %d, rnd= %d\n", arg->cmd, arg->rnd);
		}

		iov_from_buf(elem.in_sg, elem.in_num, 0, arg, sizeof(VirtioQCArg));
		virtqueue_push(vq, &elem, sizeof(VirtioQCArg));
		virtio_notify(vdev, vq);

		g_free(iov);
		free(arg);
	}
}

//####################################################################
//   class basic callback functions
//####################################################################

static void virtio_qcuda_device_realize(DeviceState *dev, Error **errp)
{
	VirtIODevice *vdev = VIRTIO_DEVICE(dev);
	VirtIOQC *qcu = VIRTIO_QC(dev);
	//Error *err = NULL;

	//ptrace("GPU mem size=%"PRIu64"\n", qcu->conf.mem_size);

	virtio_init(vdev, "virtio-qcuda", VIRTIO_ID_QC, sizeof(VirtIOQCConf));

	qcu->vq  = virtio_add_queue(vdev, 1024, virtio_qcuda_cmd_handle);
}

static uint64_t virtio_qcuda_get_features(VirtIODevice *vdev, uint64_t features, Error **errp)
{
	//ptrace("feature=%"PRIu64"\n", features);
	return features;
}

/*
   static void virtio_qcuda_device_unrealize(DeviceState *dev, Error **errp)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_get_config(VirtIODevice *vdev, uint8_t *config)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_set_config(VirtIODevice *vdev, const uint8_t *config)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_reset(VirtIODevice *vdev)
   {
   ptrace("\n");
   }

   static void virtio_qcuda_save_device(VirtIODevice *vdev, QEMUFile *f)
   {
   ptrace("\n");
   }

   static int virtio_qcuda_load_device(VirtIODevice *vdev, QEMUFile *f, int version_id)
   {
   ptrace("\n");
   return 0;
   }

   static void virtio_qcuda_set_status(VirtIODevice *vdev, uint8_t status)
   {
   ptrace("\n");
   }
 */

/*
   get the configure
ex: -device virtio-qcuda,size=2G,.....
DEFINE_PROP_SIZE(config name, device struce, element, default value)
 */
static Property virtio_qcuda_properties[] = 
{
	DEFINE_PROP_SIZE("size", VirtIOQC, conf.mem_size, 0),
	DEFINE_PROP_END_OF_LIST(),
};

static void virtio_qcuda_class_init(ObjectClass *klass, void *data)
{
	DeviceClass *dc = DEVICE_CLASS(klass);
	VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

	dc->props = virtio_qcuda_properties;

	set_bit(DEVICE_CATEGORY_MISC, dc->categories);

	vdc->get_features = virtio_qcuda_get_features;

	vdc->realize = virtio_qcuda_device_realize;
	/*	
		vdc->unrealize = virtio_qcuda_device_unrealize;

		vdc->get_config = virtio_qcuda_get_config;
		vdc->set_config = virtio_qcuda_set_config;

		vdc->save = virtio_qcuda_save_device;
		vdc->load = virtio_qcuda_load_device;

		vdc->set_status = virtio_qcuda_set_status;
		vdc->reset = virtio_qcuda_reset;
	 */	
}

static void virtio_qcuda_instance_init(Object *obj)
{
}

static const TypeInfo virtio_qcuda_device_info = {
	.name = TYPE_VIRTIO_QC,
	.parent = TYPE_VIRTIO_DEVICE,
	.instance_size = sizeof(VirtIOQC),
	.instance_init = virtio_qcuda_instance_init,
	.class_init = virtio_qcuda_class_init,
};

static void virtio_qcuda_register_types(void)
{
	type_register_static(&virtio_qcuda_device_info);
}

type_init(virtio_qcuda_register_types)

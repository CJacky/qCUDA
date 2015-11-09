#include "qemu-common.h"
#include "qemu/iov.h"
#include "qemu/error-report.h"
#include "sysemu/kvm.h"
#include "sysemu/kvm_int.h"
#include "hw/virtio/virtio.h"
#include "hw/virtio/virtio-bus.h"
#include "hw/virtio/virtio-qcuda.h"

#include "../../../../qcu-driver/qcuda_common.h"

#ifdef CONFIG_CUDA
#include <cuda.h>
#include <builtin_types.h>
#endif

#if 0
#define cjPrint(fmt, arg...) { \
		printf("### %-30s ,line: %-4d, ", __func__, __LINE__); \
		printf(fmt, ##arg); }
#else
#define cjPrint(fmt, arg...)
#endif

#define error(fmt, arg...) error_report("file %s ,line %d ,ERROR: "fmt, __FILE__, __LINE__, ##arg)

#ifndef MIN
#define MIN(a,b) ({ ((a)<(b))? (a):(b) })
#endif

CUdevice cuda_device;
CUcontext cuda_context;
CUmodule cuda_module;
CUfunction cuda_function;

size_t deviceSpaceSize = 0;
char *deviceSpace = NULL;

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

#define checkCudaErrors(err)  __checkCudaErrors(err, __LINE__) 
static void __checkCudaErrors(CUresult err, const int line)
{
	char *str;
	if ( err != CUDA_SUCCESS )
	{   
		cuGetErrorName(err, (const char**)&str);
		error("CUDA Driver API error = %04d \"%s\" line %d\n", err, str, line);
	}   
}

static void qcu_cudaRegisterFatBinary(VirtioQCArg *arg)
{
	time_begin();
	time_reset();
	cjPrint("\n");

	checkCudaErrors( cuInit(0) );
	checkCudaErrors( cuDeviceGet(&cuda_device, 0) );
	checkCudaErrors( cuCtxCreate(&cuda_context, 0, cuda_device) );

	memset(&cuda_module, 0, sizeof(CUmodule));
	memset(&cuda_function, 0, sizeof(CUfunction));

	time_add( qcu_TimeRegFatbin , time_end() );
}

static void qcu_cudaUnregisterFatBinary(VirtioQCArg *arg)
{
	time_begin();	
	cjPrint("\n");

	cuCtxDestroy(cuda_context);
	
	time_add( qcu_TimeUnFatbin , time_end() );
	time_print();
}

static void qcu_cudaRegisterFunction(VirtioQCArg *arg)
{
	void *fatBin;
	char *functionName;
	time_begin();
	
	cjPrint("\n");

	fatBin       = gpa_to_hva(arg->pA);
	functionName = gpa_to_hva(arg->pB);

	cjPrint("fatBin= %16p ,name= '%s'\n", fatBin, functionName);
	checkCudaErrors( cuModuleLoadData( &cuda_module, fatBin ));
	checkCudaErrors( cuModuleGetFunction(&cuda_function, cuda_module, functionName) );
	
	time_add( qcu_TimeRegFunc ,  time_end() );
}

static void qcu_cudaMalloc(VirtioQCArg *arg)
{
	size_t count;
	CUdeviceptr devPtr;
	time_begin();

	cjPrint("\n");

	count = arg->flag;;
	checkCudaErrors( cuMemAlloc( &devPtr, count ));	

	arg->pA = (uint64_t)devPtr;

	cjPrint("ptr= %p ,count= %lu\n", (void*)arg->pA, count);
	time_add( qcu_TimeMalloc , time_end() );
}


static void qcu_cudaMemcpyHostToDevice(VirtioQCArg *arg)
{
	CUdeviceptr dst;
	void *src;
	uint64_t *gpa_array;
	uint32_t size, len, i;
	time_begin();

	dst = (CUdeviceptr)arg->pA;
	size = arg->pBSize;

	if( size > QCU_KMALLOC_MAX_SIZE)
	{
		gpa_array = gpa_to_hva(arg->pB);
		for(i=0; size>0; i++)
		{
			src = gpa_to_hva(gpa_array[i]);
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			checkCudaErrors( cuMemcpyHtoD(dst, src, len));
			dst  += len;
			size -= len;
		}
	}
	else
	{
		src = gpa_to_hva(arg->pB);
		checkCudaErrors( cuMemcpyHtoD(dst, src, size));
	}

	cjPrint("devPtr= %p, size= %u\n", (void*)dst, size);
	time_add( qcu_TimeMemcpyH2D , time_end() );
}

static void qcu_cudaMemcpyDeviceToHost(VirtioQCArg *arg)
{
	void *dst;
	CUdeviceptr src;
	uint64_t *gpa_array;
	uint32_t size, len, i;
	time_begin();

	src = (CUdeviceptr)arg->pB;
	size = arg->pASize;

	if( size > QCU_KMALLOC_MAX_SIZE)
	{
		gpa_array = gpa_to_hva(arg->pA);
		for(i=0; size>0; i++)
		{
			dst = gpa_to_hva(gpa_array[i]);
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			checkCudaErrors( cuMemcpyDtoH(dst, src, len));
			src  += len;
			size -= len;
		}
	}
	else
	{
		dst = gpa_to_hva(arg->pA);
		checkCudaErrors( cuMemcpyDtoH(dst, src, size));
	}

	cjPrint("size= %u\n", size);
	time_add( qcu_TimeMemcpyD2H , time_end() );
}

static void qcu_cudaMemcpy(VirtioQCArg *arg)
{
	if( arg->flag == 1)
		qcu_cudaMemcpyHostToDevice(arg);
	else
		qcu_cudaMemcpyDeviceToHost(arg);
}

static void qcu_cudaFree(VirtioQCArg *arg)
{
	CUdeviceptr dst;
	time_begin();

	dst = (CUdeviceptr)arg->pA;
	checkCudaErrors( cuMemFree( dst ));

	cjPrint("ptr= %16p\n", (void*)dst);
	time_add( qcu_TimeFree , time_end() );
}

static void qcu_cudaLaunch(VirtioQCArg *arg)
{
	unsigned int *conf;
	uint64_t *para;
	size_t paraSize;
	void **paraBuf;
	int i;
	time_begin();

	conf = gpa_to_hva(arg->pA);
	para = gpa_to_hva(arg->pB);
	paraSize = (arg->pBSize)/sizeof(uint64_t);

	paraBuf = malloc(sizeof(void*)*paraSize);
	for(i=0; i<paraSize; i++)
		paraBuf[i] = &para[i];

	cjPrint("grid (%u %u %u) block(%u %u %u) sharedMem(%u)\n", 
			conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]);

	/*
	for(i=0; i<paraSize; i++)
		cjPrint("    para %d = %llu\n", i, *(unsigned long long*)paraBuf[i]);
	*/	

	checkCudaErrors( cuLaunchKernel(cuda_function, 
				conf[0], conf[1], conf[2],
				conf[3], conf[4], conf[5], 
				conf[6], NULL, paraBuf, NULL)); // not suppoer stream yeat
	
	free(paraBuf);
	time_add( qcu_TimeLaunch , time_end() );
}
#endif

static int qcu_cmd_write(VirtioQCArg *arg)
{
	void   *src, *dst;
	uint64_t *gpa_array;
	uint32_t size, len, i;

	size = arg->pASize;

	cjPrint("szie= %lu\n", size);

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
			cjPrint("deviceSpace[%lu]= %d\n", i, deviceSpace[i]);
		}
	}
	else
	{	
		err = 0;
		for(i=0; i<deviceSpaceSize; i++)
		{
			if( deviceSpace[i] != (i%17)*7 ) err++;
		}
		cjPrint("error= %llu\n", (unsigned long long)err);
	}
	cjPrint("\n\n");
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

	cjPrint("szie= %lu\n", size);
	
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

			case VIRTQC_cudaMalloc:
				qcu_cudaMalloc(arg);
				break;

			case VIRTQC_cudaMemcpy:
				qcu_cudaMemcpy(arg);
				break;

			case VIRTQC_cudaFree:
				qcu_cudaFree(arg);
				break;
#endif
			default:
				error("    unknow cmd= %d, rnd= %d\n", arg->cmd, arg->rnd);
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

	//cjPrint("GPU mem size=%"PRIu64"\n", qcu->conf.mem_size);

	virtio_init(vdev, "virtio-qcuda", VIRTIO_ID_QC, sizeof(VirtIOQCConf));

	qcu->vq  = virtio_add_queue(vdev, 1024, virtio_qcuda_cmd_handle);
}

static uint64_t virtio_qcuda_get_features(VirtIODevice *vdev, uint64_t features, Error **errp)
{
	//cjPrint("feature=%"PRIu64"\n", features);
	return features;
}

/*
   static void virtio_qcuda_device_unrealize(DeviceState *dev, Error **errp)
   {
   cjPrint("\n");
   }

   static void virtio_qcuda_get_config(VirtIODevice *vdev, uint8_t *config)
   {
   cjPrint("\n");
   }

   static void virtio_qcuda_set_config(VirtIODevice *vdev, const uint8_t *config)
   {
   cjPrint("\n");
   }

   static void virtio_qcuda_reset(VirtIODevice *vdev)
   {
   cjPrint("\n");
   }

   static void virtio_qcuda_save_device(VirtIODevice *vdev, QEMUFile *f)
   {
   cjPrint("\n");
   }

   static int virtio_qcuda_load_device(VirtIODevice *vdev, QEMUFile *f, int version_id)
   {
   cjPrint("\n");
   return 0;
   }

   static void virtio_qcuda_set_status(VirtIODevice *vdev, uint8_t status)
   {
   cjPrint("\n");
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

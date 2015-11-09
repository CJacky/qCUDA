/*
   The driver help send cuda function parameters to device.
   If the parameter is number, do nothing and send out.
   Otherwise, if the parameter is pointer address of data,
   you should copy data from user space to kernel space 
   and replace pointer address with gpa.
 */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/err.h>
#include <linux/virtio.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_pci.h>
#include <linux/scatterlist.h>
#include <linux/random.h>
#include <linux/io.h>

#include "qcuda_common.h"

#define error(fmt, arg...) printk("### func= %-30s ,line= %-4d ," fmt, __func__,  __LINE__, ##arg)

#if 0
#define cjPrint(fmt, arg...) \
	printk("### func= %-30s ,line= %-4d ," fmt, __func__,  __LINE__, ##arg)
#else
#define cjPrint(fmt, arg...)
#endif

#ifndef MIN
#define MIN(a,b) (((a)<(b))? (a):(b))
#endif

struct virtio_qcuda
{
	struct virtio_device	*vdev;
	struct virtqueue        *vq;
	spinlock_t	 	 	 	lock;
};

struct virtio_qcuda *qcu;

//####################################################################
//   helper functions
//####################################################################

static inline unsigned long copy_from_user_safe(void *to, const void __user *from, unsigned long n)
{
	unsigned long err;

	if( from==NULL || n==0 )
		return 0;

	err = copy_from_user(to, from, n);
	if( err ){
		error("copy_from_user is could not copy  %lu bytes\n", err);
		BUG_ON(1);
	}

	return err;
}

static inline unsigned long copy_to_user_safe(void __user *to, const void *from, unsigned long n)
{
	unsigned long err;

	if( to==NULL || n==0 )
		return 0;

	err = copy_to_user(to, from, n);
	if( err ){
		error("copy_to_user is could not copy  %lu bytes\n", err);
	}

	return err;
}

static inline void* kzalloc_safe(size_t size)
{
	void *ret;

	ret = kzalloc(size, GFP_KERNEL);
	if( !ret ){
		error("kzalloc failed, size= %lu\n", size);
		BUG_ON(1);
	}

	return ret;
}

static inline void* kmalloc_safe(size_t size)
{
	void *ret;

	ret = kmalloc(size, GFP_KERNEL);
	if( !ret ){
		error("kmalloc failed, size= %lu\n", size);
		BUG_ON(1);
	}

	return ret;
}

static void gpa_to_user(void *user, uint64_t gpa, uint32_t size)
{
	uint32_t i, len;
	uint64_t *gpa_array;
	void *gva;

	if(size > QCU_KMALLOC_MAX_SIZE)
	{
		gpa_array = (uint64_t*)phys_to_virt((phys_addr_t)gpa);
		for(i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);

			gva = phys_to_virt(gpa_array[i]);
			copy_to_user_safe(user, gva, len);

			user += len;
			size -= len;
		}
	}
	else
	{
		gva = phys_to_virt((phys_addr_t)gpa);
		copy_to_user_safe(user, gva, size);
	}
}

static uint64_t user_to_gpa_small(uint64_t from, uint32_t n)
{
	void *gva;

	cjPrint("from= %p, size= %u\n", (void*)from, n);

	gva = kmalloc_safe(n);

	if( from )
	{ // there is data needed to copy
		copy_from_user_safe(gva, (const void*)from, n);
	}

	return (uint64_t)virt_to_phys(gva);
}

static uint64_t user_to_gpa_large(uint64_t from, uint32_t size)
{
	uint32_t i, order, len;
	uint64_t *gpa_array;
	void *gva;

	cjPrint("from= %p, size= %u\n", (void*)from, size);

	order = (size >> QCU_KMALLOC_SHIFT_BIT) + 
		((size & (QCU_KMALLOC_MAX_SIZE-1)) > 0);

	gpa_array = (uint64_t*)kmalloc_safe( sizeof(uint64_t) * order );

	for(i=0; size>0; i++)
	{
		len = MIN(size, QCU_KMALLOC_MAX_SIZE);
		gva = kmalloc_safe(len);
		if(from)
		{
			copy_from_user_safe(gva, (const void*)from, len);
			from += len;
		}
		gpa_array[i] = (uint64_t)virt_to_phys(gva);
		size -= len;
	}

	return (uint64_t)virt_to_phys(gpa_array);
}


static uint64_t user_to_gpa(uint64_t from, uint32_t size)
{
	if(size > QCU_KMALLOC_MAX_SIZE)
		return user_to_gpa_large(from, size);
	else if( size > 0 )
		return user_to_gpa_small(from, size);
	else
		return from;
}

static void kfree_gpa(uint64_t pa, uint32_t size)
{
	uint64_t *gpa_array;
	uint32_t i, len;

	if(size > QCU_KMALLOC_MAX_SIZE)
	{
		cjPrint("large\n");
		gpa_array = (uint64_t*)phys_to_virt((phys_addr_t)pa);
		for( i=0; size>0; i++)
		{
			len = MIN(size, QCU_KMALLOC_MAX_SIZE);
			cjPrint("i= %u, len= %u, pa= %p\n", i, len, (void*)gpa_array[i]);
			kfree(phys_to_virt((phys_addr_t)gpa_array[i]));
			size -= len;
		}
	}
	cjPrint("phys= %p, virt= %p\n", (void*)pa, phys_to_virt((phys_addr_t)pa));
	kfree(phys_to_virt((phys_addr_t)pa));
}
//####################################################################
//   misc operations
//####################################################################

// Send VirtuiHMCmd to virtio device
// @req: struct include command and arguments
// if the function is corrent, it return 0. otherwise, is -1
static int qcu_misc_send_cmd(VirtioHMArg *req)
{
	struct scatterlist *sgs[2], req_sg, res_sg;
	unsigned int len;
	int err;
	VirtioHMArg *res;

	res = kmalloc_safe(sizeof(VirtioHMArg));
	memcpy(res, req, sizeof(VirtioHMArg));

	sg_init_one(&req_sg, req, sizeof(VirtioHMArg));
	sg_init_one(&res_sg, res, sizeof(VirtioHMArg));

	sgs[0] = &req_sg;
	sgs[1] = &res_sg;

	spin_lock(&qcu->lock);

	err =  virtqueue_add_sgs(qcu->vq, sgs, 1, 1, req, GFP_ATOMIC);
	if( err ){
		virtqueue_kick(qcu->vq);
		error("virtqueue_add_sgs failed\n");
		goto out;
	}

	if(unlikely(!virtqueue_kick(qcu->vq))){
		error("unlikely happen\n");
		goto out;
	}

	while (!virtqueue_get_buf(qcu->vq, &len) &&	!virtqueue_is_broken(qcu->vq))
		cpu_relax();

out:
	spin_unlock(&qcu->lock);

	memcpy(req, res, sizeof(VirtioHMArg));
	kfree(res);

	return err;
}

int qcu_cudaRegisterFatBinary(VirtioHMArg *arg)
{	// no extra parameters
	time_begin();
	cjPrint("\n");

	qcu_misc_send_cmd(arg);

	time_add( qcu_TimeRegFatbin, time_end() );
	return arg->cmd;
}

int qcu_cudaUnregisterFatBinary(VirtioHMArg *arg)
{	// no extra parameters
	time_begin();
	cjPrint("\n");

	qcu_misc_send_cmd(arg);

	time_add( qcu_TimeUnFatbin, time_end() );
	return arg->cmd;
}

int qcu_cudaRegisterFunction(VirtioHMArg *arg)
{	// pA: fatBin
	// pB: functrion name
	time_begin();
	cjPrint("function name= %s\n", (char*)arg->pB);

	arg->pA = user_to_gpa(arg->pA, arg->pASize);
	arg->pB = user_to_gpa(arg->pB, arg->pBSize);

	qcu_misc_send_cmd(arg);

	kfree_gpa(arg->pA, arg->pASize);
	kfree_gpa(arg->pB, arg->pBSize);

	time_add( qcu_TimeRegFunc, time_end() );
	return arg->cmd;
}

int qcu_cudaLaunch(VirtioHMArg *arg)
{	// pA: cuda kernel configuration
	// pB: cuda kernel parameters
	time_begin();
	cjPrint("\n");

	arg->pA = user_to_gpa(arg->pA, arg->pASize);
	arg->pB = user_to_gpa(arg->pB, arg->pBSize);

	qcu_misc_send_cmd(arg);

	kfree_gpa(arg->pA, arg->pASize);
	kfree_gpa(arg->pB, arg->pBSize);

	time_add( qcu_TimeLaunch, time_end() );
	return arg->cmd;
}

int qcu_cudaMalloc(VirtioHMArg *arg)
{	// pA: pointer of devPtr
	time_begin();
	cjPrint("\n");

	qcu_misc_send_cmd(arg);

	time_add( qcu_TimeMalloc, time_end() );
	return arg->cmd;
}

int qcu_cudaFree(VirtioHMArg *arg)
{	// pA: devPtr address
	time_begin();
	cjPrint("\n");

	qcu_misc_send_cmd(arg);

	time_add( qcu_TimeFree, time_end() );
	return arg->cmd;
}

int qcu_cudaMemcpy(VirtioHMArg *arg)
{	//pA: dst pointer 
	//pB: src pointer and size
	void* u_dst = NULL;
	time_begin();

	cjPrint("\n");

	cjPrint("pA= %p ,size= %u ,pB= %p, size= %u ,kind= %s\n", 
			(void*)arg->pA, arg->pASize, (void*)arg->pB, arg->pBSize,
			(arg->flag==1)? "H2D":"D2H");

	if( arg->flag == 1 )
	{
		arg->pA = user_to_gpa(arg->pA, arg->pASize); // device
		arg->pB = user_to_gpa(arg->pB, arg->pBSize); // host
	}
	else  // arg->flag == 2
	{
		u_dst = (void*)arg->pA;
		arg->pA = user_to_gpa(      0, arg->pASize); // host
		arg->pB = user_to_gpa(arg->pB, arg->pBSize); // device
	}

	qcu_misc_send_cmd(arg);

	if( arg->flag == 1 )
	{//cudaMemcpyHostToDevice:   device, host
		kfree_gpa(arg->pB, arg->pBSize);
	}
	else // arg->flag == 2
	{//cudaMemcpyDeviceToHost:  host, device
		gpa_to_user(u_dst, arg->pA, arg->pASize);
		kfree_gpa(arg->pA, arg->pASize);
	}

	time_add( qcu_TimeMemcpy, time_end() );
	return arg->cmd;
}

static int qcu_misc_write(VirtioHMArg *arg)
{	// pA: user src pointer and size

	arg->pA = user_to_gpa(arg->pA, arg->pASize);
	qcu_misc_send_cmd(arg);
	kfree_gpa(arg->pA, arg->pASize);
	return arg->cmd;
}

static int qcu_misc_read(VirtioHMArg *arg)
{	// pA: user buffer

	void *u_dst = NULL;

	arg->pA = user_to_gpa(0, arg->pASize);
	qcu_misc_send_cmd(arg);
	gpa_to_user(u_dst, arg->pA, arg->pASize);
	kfree_gpa(arg->pA, arg->pASize);

	return arg->cmd;
}

// @_cmd: device command
// @_arg: argument of cuda function
// this function reture cudaError_t.
static long qcu_misc_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg)
{
	VirtioHMArg *arg;
	int err;

	arg = kzalloc_safe(sizeof(VirtioHMArg));
	copy_from_user_safe(arg, (void*)_arg, sizeof(VirtioHMArg));
	//cjPrint("_arg= %p, arg= %p\n", (void*)_arg, arg);

	arg->cmd = _cmd;
	get_random_bytes(&arg->rnd, sizeof(int));

	switch(arg->cmd)
	{
		case VIRTHM_CMD_WRITE:
			//cjPrint("VIRTHM_cudaMalloc, rnd= %d\n", cmd->rnd);
			err = (int)qcu_misc_write(arg);
			break;

		case VIRTHM_CMD_READ:
			//cjPrint("VIRTHM_cudaMalloc, rnd= %d\n", cmd->rnd);
			err = (int)qcu_misc_read(arg);
			break;

		case VIRTHM_cudaRegisterFatBinary:
			qcu_cudaRegisterFatBinary(arg);
			break;

		case VIRTHM_cudaUnregisterFatBinary:
			qcu_cudaUnregisterFatBinary(arg);
			break;

		case VIRTHM_cudaRegisterFunction:
			qcu_cudaRegisterFunction(arg);
			break;

		case VIRTHM_cudaLaunch:
			qcu_cudaLaunch(arg);
			break;

		case VIRTHM_cudaMalloc:
			//cjPrint("VIRTHM_cudaMalloc, rnd= %d\n", cmd->rnd);
			err = qcu_cudaMalloc(arg);
			break;

		case VIRTHM_cudaMemcpy:
			//cjPrint("VIRTHM_cudaMemcpy, rnd= %d\n", cmd->rnd);
			err = qcu_cudaMemcpy(arg);
			break;

		case VIRTHM_cudaFree:
			//cjPrint("VIRTHM_cudaFree, rnd= %d\n", cmd->rnd);
			err = qcu_cudaFree(arg);
			break;

		default:
			error("unknow cmd= %d, rnd= %d\n", arg->cmd, arg->rnd);
			break;
	}

	copy_to_user_safe((void*)_arg, arg, sizeof(VirtioHMArg));

	err = arg->cmd;
	kfree(arg);
	return err;
}



static int qcu_misc_open(struct inode *inode, struct file *filp)
{
	return 0;
}

static int qcu_misc_release(struct inode *inode, struct file *filp)
{
	return 0;
}

struct file_operations qcu_misc_fops = {
	.owner   = THIS_MODULE,
	.open    = qcu_misc_open,
	.release = qcu_misc_release,
	.unlocked_ioctl = qcu_misc_ioctl,
};

static struct miscdevice qcu_misc_driver = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = "qcuda",
	.fops  = &qcu_misc_fops,
};


//####################################################################
//   virtio operations
//####################################################################


static void qcu_virtio_cmd_vq_cb(struct virtqueue *vq)
{
	/*
	   VirtioHMArg *cmd;
	   unsigned int len;

	   while( (cmd = virtqueue_get_buf(vq, &len))!=NULL ){	
	   cjPrint("read cmd= %d , rnd= %d\n", cmd->cmd, cmd->rnd);
	   }
	 */
}
/*
   static int qcu_virtio_init_vqs()
   {
   stract virtqueue *vqs[2];
   vq_callback_t *cbs[] = { qcu_virtio_in_vq_cb, qcu_virtio_out_vq_cb };
   const char *names[] = { "input_handle", "output_handle" };
   int err;

   err = qcu->vdev->config->find_vqs(qcu->vdev, 2, vqs, cbs, names);
   if( err ){
   cjPrint("find_vqs failed.\n");
   return err;
   }

   qcu->in_vq = vqs[0];
   qcu->out_vq= vqs[1];

   return 0;
   }

   static int qcu_virtio_remove_vqs()
   {
   qcu->vdev->config->del_vqs(qcu->vdev);
   kfree(qcu->in_vq);
   kfree(qcu->out_vq);
   }
 */
static int qcu_virtio_probe(struct virtio_device *vdev)
{
	int err;

	qcu = kzalloc_safe(sizeof(struct virtio_qcuda));
	if( !qcu ){
		err = -ENOMEM;
		goto err_kzalloc;
	}

	vdev->priv = qcu;
	qcu->vdev = vdev;

	qcu->vq = virtio_find_single_vq(vdev, qcu_virtio_cmd_vq_cb, 
			"request_handle");
	if (IS_ERR(qcu->vq)) {
		err = PTR_ERR(qcu->vq);
		error("init vqs failed.\n");
		goto err_init_vq;
	} 

	err = misc_register(&qcu_misc_driver);
	if (err)
	{
		error("virthm: register misc device failed.\n");
		goto err_reg_misc;
	}

	spin_lock_init(&qcu->lock);

	return 0;

err_reg_misc:
	vdev->config->del_vqs(vdev);
err_init_vq:
	kfree(qcu);
err_kzalloc:
	return err;
}

static void qcu_virtio_remove(struct virtio_device *vdev)
{
	int err;

	err = misc_deregister(&qcu_misc_driver);
	if( err ){
		error("misc_deregister failed\n");
	}

	qcu->vdev->config->reset(qcu->vdev);
	qcu->vdev->config->del_vqs(qcu->vdev);
	kfree(qcu->vq);
	kfree(qcu);
}

static unsigned int features[] = {};

static struct virtio_device_id id_table[] = {
	{ VIRTIO_ID_QCUDA, VIRTIO_DEV_ANY_ID },
	{ 0 },
};

static struct virtio_driver virtio_qcuda_driver = {
	.feature_table      = features,
	.feature_table_size = ARRAY_SIZE(features),
	.driver.name        = KBUILD_MODNAME,
	.driver.owner       = THIS_MODULE,
	.id_table           = id_table,
	.probe              = qcu_virtio_probe,
	.remove             = qcu_virtio_remove,
};


static int __init init(void)
{
	int ret;

	ret = register_virtio_driver(&virtio_qcuda_driver);
	if( ret < 0 ){
		error("register virtio driver faild (%d)\n", ret);
	}
	return ret;
}

static void __exit fini(void)
{
	unregister_virtio_driver(&virtio_qcuda_driver);
}

module_init(init);
module_exit(fini);

MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_DESCRIPTION("Qemu Virtio CUDA CUDA(qcu) driver");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("CJacky (Jacky Chen)");

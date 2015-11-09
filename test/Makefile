DUMP_FILE=1
NVFLAGS=-arch sm_35

RW=normal_rw

MEMCPY=memcpy
MEMCPY_TRUNK_WITH_BUF=memcpy_trunk_with_buf
MEMCPY_TRUNK_WITHOUT_BUF=memcpy_trunk_without_buf

MMUL_GPU=mmul_gpu
MMUL_GPU_DRV=mmul_gpu_drv
MMUL_CPU=mmul_cpu

MADD_GPU=madd_gpu
MADD_GPU_DRV=madd_gpu_drv
MADD_CPU=madd_cpu

all: $(RW)

######################################################################
$(MEMCPY): $(MEMCPY).cu
	nvcc $(NVFLAGS) --cudart=shared $(MEMCPY).cu -o $(MEMCPY) -DDUMP_FILE=$(DUMP_FILE)

$(MEMCPY_TRUNK_WITH_BUF): $(MEMCPY_TRUNK_WITH_BUF).cu
	nvcc $(NVFLAGS) --cudart=shared $(MEMCPY_TRUNK_WITH_BUF).cu -o $(MEMCPY_TRUNK_WITH_BUF)

$(MEMCPY_TRUNK_WITHOUT_BUF): $(MEMCPY_TRUNK_WITHOUT_BUF).cu
	nvcc $(NVFLAGS) --cudart=shared $(MEMCPY_TRUNK_WITHOUT_BUF).cu -o $(MEMCPY_TRUNK_WITHOUT_BUF)

######################################################################
$(MMUL_CPU): $(MMUL_CPU).c
	gcc -o $(MMUL_CPU) $(MMUL_CPU).c -DDUMP_FILE=$(DUMP_FILE)

$(MMUL_GPU_DRV).cubin: $(MMUL_GPU_DRV).cu
	nvcc $(NVFLAGS) -cubin $(MMUL_GPU_DRV).cu 

$(MMUL_GPU_DRV): $(MMUL_GPU_DRV).cubin $(MMUL_GPU_DRV).c  
	gcc -o $(MMUL_GPU_DRV) -I/usr/local/cuda/include $(MMUL_GPU_DRV).c -DDUMP_FILE=$(DUMP_FILE) -lcuda

$(MMUL_GPU): $(MMUL_GPU).cu
	nvcc $(NVFLAGS) --cudart=shared $(MMUL_GPU).cu -o $(MMUL_GPU) -DDUMP_FILE=$(DUMP_FILE)

######################################################################
$(MADD_CPU): $(MADD_CPU).c
	gcc -o $(MADD_CPU) $(MADD_CPU).c -DDUMP_FILE=$(DUMP_FILE)

$(MADD_GPU_DRV).cubin: $(MADD_GPU_DRV).cu
	nvcc $(NVFLAGS) -cubin $(MADD_GPU_DRV).cu 

$(MADD_GPU_DRV): $(MADD_GPU_DRV).cubin $(MADD_GPU_DRV).c  
	gcc -o $(MADD_GPU_DRV) -I/usr/local/cuda/include $(MADD_GPU_DRV).c -DDUMP_FILE=$(DUMP_FILE) -lcuda

$(MADD_GPU): $(MADD_GPU).cu
	nvcc $(NVFLAGS) --cudart=shared $(MADD_GPU).cu -o $(MADD_GPU) -DDUMP_FILE=$(DUMP_FILE)

######################################################################

$(RW): $(RW).c 
	gcc -o $(RW) $(RW).c

elasped: elapsed.c
	gcc -o elapsed elapsed.c

clean:
	rm -f $(MEMCPY) $(MEMCPY_TRUNK_WITH_BUF) $(MEMCPY_TRUNK_WITHOUT_BUF) \
		$(MMUL_GPU) $(MMUL_CPU) $(MMUL_GPU_DRV) \
		$(MADD_GPU) $(MADD_CPU) $(MADD_GPU_DRV) \
		$(RW) elapsed *.cubin *_out

#/usr/local/cuda/bin/nvcc --cudart=shared $(USR).c

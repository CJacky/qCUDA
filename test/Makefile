NVFLAGS=-arch sm_35

RW=normal_rw

MEMCPY=memcpy

MMUL_GPU=mmul_gpu
MMUL_CPU=mmul_cpu

MADD_GPU=madd_gpu
MADD_CPU=madd_cpu

all: $(MMUL_GPU)

######################################################################
$(MEMCPY): $(MEMCPY).cu
	nvcc $(NVFLAGS) --cudart=shared $(MEMCPY).cu -o $(MEMCPY)

######################################################################
$(MMUL_CPU): $(MMUL_CPU).c
	gcc -o $(MMUL_CPU) $(MMUL_CPU).c

$(MMUL_GPU): $(MMUL_GPU).cu
	nvcc $(NVFLAGS) --cudart=shared $(MMUL_GPU).cu -o $(MMUL_GPU)

######################################################################
$(MADD_CPU): $(MADD_CPU).c
	gcc -o $(MADD_CPU) $(MADD_CPU).c

$(MADD_GPU): $(MADD_GPU).cu
	nvcc $(NVFLAGS) --cudart=shared $(MADD_GPU).cu -o $(MADD_GPU)

######################################################################

$(RW): $(RW).c 
	gcc -o $(RW) $(RW).c

elasped: elapsed.c
	gcc -o elapsed elapsed.c

clean:
	rm -f $(MEMCPY) $(RW) elapsed *.cubin *_out \
		$(MMUL_GPU) $(MMUL_CPU) \
		$(MADD_GPU) $(MADD_CPU)


INC=-I/usr/local/cuda/include
LIB=-L/usr/local/cuda/lib64
lib=-lcuda -lcudart
NVCCFLAG=-arch sm_35 --cudart=shared

all: bw mmul vadd

bw: bw.cu
	nvcc $(NVCCFLAG) -o $@ $<

mmul: mmul.cu
	nvcc $(NVCCFLAG) -o $@ $<

vadd: vadd.cu
	nvcc $(NVCCFLAG) -o $@ $<

clean:
	rm -f bw mmul vadd


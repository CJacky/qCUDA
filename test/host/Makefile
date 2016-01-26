INC=-I/usr/local/cuda/include
LIB=-L/usr/local/cuda/lib64
lib=-lcuda -lcudart

all: aio

bw.o: bw.c
	gcc $(INC) -o $@ -c $<

mmul.fatbin: mmul.cu
	nvcc -fatbin -arch sm_35 $<
	
mmul.o: mmul.c mmul.fatbin
	gcc $(INC) -o $@ -c $<

vadd.fatbin: vadd.cu
	nvcc -fatbin -arch sm_35 $<

vadd.o: vadd.c vadd.fatbin
	gcc $(INC) -o $@ -c $<
	
aio.o: aio.c
	gcc $(INC) -o $@ -c $<

aio: bw.o  aio.o mmul.o vadd.o
	gcc $(INC) $(LIB) -o $@ $+ $(lib)

clean:
	rm -f *.o *.fatbin aio


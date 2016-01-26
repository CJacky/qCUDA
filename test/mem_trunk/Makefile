INC=-I/usr/local/cuda/include
LIB=-L/usr/local/cuda/lib64
lib=-lcuda -lcudart

all: native withBuf withoutBuf

native: native.c
	gcc $(INC) $(LIB) -o $@ $< $(lib)

withBuf: withBuf.c
	gcc $(INC) $(LIB) -o $@ $< $(lib)

withoutBuf: withoutBuf.c
	gcc $(INC) $(LIB) -o $@ $< $(lib)

clean:
	rm -f native withBuf withoutBuf


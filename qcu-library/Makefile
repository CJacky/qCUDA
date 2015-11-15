
CUDA_INC=/usr/local/cuda/include

LIB=libcudart_dl

all: lib link

libcudart_dl:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o -lcuda -ldl

libcudart_cu:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o -lcuda

libcudart:
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o libcudart.so $(LIB).o

lib: $(LIB)

link: 
	sudo cp libcudart.so /usr/local/lib/
	sudo ln -sf libcudart.so /usr/local/lib/libcudart.so.7.5
	sudo ldconfig

clean:
	rm -f *.so *.o 

remove:
	sudo rm -f /usr/local/lib/libcudart.so /usr/local/lib/libcudart.so.7.5
	sudo ldconfig

#/usr/local/cuda/bin/nvcc --cudart=shared $(USR).c

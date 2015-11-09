
CUDA_INC=/usr/local/cuda/include

LIB=libcudart

all: lib link

lib: $(LIB).c
	gcc -c -Wall -Werror -fpic -I$(CUDA_INC) $(LIB).c
	gcc -shared -o $(LIB).so $(LIB).o

link: 
	sudo cp $(LIB).so /usr/local/lib/
	sudo ln -sf $(LIB).so /usr/local/lib/libcudart.so.7.5
	sudo ldconfig

clean:
	rm -f *.so *.o 

remove:
	sudo rm -f /usr/local/lib/$(LIB).so /usr/local/lib/libcudart.so.7.5
	sudo ldconfig

#/usr/local/cuda/bin/nvcc --cudart=shared $(USR).c

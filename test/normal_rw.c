#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/mman.h> // mmap

#include "../vhm-driver/virtio_hm.h"
#include <stdio.h>
#include <ctype.h> // isalpha
#include <sys/time.h>
#include <unistd.h>
#define time_define() struct timeval timeval_begin
#define time_begin() gettimeofday (&timeval_begin, NULL);
#define time_end() ({ \
	struct timeval timeval_end; \
	gettimeofday (&timeval_end, NULL); \
	(double)((timeval_end.tv_sec-timeval_begin.tv_sec)+((timeval_end.tv_usec-timeval_begin.tv_usec)/1000000.0)); \
	})

#define KB (1UL<<10)
#define MB (1UL<<20)
#define GB (1UL<<30)

void get_data_size(int *N, size_t *size, char u[4], int argc, char* argv[])
{
	if( argc==2 ){
		*N = atoi(argv[1]);

		if( isalpha( argv[1][strlen(argv[1])-1]) )
		{
			switch( argv[1][strlen(argv[1])-1] )
			{
				case 'k':case'K': *size = (*N) * KB * sizeof(char); strcpy(u, "KB");  break;
				case 'm':case'M': *size = (*N) * MB * sizeof(char); strcpy(u, "MB");  break;
				case 'g':case'G': *size = (*N) * GB * sizeof(char); strcpy(u, "GB");  break;
			}
		}
		else
		{
			*size = (*N) * sizeof(char);
			strcpy(u, "Byte");
		}
	}
	else
	{
		*N = *size = 10;
		strcpy(u, "Byte");
	}
}

void test_read(char *dst, size_t size)
{

}

void test_write(char *src, size_t size)
{
	VirtioHMArg *arg;
	int fd;
	size_t i;
	
	arg = (VirtioHMArg*)malloc(sizeof(VirtioHMArg));
	fd = open("/dev/virthm", O_RDWR);

	if( fd<0 )
	{
		printf("open device (/dev/virthm) faild, %s (%d)\n", strerror(errno), errno);
		exit(-1);
	}
	
	arg->pA  = (uint64_t)src;
	arg->pASize = size;

	// call driver to write
	for(i=0; i<size && size<32; i++)
		printf("str[%lu]= %d\n", i, src[i]);

	ioctl(fd, VIRTHM_CMD_WRITE, arg);

	free(arg);
}

int main(int argc, char* argv[])
{
	double time_write, time_read;
	char unit[4];
	char *buf;
	int i, N;
	size_t size;
	uint64_t idx;
	time_define();
	
	get_data_size(&N, &size, unit, argc, argv);

//##########################################################
	buf = (char*)malloc(size);
	for(i=0, idx=0; i<size; i++){
		buf[i] = 7*(i%17);
		idx++;
	}
	time_begin();
	test_write(buf, size);
	time_end();

	free(buf);
//##########################################################
/*
	// prepare read buf
	buf = (char*)malloc(size);
	memset(buf, 0, size); 

	// put parameters into VirtioHMArg
	arg.dst  = buf;
	arg.size = size;

	// call driver to read
	time_begin();
	ioctl(fd, VIRTHM_CMD_READ, &arg);
	time_read = time_end();
	printf("read : '%s'\n", buf);

	free(buf);
*/
//##########################################################
	/*
	printf("N= %d(%s), size= %lu Byte\n", N, unit, size);
	printf("write= %f MB/sec, %f sec\n", size/(MB*time_write), time_write);
	printf("read = %f MB/sec, %f sec\n", size/(MB*time_read ), time_read );
	*/
	//printf("%.3f\t%.3f\t%.3f\t\n", (double)size/KB, (double)size/(MB*time_h2d), (double)size/(MB*time_d2h));
	return 0;
}

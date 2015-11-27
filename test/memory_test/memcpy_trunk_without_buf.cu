#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

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

#define hash(i) (unsigned char)((i)%256)
#define TRUNK_SIZE (1<<20)

#ifndef MIN
#define MIN(a,b) (((a)<(b))? (a):(b))
#endif

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
int main(int argc, char* argv[])
{
	double time_h2d, time_d2h;
	size_t size, err_num, len, size_cpy;
	int i, N;
	char unit[4];
	unsigned char *d, *h, *d_cpy, *h_cpy;
	cudaError_t err;

	get_data_size(&N, &size, unit, argc, argv);

	time_define();

	h = (unsigned char*)malloc(size);

	err = cudaMalloc((void**)&d, size);
	if( err != cudaSuccess){
		printf("malloc failed (%d)\n", err);
	}

	for(i=0; i<size; i++){
		h[i] = hash(i);
	}

	//#######################################################
	time_begin();
	size_cpy = size;
	d_cpy = d;
	h_cpy = h;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		err = cudaMemcpy(d_cpy, h_cpy, len, cudaMemcpyHostToDevice );
		size_cpy -= len;
		d_cpy += len;
		h_cpy += len;

		if( err != cudaSuccess){
			printf("H2D fa1iled i= %d, err= %d\n", i, err);
		}
	}
	time_h2d = time_end();
	//#######################################################

	memset(h, 0, size); // kernel

	//#######################################################
	time_begin();
	size_cpy = size;
	d_cpy = d;
	h_cpy = h;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		err = cudaMemcpy(h_cpy, d_cpy, len, cudaMemcpyDeviceToHost );
		size_cpy -= len;
		d_cpy += len;
		h_cpy += len;

		if( err != cudaSuccess){
			printf("D2H fa1iled i= %d, err= %d\n", i, err);
		}
	}
	time_d2h = time_end();
	//#######################################################

	if(size<32)
	{
		for(i=0; i<size; i++){
			printf("h[%d] = %d\n", i, h[i]);
		}
	}else{
		err_num=0;
		for(i=0; i<size; i++)
		{
			if( h[i] != hash(i))
			{
				if(err_num<20)
				{
					printf("h[%d]=%d\n", i, h[i]);
				}
				err_num++;
			}
		}
		if( err_num>0 )
			printf("error num= %lu\n", err_num);
	}


#if 0
	printf("N= %d(%s), size= %lu Byte\n", N, unit, size);
	printf("H2D= %f MB/sec, %f sec\n", size/(MB*time_h2d), time_h2d);
	printf("D2H= %f MB/sec, %f sec\n", size/(MB*time_d2h), time_d2h);
#else
	//printf("trunk size %u\n", TRUNK_SIZE);
	printf("%.3f\t%.3f\t%.3f\t\n", (double)size/KB, (double)size/(MB*time_h2d), (double)size/(MB*time_d2h));
#endif

	err = cudaFree(d);
	if( err != cudaSuccess){
		printf("free failed (%d)\n", err);
	}

	free(h);

	return 0;
}

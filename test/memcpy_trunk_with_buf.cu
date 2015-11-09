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
	size_t size, err_num, len, size_cpy, order, idx;
	int i, j, N;
	char unit[4];
	unsigned char *d, **h, *buf, *h_cpy;
	cudaError_t err;
	time_define();

	get_data_size(&N, &size, unit, argc, argv);

	order = (size/TRUNK_SIZE) + ((size%TRUNK_SIZE)>0);

	h = (unsigned char**)malloc(order*sizeof(void*));
	idx = 0;
	size_cpy = size;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		h[i] = (unsigned char*)malloc(len*sizeof(unsigned char));
		for(j=0; j<len; j++){
			h[i][j] = hash(idx);
			idx++;
		}
		size_cpy -= len;
	}

	err = cudaMalloc((void**)&d, size);
	if( err != cudaSuccess){
		printf("malloc failed (%d)\n", err);
	}

	//#######################################################
	time_begin();
	buf = (unsigned char*)malloc(size);
	size_cpy = size;
	h_cpy = buf;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		memcpy(h_cpy, h[i], len);
		h_cpy += len;
		size_cpy -= len;
	}

	err = cudaMemcpy(d, buf, size, cudaMemcpyHostToDevice );
	if( err != cudaSuccess){
		printf("H2D fa1iled err= %d\n", err);
	}
	free(buf);
	time_h2d = time_end();
	//#######################################################

	size_cpy = size;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		memset(h[i], 0, len); 
		size_cpy -= len;
	}


	//#######################################################
	time_begin();
	buf = (unsigned char*)malloc(size);
	err = cudaMemcpy(buf, d, size, cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){
		printf("D2H fa1iled i= %d, err= %d\n", i, err);
	}

	size_cpy = size;
	h_cpy = buf;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		memcpy(h[i], h_cpy, len);
		size_cpy -= len;
		h_cpy += len;
	}
	free(buf);
	cudaFree(d);
	time_d2h = time_end();
	//#######################################################
	err_num=0;
	idx = 0;
	size_cpy = size;
	for(i=0; size_cpy>0; i++)
	{
		len = MIN(size_cpy, TRUNK_SIZE);
		for(j=0; j<len; j++){
			if( h[i][j] != hash(idx))
			{
				if(err_num<20)
				{
					printf("h[%d][%d]=%d\n", i, j, h[i][j]);
				}
				err_num++;
			}
			idx++;
		}
		size_cpy -= len;
	}
	for(i=0; i<order; i++)
		free(h[i]);
	free(h);
	if( err_num>0 )
		printf("error num= %lu\n", err_num);


#if 0
	printf("N= %d(%s), size= %lu Byte\n", N, unit, size);
	printf("H2D= %f MB/sec, %f sec\n", size/(MB*time_h2d), time_h2d);
	printf("D2H= %f MB/sec, %f sec\n", size/(MB*time_d2h), time_d2h);
#else
	//printf("trunk size %u\n", TRUNK_SIZE);
	printf("%.3f\t%.3f\t%.3f\t\n", (double)size/KB, (double)size/(MB*time_h2d), (double)size/(MB*time_d2h));
#endif

	return 0;
}

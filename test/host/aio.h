
#include <cuda_runtime.h>

struct timeval timeval_begin, timeval_end;


#define time_begin() gettimeofday (&timeval_begin, NULL)
#define time_end() 	({\
	gettimeofday (&timeval_end, NULL); \
	(unsigned int)((timeval_end.tv_sec  - timeval_begin.tv_sec)*1000000 + \
					   (timeval_end.tv_usec - timeval_begin.tv_usec)); })


void cudaInit(void);
void cudaFini(void);

void cudaRegFunc(char*, char*);
void cudaExecFunc(int, int, int, int, int, int, int, void**);

void bw(uint64_t);
void mmul(uint64_t);
void vadd(uint64_t);

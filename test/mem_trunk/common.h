
#include <sys/time.h>

struct timeval timeval_begin, timeval_end;

#define time_begin() gettimeofday (&timeval_begin, NULL)
#define time_end() 	({\
	gettimeofday (&timeval_end, NULL); \
	(unsigned int)((timeval_end.tv_sec  - timeval_begin.tv_sec)*1000000 + \
					   (timeval_end.tv_usec - timeval_begin.tv_usec)); })

#define TRUNK_SIZE (1<<22)
#define MIN(a,b) (((a)<(b))? a:b)
#define hash(i) (uint8_t)((i)%255)

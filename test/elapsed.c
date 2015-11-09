#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>
#define time_define() struct timeval timeval_begin
#define time_begin() gettimeofday (&timeval_begin, NULL);
#define time_end() ({ \
	struct timeval timeval_end; \
	gettimeofday (&timeval_end, NULL); \
	(((timeval_end.tv_sec-timeval_begin.tv_sec)*1000000)+(timeval_end.tv_usec-timeval_begin.tv_usec)); \
	})

int main(int argc, char* argv[])
{
	char cmd[512]="", buf[32];
	time_define();
	int i;

	for(i=1; i<argc; i++)
	{
		sprintf(buf, "%s ", argv[i]);
		strcat(cmd, buf);
	}

#if 0
	printf("%s\n", cmd);
#else
	time_begin();
	system(cmd);
	printf("%lu", time_end());
#endif

	return 0;
}

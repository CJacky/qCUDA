#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int main(int argc, char* argv[])
{
	char cmd[512]="", buf[32];
	struct timeval begin, end;
	int i;

	for(i=1; i<argc; i++)
	{
		sprintf(buf, "%s ", argv[i]);
		strcat(cmd, buf);
	}

#if 0
	printf("%s\n", cmd);
#else
	gettimeofday (&begin, NULL);
	system(cmd);
	gettimeofday (&end, NULL);

	printf("%lu", (((end.tv_sec -begin.tv_sec)*1000000)+
				    (end.tv_usec-begin.tv_usec)));
#endif

	return 0;
}

#include <stdio.h>
#include <stdlib.h>

/*
#include <sys/time.h>
#define time_define() struct timeval timeval_begin
#define time_begin() gettimeofday (&timeval_begin, NULL);
#define time_end() ({ \
	struct timeval timeval_end; \
	gettimeofday (&timeval_end, NULL); \
	(double)((timeval_end.tv_sec-timeval_begin.tv_sec)+((timeval_end.tv_usec-timeval_begin.tv_usec)/1000000.0)); \
	})
*/

#define a(i, j) ((i)*y+(j))
#define b(i, j) ((i)*z+(j))
#define c(i, j) ((i)*z+(j))

#ifndef DUMP_FILE
#define DUMP_FILE 0
#endif

typedef int elem_t;

int main(int argc, char* argv[])
{
	int x, y, z; // matrix dim  xy, yz, xz
	int i, j, k;
	elem_t *A;
	elem_t *B;
	elem_t *C;
	//time_define();

	x = (argc>=2)? atoi(argv[1]):3;
	y = (argc>=3)? atoi(argv[2]):3;
	z = (argc>=4)? atoi(argv[3]):3;
	
	if( x<=0 || y<=0 || z<=0){
		printf("dim error, x= %d, y= %d, z= %d\n", x, y, z);
		return -1;
	}

	// malloc matrix memory in host
	A = (elem_t*)malloc( x*y*sizeof(elem_t));
	B = (elem_t*)malloc( y*z*sizeof(elem_t));
	C = (elem_t*)malloc( x*z*sizeof(elem_t));
	
	// init matrix 
	for(i=0; i<x*y; i++) A[i] = rand()%10;
	for(i=0; i<y*z; i++) B[i] = rand()%10;

//********************************************************************
	//time_begin();

	// calculate
	for(i=0; i<x; i++)
		for(j=0; j<z; j++){
			C[c(i,j)] = 0;
			for(k=0; k<y; k++)
				C[c(i,j)] += A[a(i,k)]*B[b(k,j)];
		}

	//printf("%f\n", time_end());
//********************************************************************
#if DUMP_FILE
	FILE *f = fopen("mmul_cpu_out", "w");

	fprintf(f, "%d %d %d\n", x, y, z);
	for(i=0; i<((x>y)?x:y); i++)
	{
		for(j=0; j<y; j++){
			if(i<x)	fprintf(f, "%2d ", A[a(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "  ");

		for(j=0; j<z; j++){
			if(i<y) fprintf(f, "%2d ", B[b(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "    ");

		for(j=0; j<z; j++){
			if(i<x) fprintf(f, "%4d ", C[c(i,j)]);
			else    fprintf(f, "%4c ", ' ');
		}
		fprintf(f, "\n");
	}
	fclose(f);
#endif
	free(A);
	free(B);
	free(C);

	return 0;
}

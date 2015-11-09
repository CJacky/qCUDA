#include <stdio.h>
#include <stdlib.h>

#define m(i, j) ((i)*y+(j))

#ifndef DUMP_FILE
#define DUMP_FILE 0
#endif

void matrixAdd(int *A, int *B, int x, int y)
{
	size_t i, j;

	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
			A[m(i,j)] += B[m(i,j)];
	}
}

void print_matrix(int *M, int x, int y, FILE *f)
{
	int i, j;

	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
		{
			fprintf(f, "%2d ", M[m(i,j)]);
		}
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
}

int main(int argc, char* argv[])
{
	int *A, *B;
	int x, y, n; // matrix dim  xy, yz, xz
	size_t i, j;

	n = (argc>=2)? atoi(argv[1]):2;
	x = (argc>=3)? atoi(argv[2]):3;
	y = (argc>=4)? atoi(argv[3]):3;

	if( x<=0 || y<=0 || n<2){
		printf("dim error, n= %d, x= %d, y= %d\n", n, x, y);
		return -1;
	}

	A = (int*)malloc( x*y*sizeof(int));
	B = (int*)malloc( x*y*sizeof(int));
	
	for(j=0; j<x*y; j++) A[j] = rand()%10;

	for(i=1; i<n; i++)
	{
		for(j=0; j<x*y; j++) B[j] = rand()%10;
		matrixAdd(A, B, x, y);
	}
#if DUMP_FILE
	FILE *f = fopen("madd_cpu_out", "w");
	print_matrix(A, x, y, f);
	fclose(f);
#endif

	free(A);
	free(B);

	return 0;
}

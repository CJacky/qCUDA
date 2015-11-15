#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define a(i, j) ((i)*y+(j))
#define b(i, j) ((i)*z+(j))
#define c(i, j) ((i)*z+(j))

#ifndef DUMP_FILE
#define DUMP_FILE 0
#endif

typedef int elem_t;

__global__ void matrixMul(int *A, int *B, int *C, int x, int y, int z)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cul = blockIdx.x*blockDim.x + threadIdx.x;
	int k, val;

	if( row<x && cul<z )
	{
		val = 0;
		for(k=0; k<y; k++)
			val += A[a(row, k)]*B[b(k, cul)];
		C[c(row, cul)] = val;
	}
}

int main(int argc, char* argv[])
{
	int x, y, z; // matrix dim  xy, yz, xz
	int i;
	elem_t *A_h, *A_d;
	elem_t *B_h, *B_d;
	elem_t *C_h, *C_d;

	x = (argc>=2)? atoi(argv[1]):3;
	y = (argc>=3)? atoi(argv[2]):3;
	z = (argc>=4)? atoi(argv[3]):3;
	
	if( x<=0 || y<=0 || z<=0){
		printf("dim error, x= %d, y= %d, z= %d\n", x, y, z);
		return -1;
	}


	// malloc matrix memory in host
	A_h = (elem_t*)malloc( x*y*sizeof(elem_t));
	B_h = (elem_t*)malloc( y*z*sizeof(elem_t));
	C_h = (elem_t*)malloc( x*z*sizeof(elem_t));


	// malloc matrix memory in device
	cudaMalloc(&A_d, x*y*sizeof(elem_t));
	cudaMalloc(&B_d, y*z*sizeof(elem_t));
	cudaMalloc(&C_d, x*z*sizeof(elem_t));


	// init matrix 
	for(i=0; i<x*y; i++) A_h[i] = rand()%10;
	for(i=0; i<y*z; i++) B_h[i] = rand()%10;
	//for(int i=0; i<x*z; i++) C_h[i] = 0;


	// copy matrix from host to device
	cudaMemcpy(A_d, A_h, x*y*sizeof(elem_t), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, y*z*sizeof(elem_t), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, x*z*sizeof(elem_t), cudaMemcpyHostToDevice);


	dim3 threads(32, 32);
	dim3 blocks( (x%32)?x/32+1:x/32, (z%32)?z/32+1:z/32);
	

	matrixMul <<< blocks, threads >>>(A_d, B_d, C_d, x, y, z);


	cudaMemcpy(C_h, C_d, x*z*sizeof(elem_t), cudaMemcpyDeviceToHost);

//********************************************************************

#if DUMP_FILE
	FILE *f = fopen("mmul_gpu_out", "w");
	int j;

	fprintf(f, "%d %d %d\n", x, y, z);
	for(i=0; i<((x>y)?x:y); i++)
	{
		for(j=0; j<y; j++){
			if(i<x)	fprintf(f, "%2d ", A_h[a(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "  ");

		for(j=0; j<z; j++){
			if(i<y) fprintf(f, "%2d ", B_h[b(i,j)]);
			else    fprintf(f, "%2c ", ' ');
		}
		fprintf(f, "    ");

		for(j=0; j<z; j++){
			if(i<x) fprintf(f, "%4d ", C_h[c(i,j)]);
			else    fprintf(f, "%4c ", ' ');
		}
		fprintf(f, "\n");
	}
	fclose(f);
#endif

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);


	free(A_h);
	free(B_h);
	free(C_h);
//	cudaDeviceReset();
	return 0;
}

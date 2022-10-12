#include "d_utils.h"

// #include "utils.hpp"
// #include<stdio.h>
// #include<stdlib.h>

__global__ void device_add(int *a, int *b, int *c) {

        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
//basically just fills the array with index.
void fill_array(int *data) {
	for(int idx=0;idx<512;idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
	for(int idx=0;idx<512;idx++)
        cout<<a[idx] <<", "<< b[idx]<<", "<< c[idx];
		// printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}

void __add_da()
{
    cout<<"im here __add"<<endl;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device copies of a, b, c

	int size = 512 * sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size); fill_array(b);
	c = (int *)malloc(size);

        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

       // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


	device_add<<<1,512>>>(d_a,d_b,d_c);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	print_output(a,b,c);

	free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}
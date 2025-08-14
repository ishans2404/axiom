#include <cuda_runtime.h>

// Simple GPU kernel
__global__ void add_kernel(const int* a, const int* b, int* c) {
    *c = *a + *b;
}

extern "C" int add_gpu_impl(int a, int b) {
    int *d_a, *d_b, *d_c;
    int result;

    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    add_kernel<<<1, 1>>>(d_a, d_b, d_c);
    cudaMemcpy(&result, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

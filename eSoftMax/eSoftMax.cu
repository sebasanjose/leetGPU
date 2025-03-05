#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // TODO: Implement the softmax kernel.  Remember to use the max trick!
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
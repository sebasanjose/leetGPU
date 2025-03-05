#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Calculate thread ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < N) {
        // First step: find the maximum value in the array
        float max_val = input[0];
        for (int i = 1; i < N; i++) {
            max_val = max(max_val, input[i]);
        }
        
        // Calculate exp(x_i - max) for this thread's element
        float exp_val = expf(input[tid] - max_val);
        
        // Store in output temporarily
        output[tid] = exp_val;
        
        // We need all threads to finish calculating their exp values before summing
        __syncthreads();
        
        // Calculate the sum of all exp values
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += output[i];
        }
        
        // Normalize by dividing by the sum
        output[tid] = exp_val / sum;
    }
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
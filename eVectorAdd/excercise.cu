#include <cuda_runtime.h>
#include <iostream>


__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < N ) {
        C[idx] = A[idx] + B[idx];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void main() {
    int N = 4;
    float A[N] = {1, 2, 3, 4};
    float B[N] = {1, 2, 3, 4};

    float C[N];

    solve(A, B, C, N);

    std::cout << "Result: ";
    for (int i=0; i<N; i++) {
        std::cout << C[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}

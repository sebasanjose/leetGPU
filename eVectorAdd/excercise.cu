#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_CALL(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
        printf("Thread %d: A=%f, B=%f, C=%f\n", idx, A[idx], B[idx], C[idx]);
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CHECK_CUDA_CALL(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_B, N * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA_CALL(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_CALL(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_CALL(cudaFree(d_A));
    CHECK_CUDA_CALL(cudaFree(d_B));
    CHECK_CUDA_CALL(cudaFree(d_C));
}

int main() {
    const int N = 4;
    float A[N] = {1.0, 2.0, 3.0, 4.0};
    float B[N] = {5.0, 6.0, 7.0, 8.0};
    float C[N];

    solve(A, B, C, N);

    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

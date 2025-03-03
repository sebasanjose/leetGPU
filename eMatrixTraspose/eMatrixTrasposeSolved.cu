#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for transposing a matrix
__global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// Function to perform matrix transpose using CUDA
void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, rows * cols * sizeof(float), d_output, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to print a matrix
void print_matrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int rows = 2, cols = 3;
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3 matrix
    float output[6]; // Transposed 3x2 matrix

    std::cout << "Input Matrix:" << std::endl;
    print_matrix(input, rows, cols);

    // Transpose the matrix
    solve(input, output, rows, cols);

    std::cout << "Transposed Matrix:" << std::endl;
    print_matrix(output, cols, rows);

    return 0;
}

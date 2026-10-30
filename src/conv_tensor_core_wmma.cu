#include <cuda_fp16.h>
#include <mma.h>
#include "gpu-new-forward.h"
#include <cstdio>     // for printf (works everywhere)
#include <iostream> 
#include "matmul.h"
using namespace nvcuda;
using namespace nvcuda::wmma;  // <- precision::tf32 lives here

#define PERMUTE_BLOCK_SIZE 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8                 // <-- TF32 requires K = 8
#define TILE   WMMA_M 

__global__ void matrixMultiplyTF32(const float *A, const float *B, float *C,
    int numARows,  int numAColumns,
    int numBRows,  int numBColumns,
    int numCRows,  int numCColumns)
{
    if (threadIdx.x >= 32) return;          // one warp per block

    __shared__ float A_tile[WMMA_M][WMMA_K];
    __shared__ float B_tile[WMMA_K][WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x;
    size_t tileRow = by * WMMA_M;
    size_t tileCol = bx * WMMA_N;

    // --------------------------------------------------------------------
    // sweep K dimension in tiles of 8 (WMMA_K)
    // --------------------------------------------------------------------
    for (int tileId = 0; tileId < (numAColumns + WMMA_K - 1) / WMMA_K; ++tileId) {

        // 1. ---------------- load A tile into shared --------------------

        /* Each warp thread loads **4 elements** so the whole 16×8 tile is filled.
           Index inside the tile: idx ∈ [0,127] → row = idx/WMMA_K , col = idx%WMMA_K */
        for (int idx = tx; idx < WMMA_M * WMMA_K; idx += 32) {       // FIX
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            size_t aRow = tileRow + r;
            size_t aCol = tileId * WMMA_K + c;
            A_tile[r][c] = (aRow < numARows && aCol < numAColumns)
                           ? A[aRow * numAColumns + aCol]
                           : 0.0f;                                   // zero‑pad
        }

        // 2. ---------------- load B tile into shared --------------------
        for (int idx = tx; idx < WMMA_K * WMMA_N; idx += 32) {       // FIX
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            size_t bRow = tileId * WMMA_K + r;
            size_t bCol = tileCol + c;
            B_tile[r][c] = (bRow < numBRows && bCol < numBColumns)
                           ? B[bRow * numBColumns + bCol]
                           : 0.0f;
        }

        __syncthreads();                                             // FIX

        // 3. ---------------- WMMA operations ---------------------------
        wmma::load_matrix_sync(a_frag, &A_tile[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &B_tile[0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();                                             // make sure shared is free
    }

    // 4. ---------------- store back to C -------------------------------

    // All 32 lanes must call store; write to shared temp, then have lane‑0
    // copy only the valid rows/cols for boundary tiles.
    __shared__ float C_tile[WMMA_M * WMMA_N];                        // FIX (temp)
    wmma::store_matrix_sync(C_tile, c_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    if (tx == 0) {                                                   // single lane clamps
        for (size_t i = 0; i < WMMA_M; ++i) {
            size_t row = tileRow + i;
            if (row >= numCRows) break;
            #pragma unroll
            for (size_t j = 0; j < WMMA_N; ++j) {
                size_t col = tileCol + j;
                if (col >= numCColumns) break;
                C[row * numCColumns + col] = C_tile[i * WMMA_N + j];
            }
        }
    }
}


__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
     // silence declared but never referenced warning. remove this line when you start working
    // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    size_t total_elements = (Channel * K * K) * (Batch * Height_out * Width_out);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_elements) {

    size_t width_unroll = Batch * Height_out * Width_out;
    size_t h_unroll = tid / width_unroll;
    size_t w_unroll = tid % width_unroll;

    int batch =(int) (w_unroll / (Height_out * Width_out));
    int hw_out =(int) (w_unroll % (Height_out * Width_out));
    int h_out = hw_out / Width_out;
    int w_out = hw_out % Width_out;
    int c = h_unroll / (K * K);
    int pq = h_unroll % (K * K);
    int p = pq / K;
    int q = pq % K;
    int h_in = h_out + p;
    int w_in = w_out + q;
    output[tid] = in_4d(batch, c, h_in, w_in);
    }

    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    cudaMalloc((void**)device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
    cudaMalloc((void**)device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));

    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (size_t)(Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    size_t total_elements = (size_t)Height_unrolled * Width_unrolled;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    matrix_unrolling_kernel<<<blocks, threads_per_block>>>(
        device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    );


    // Matrix multiplication and permutation. Do not modify.
    // Multiply the mask with the unrolled matrix
    dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
                         (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(32, 1, 1);
    matrixMultiplyTF32<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled,
        Height_unrolled, Width_unrolled, Map_out, Width_unrolled
    );

    // Permute the result of matrix multiplication
    const size_t out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
      matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMemcpy(host_output, device_output,
               sizeof(float) * Batch * Map_out * Height_out * Width_out,
               cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
    // TODO: Free device memory

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

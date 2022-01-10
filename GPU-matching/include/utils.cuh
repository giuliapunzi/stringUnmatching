#pragma once
#include <cuda_runtime.h>

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

#ifdef DEBUG
#include <cstdio>
#define CUDA_CHECK( call ) {                        \
    const cudaError_t error = call;                 \
    if (error != cudaSuccess) {                     \
        fprintf(stderr, "Error in %s, line %d: %s", \
                __FILE__, __LINE__,                 \
                cudaGetErrorString(error));         \
        exit(-error);                               \
    }                                               \
}
#else
#define CUDA_CHECK( call ) call;
#endif

namespace strum {
    // Textbook min-reduce on GPU (there is room for improvement)
    template<typename T>
    __global__
    void min_reduce_kernel(T *data, size_t length) {
        auto idx = threadIdx.x + blockIdx.x * blockDim.x;

        auto stride = gridDim.x / 2;
        auto next_idx = idx + stride * blockDim.x;
        auto num_blocks = max(stride, gridDim.x - stride);

        while (stride && blockIdx.x < num_blocks && next_idx < length) {
            data[idx] = min(data[idx], data[next_idx]);

            __syncthreads();

            stride = num_blocks / 2;
            next_idx = idx + stride * blockDim.x;
            num_blocks = max(stride, num_blocks - stride);
        }
    }

    template<typename T>
    __global__
    void mem_init(T *array, size_t length, T value = 0) {
        auto idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < length) {
            array[idx] = value;
        }
    }

    template<typename T>
    T ceil_div(T x, T y) {
        return x/y + (x % y != 0);
    }
}
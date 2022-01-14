#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <algorithm>

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
    // Adapted from CUDA Toolkit Samples
    template<typename T>
    __global__ void min_reduce_kernel(T *data, size_t length, T def_value = (T) -1) {
        __shared__ T sh_data[BLOCK_DIM];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        sh_data[tid] = (i < length) ? data[i] : def_value;

        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sh_data[tid] = min(sh_data[tid], sh_data[tid + s]);

            __syncthreads();
        }

        if (tid == 0)
            data[blockIdx.x] = sh_data[0];
    }

    template<typename T>
    T ceil_div(T x, T y) {
        return x/y + (x % y != 0);
    }

    template<typename T>
    T min_reduce(T *data, size_t length) {
        const auto block_dim = BLOCK_DIM;
        auto grid_dim = ceil_div<size_t>(length, block_dim);
        constexpr T def_value = std::numeric_limits<T>::max();

        while (grid_dim > 1) {
            min_reduce_kernel<<<grid_dim, block_dim>>>(data, length, def_value);
            length = grid_dim;
            grid_dim = ceil_div<size_t>(length, block_dim);
        }

        T result[BLOCK_DIM];
        CUDA_CHECK(cudaMemcpy(result, data, length*sizeof(T), cudaMemcpyDeviceToHost))

        return *std::min_element(result, result + length);
    }
}
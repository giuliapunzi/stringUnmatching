#include "matcher.hpp"
#include "io.hpp"
#include "cuda_helper.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

using namespace strum;


__global__
void expand_kernel(byte_t* matrix, length_t length) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length) {
        auto current = matrix[idx];
        auto next = idx == length - 1? (byte_t) 0 : matrix[idx + 1];

        #pragma unroll
        for (auto i = 1; i < io::Q; ++i) { 
            matrix[idx + i*length] = (byte_t) ((current << 2*i) | (next >> 2*(io::Q - i)));
        }
    }
}

void Matcher::init()  {
    CUDA_CHECK(cudaMalloc((void **) &d_bytes, length*io::Q))
    CUDA_CHECK(cudaMemcpy(d_bytes, reinterpret_cast<const byte_t *>(h_bytes.c_str()),
                          length, cudaMemcpyHostToDevice))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = length/block_dim + !!(length % block_dim);
    expand_kernel<<<grid_dim, block_dim>>>(d_bytes, length);
}

// By Bob Gross
__device__ __forceinline__
byte_t hamming_distance(chunk_t x, chunk_t y) {
    auto diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return (byte_t) (NUM_NUCLEOTIDES - __popcll(diff));
}

__device__ 
void min_reduce_kernel(byte_t* data, length_t length) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    auto stride = gridDim.x/2;
    auto next_idx = idx + stride*blockDim.x;
    auto num_blocks = max(stride, gridDim.x - stride);

    while (stride && blockIdx.x < num_blocks && next_idx < length) {
        data[idx] = min(data[idx], data[next_idx]);
    
        __syncthreads();

        stride = num_blocks/2;
        next_idx = idx + stride*blockDim.x;
        num_blocks = max(stride, num_blocks - stride);
    }    
}

__global__
void min_hamming_distance_kernel(chunk_t sample, const byte_t* bytes, byte_t* result,
                                 length_t length, char excess = 0) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < length) {
        result[idx] = UCHAR_MAX;
    }

    __syncthreads();

    chunk_t chunk = 0;
    byte_t* chunk_bytes = (byte_t*) &chunk;

    for (auto i = 0; i < io::Q; ++i) {
        if (idx*io::Q + excess + i + CHUNK_SIZE <= length) {  // limit*io::Q/io::Q
            #pragma unroll
            for (auto c = 0; c < CHUNK_SIZE; ++c)
                chunk_bytes[c] = bytes[idx + i*length + c];

            auto dist = hamming_distance(sample, chunk);
            result[idx] = min(dist, result[idx]);
        }
    }

    __syncthreads();

    min_reduce_kernel(result, length);
}

byte_t Matcher::min_hamming_distance(chunk_t sample) {
    byte_t* distances;

    CUDA_CHECK(cudaMalloc((void**) &distances, length))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = length/block_dim + !!(length % block_dim);

    min_hamming_distance_kernel<<<grid_dim, block_dim>>>(sample, d_bytes, distances, length, excess);

    byte_t result[BLOCK_DIM];
    auto limit = std::min<length_t>(BLOCK_DIM, length - CHUNK_SIZE + 1);

    CUDA_CHECK(cudaMemcpy(result, distances, limit*sizeof(byte_t), cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaFree(distances))

    return *std::min_element(result, result + limit);
}

Matcher::~Matcher() {
    CUDA_CHECK(cudaFree(d_bytes))
}

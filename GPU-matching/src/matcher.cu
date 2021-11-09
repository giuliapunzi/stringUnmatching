#include "matcher.hpp"
#include "io.hpp"
#include "cuda_helper.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>

#define WARP_SIZE 32

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

__global__ 
void min_reduce_kernel(byte_t* data, length_t length) {
    auto tid = threadIdx.x;
    auto idx = tid + blockIdx.x * blockDim.x;
    volatile byte_t* block = data + blockIdx.x * blockDim.x;
    
    for (auto stride = blockDim.x/2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride && idx + stride < length)
            block[tid] = min(block[tid], block[tid + stride]);
        
        __syncthreads(); 
    }

    // No synchronization needed within warps
    if (tid < WARP_SIZE) {
        #pragma unroll
        for (auto stride = WARP_SIZE; stride > 0; stride >>= 1)
            block[tid] = min(block[tid], block[tid + stride]);
    }

    if (!tid)
        data[blockIdx.x] = block[tid];
}
 
byte_t min_reduce(byte_t* data, length_t length, 
        bool inplace = true, unsigned int max_blocks = 1) {
    auto grid_dim = length/BLOCK_DIM + !!(length % BLOCK_DIM);

    if (!inplace) {
        byte_t* copy;
        CUDA_CHECK(cudaMalloc((void**) &copy, length))
        CUDA_CHECK(cudaMemcpy(copy, data, length, cudaMemcpyDeviceToDevice))
        data = copy;
    }

    while (grid_dim > max_blocks) {
        min_reduce_kernel<<<grid_dim, BLOCK_DIM>>>(data, length);
        CUDA_CHECK(cudaDeviceSynchronize())

        grid_dim = grid_dim/BLOCK_DIM + !!(grid_dim % BLOCK_DIM);
    }

    byte_t result[BLOCK_DIM*max_blocks];
    auto limit = std::min<length_t>(BLOCK_DIM, length);

    CUDA_CHECK(cudaMemcpy(result, data, limit, cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaFree(data))

    return *std::min_element(result, result + limit);
}

__global__
void min_hamming_distance_kernel(chunk_t sample, const byte_t* bytes, byte_t* result,
                                 const length_t length, const byte_t excess) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < length) {
        result[idx] = UCHAR_MAX;
    }

    chunk_t chunk = 0;
    byte_t* chunk_bytes = (byte_t*) &chunk;

    for (auto i = 0; i < io::Q; ++i) {
        // We compare the positions in terms of nucleotides, not bytes
        if (idx*io::Q + excess + i + NUM_NUCLEOTIDES <= length*io::Q) {
            #pragma unroll
            for (auto c = 0; c < CHUNK_SIZE; ++c)
                chunk_bytes[c] = bytes[idx + i*length + c];

            auto dist = hamming_distance(sample, chunk);
            result[idx] = min(dist, result[idx]);
        }
    }
}

byte_t Matcher::min_hamming_distance(chunk_t sample) const {
    byte_t* distances;

    CUDA_CHECK(cudaMalloc((void**) &distances, length))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = length/block_dim + !!(length % block_dim);

    min_hamming_distance_kernel<<<grid_dim, block_dim>>>(sample, d_bytes, 
        distances, length, excess);

    return min_reduce(distances, length, true);
}

Matcher::~Matcher() {
    CUDA_CHECK(cudaFree(d_bytes))
}

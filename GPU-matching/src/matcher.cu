#include "matcher.hpp"
#include "io.hpp"
#include "cuda_helper.cuh"
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_DIM 256

using namespace strum;

constexpr char CHUNK_SIZE = sizeof(chunk_t);
constexpr char NUM_NUCLEOTIDES = CHUNK_SIZE * io::Q;


__global__
void expand_kernel(byte_t* matrix, length_t length) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length) {
        auto current = matrix[idx];
        auto next = idx == length - 1? (byte_t) 0 : matrix[idx + 1];

        for (auto i = 2; i < CHAR_BIT; i += 2) {  // io::Q shifts of 2 bits each
            matrix[idx + i*length] = (byte_t) ((current << i) | (next >> (CHAR_BIT - i)));
        }
    }
}

void strum::copy_and_expand(const byte_t* bytes, byte_t* output, length_t length) {
    CUDA_CHECK(cudaMalloc((void **) &output, length*io::Q))
    CUDA_CHECK(cudaMemcpy(output, bytes, length, cudaMemcpyHostToDevice))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = length/block_dim + !!(length % block_dim);
    expand_kernel<<<grid_dim, block_dim>>>(output, length);
}

__device__ __forceinline__
byte_t hamming_distance(chunk_t x, chunk_t y) {
    auto diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return (byte_t) (NUM_NUCLEOTIDES - __popcll(diff));
}

__global__
void min_hamming_distance_kernel(chunk_t sample, const byte_t* bytes, byte_t* result,
                                 length_t length, char excess = 0) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto limit = length - CHUNK_SIZE + 1 - !!excess;

    for (auto i = 0; i < io::Q; ++i) {
        if (idx*io::Q + i < limit*io::Q - excess) {
            auto chunk = (chunk_t) *(bytes + idx + i*length);
            auto dist = hamming_distance(sample, chunk);

            if (i == 0 || dist < result[idx]) {
                result[idx] = dist;
            }
        }
    }

    __syncthreads();

    auto stride = blockDim.x * gridDim.x;

    while (stride >>= 1) {
        if (idx < stride && idx + stride < limit) {
            result[idx] = min(result[idx], result[idx + stride]);
        }

        __syncthreads();
    }
}

char Matcher::min_hamming_distance(chunk_t sample) {
    byte_t* distances;

    CUDA_CHECK(cudaMalloc((void**) &distances, length))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = length/block_dim + !!(length % block_dim);

    min_hamming_distance_kernel<<<grid_dim, block_dim>>>(sample, d_bytes, distances, length, excess);

    char result = -1;

    CUDA_CHECK(cudaMemcpy((byte_t*) &result, distances, 1, cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaFree(distances))

    return result;
}

Matcher::~Matcher() {
    CUDA_CHECK(cudaFree(d_bytes))
}

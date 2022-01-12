#include "hamming.hpp"
#include "io.hpp"
#include "utils.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <endian.h>

using namespace strum;


/*
 * Copy and shift the whole sequence 4 times.
 * For example, given the sequence GATTACAGATTACA, the final memory will be
 *
 *     byte   0    1    2    3
 *    shift
 *        0   GATT ACAG ATTA CA**
 *        1   ATTA CAGA TTAC A***
 *        2   TTAC AGAT TACA ****
 *        3   TACA GATT ACA* ****
 *
 * The actual memory is 1-dimensional, thus we have
 *
 *     byte   0    1    2    3    4    5    6    7    8    ...
 *    shift   0    0    0    0    1    1    1    1    2    ...
 *            GATT ACAG ATTA CA** ATTA CAGA TTAC A*** TTAC ...
 */
__global__
void expand_kernel(byte_t* matrix, size_t length) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length) {
        byte_t current = matrix[idx];
        byte_t next = idx == length - 1? (byte_t) 0 : matrix[idx + 1];

        #pragma unroll
        for (auto i = 1; i < io::Q; ++i) { 
            matrix[idx + i*length] = (current << 2*i) | (next >> 2*(io::Q - i));
        }
    }
}

void HammingMatcher::init() {
    const byte_t* c_str = reinterpret_cast<const byte_t*>(bytes_.c_str());

    CUDA_CHECK(cudaMalloc((void**) &distances_, length_))
    CUDA_CHECK(cudaMalloc((void**) &d_bytes_, length_*io::Q))
    CUDA_CHECK(cudaMemcpy(d_bytes_, c_str, length_, cudaMemcpyHostToDevice))

    auto block_dim = BLOCK_DIM;
    auto grid_dim = ceil_div<size_t>(length_, block_dim);
    expand_kernel<<<grid_dim, block_dim>>>(d_bytes_, length_);
}

HammingMatcher::~HammingMatcher() {
    CUDA_CHECK(cudaFree(d_bytes_))
    CUDA_CHECK(cudaFree(distances_))
}

// By Bob Gross
__device__ __forceinline__
byte_t hamming_distance(chunk_t x, chunk_t y) {
    chunk_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return static_cast<byte_t>(NUM_NUCLEOTIDES - __popcll(diff));
}

/*
 * Given a (8-byte) template, for every shift in [0, 3] compare every 8 bytes
 * in the shifted-sequence and store the result in `result` if the distance is
 * lower than the on already computed in the previous shift.
 */
__global__
void min_hamming_distance_kernel(chunk_t sample, byte_t* bytes, byte_t* result,
                                 size_t length, byte_t excess = 0) {
    chunk_t chunk = 0;
    auto chunk_bytes = reinterpret_cast<byte_t*>(&chunk);
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = 0; i < io::Q; ++i) {
        if (idx*io::Q + excess + i + CHUNK_SIZE <= length) {  // limit*io::Q/io::Q
            #pragma unroll
            for (auto c = 0; c < CHUNK_SIZE; ++c)
                chunk_bytes[c] = bytes[idx + i*length + c];

            byte_t dist = hamming_distance(sample, chunk);
            result[idx] = min(dist, result[idx]);
        }
    }
}

byte_t HammingMatcher::get_distance(chunk_t sample) {
    sample = htobe64(sample);
    auto block_dim = BLOCK_DIM;
    auto grid_dim = ceil_div<size_t>(length_, block_dim);

    mem_init<byte_t><<<grid_dim, block_dim>>>(distances_, length_, UCHAR_MAX);
    min_hamming_distance_kernel<<<grid_dim, block_dim>>>(sample, d_bytes_, distances_, length_, excess_);
    min_reduce_kernel<<<grid_dim, block_dim>>>(distances_, length_);

    byte_t result[BLOCK_DIM];
    auto limit = std::min<size_t>(BLOCK_DIM, length_ - CHUNK_SIZE + 1);

    CUDA_CHECK(cudaMemcpy(result, distances_, limit, cudaMemcpyDeviceToHost))

    return *std::min_element(result, result + limit);
}

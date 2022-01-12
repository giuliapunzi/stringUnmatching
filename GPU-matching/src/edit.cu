#include "edit.hpp"
#include "io.hpp"
#include "utils.cuh"
#include <algorithm>
#include <climits>
#include <cmath>

#define STRIDE  64
#define OVERLAP 16

using namespace strum;


void EditMatcher::init() {
    num_blocks_ = ceil_div<size_t>(length_, STRIDE);

    auto c_str = (const byte_t*) bytes_.c_str();

    CUDA_CHECK(cudaMalloc((void**) &distances_, num_blocks_))
    CUDA_CHECK(cudaMalloc((void**) &d_bytes_, length_))
    CUDA_CHECK(cudaMemcpy((byte_t*) d_bytes_, c_str, length_, cudaMemcpyHostToDevice))
}

EditMatcher::~EditMatcher() {
    CUDA_CHECK(cudaFree(d_bytes_))
    CUDA_CHECK(cudaFree(distances_))
}

template<typename T>
__device__ __host__ __forceinline__
byte_t get_nucleotide(T x, byte_t pos) {
    return static_cast<byte_t>((x >> (sizeof(T)*CHAR_BIT - 2*pos - 2)) & 0x03);
}

void init_masks(chunk_t sample, mask_t *masks) {
    for (auto i = 0; i < io::Q; ++i)
        masks[i] = 0;

    for (auto i = 0; i < NUM_NUCLEOTIDES; ++i) {
        auto c = get_nucleotide<chunk_t>(sample, i);
        masks[c] |= (mask_t) 1 << i;
    }
}

__device__ __forceinline__
char myers_update(mask_t x, mask_t &v_pos, mask_t &v_neg) {
    mask_t d_0 = (((x & v_pos) + v_pos)^v_pos) | x | v_neg;
    mask_t h_pos = v_neg | ~(d_0 | v_pos);
    mask_t h_neg = v_pos & d_0;

    v_pos = (h_neg << 1) | ~(d_0 | (h_pos << 1));
    v_neg = (h_pos << 1) & d_0;

    return static_cast<char>(!!(h_pos & 0x80000000) - !!(h_neg & 0x80000000));
}

__global__
void min_edit_distance_kernel(const byte_t *bytes, byte_t* result, const mask_t* masks,
                              size_t length, byte_t excess = 0) {
    auto block_idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx = block_idx*STRIDE;

    byte_t min_dist = NUM_NUCLEOTIDES, dist = NUM_NUCLEOTIDES;
    mask_t v_pos = ~((mask_t) 0), v_neg = 0, pattern_mask;

    for (size_t i = 0; (i < STRIDE + OVERLAP) && (idx + i < length); ++i) {
        byte_t b = bytes[idx + i];
        auto limit = idx + i + 1 == length? io::Q - excess : io::Q;

        for (auto pos = 0; pos < limit; ++pos) {
            pattern_mask = masks[get_nucleotide<byte_t>(b, pos)];
            dist = (byte_t) (dist + myers_update(pattern_mask, v_pos, v_neg));
            min_dist = (byte_t) min(dist, min_dist);
        }
    }

    if (idx < length)
        result[block_idx] = min_dist;
}

byte_t EditMatcher::get_distance(chunk_t sample) {
    auto block_dim = BLOCK_DIM;
    auto grid_dim = ceil_div<size_t>(num_blocks_, block_dim);

    mem_init<byte_t><<<grid_dim, block_dim>>>(distances_, num_blocks_, NUM_NUCLEOTIDES);

    mask_t h_masks[io::Q] = {0}, *d_masks;

    init_masks(sample, h_masks);
    CUDA_CHECK(cudaMalloc((void**) &d_masks, io::Q*MASK_SIZE))
    CUDA_CHECK(cudaMemcpy(d_masks, h_masks, io::Q*MASK_SIZE, cudaMemcpyHostToDevice))

    min_edit_distance_kernel<<<grid_dim, block_dim>>>(
        d_bytes_, distances_, d_masks, length_, excess_);
    min_reduce_kernel<byte_t><<<grid_dim, block_dim>>>(distances_, num_blocks_);

    byte_t result[BLOCK_DIM];
    auto limit = std::min<size_t>(BLOCK_DIM, num_blocks_);

    CUDA_CHECK(cudaMemcpy(result, distances_, limit*sizeof(byte_t), cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaFree(d_masks))

    return *std::min_element(result, result + limit);
}

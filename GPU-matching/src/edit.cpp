#include "edit.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace strum;


EditMatcher::EditMatcher(const std::string &bytes, byte_t excess)
        : Matcher(bytes, excess),
          d_bytes_(), distances_(), num_blocks_() {
    init();
}

EditMatcher::EditMatcher(std::string &&bytes, byte_t excess)
        : Matcher(std::move(bytes), excess),
          d_bytes_(), distances_(), num_blocks_() {
    init();
}

EditMatcher::EditMatcher(EditMatcher&& matcher) noexcept
        : Matcher(std::move(matcher.bytes_), matcher.excess_),
          d_bytes_(matcher.d_bytes_),
          distances_(matcher.distances_),
          num_blocks_(matcher.num_blocks_) {
    matcher.d_bytes_ = nullptr;
    matcher.distances_ = nullptr;
}

EditMatcher EditMatcher::from_fasta(const std::string &sequence) {
    std::istringstream iss(sequence);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(iss, oss);
    return EditMatcher(oss.str(), excess);
}

EditMatcher EditMatcher::from_fasta_file(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(ifs, oss);
    return EditMatcher(oss.str(), excess);
}

byte_t EditMatcher::get_distance(const std::string &fasta)  {
    std::istringstream iss(fasta.substr(0, NUM_NUCLEOTIDES));
    std::ostringstream oss;

    io::fasta_to_bytes(iss, oss);
    chunk_t sample = *((const chunk_t *) oss.str().c_str());

    return get_distance(sample);
}

//inline mask_t rshift(mask_t x, int delta = 1) {
//    return (x >> delta) | (~((mask_t) 0) << (NUM_NUCLEOTIDES - delta));
//}
//
//template<typename T>
//inline byte_t get_nucleotide(T x, byte_t pos) {
//    return (x >> (sizeof(T)*CHAR_BIT - 2*pos - 2)) & 0x03;
//}
//
//void init_masks(chunk_t sample, mask_t *masks) {
//    for (auto i = 0; i < io::Q; ++i)
//        masks[i] = 0;
//
//    for (auto i = 0; i < NUM_NUCLEOTIDES; ++i) {
//        auto c = get_nucleotide(sample, i);
//        masks[c] |= (mask_t) 1 << i;
//    }
//}
//
//inline char myers_update(mask_t pattern_mask, mask_t &v_pos, mask_t &v_neg) {
//    mask_t v_mask = pattern_mask | v_neg;
//    mask_t h_mask = (((pattern_mask & v_pos) + v_pos)^v_pos) | pattern_mask;
//    mask_t h_pos = v_neg | ~(h_mask | v_pos);
//    mask_t h_neg = v_pos & h_mask;
//
//    v_pos = (h_neg << 1) | ~(v_mask | (h_pos << 1));
//    v_neg = (h_pos << 1) & v_mask;
//
//    return (char) (!!(h_pos & 0x80000000) - !!(h_neg & 0x80000000));
//}
//
//byte_t EditMatcher::get_distance(chunk_t sample) {
//    mask_t B[4] = {0};
//    init_masks(sample, B);
//    byte_t result = NUM_NUCLEOTIDES;
//
//    #pragma omp parallel firstprivate(B)
//    {
//        const auto num_threads = omp_get_num_threads();
//        const auto block_length = length_/num_threads + !!(length_ % num_threads);
//
//        #pragma omp for reduction(min:result) schedule(static, 1)
//        for (auto i = 0; i < num_threads; ++i) {
//            const auto substr = bytes_view_.substr(i*block_length, block_length + 16);
//
//            byte_t min_dist = NUM_NUCLEOTIDES, dist = NUM_NUCLEOTIDES;
//            mask_t v_pos = ~((mask_t) 0), v_neg = 0;
//
//            for (auto c: substr) {
//                for (auto pos = 0; pos < io::Q; ++pos) {
//                    byte_t nucl = get_nucleotide<byte_t>(c, pos);
//                    dist = (byte_t) (dist + myers_update(B[nucl], v_pos, v_neg));
//                    min_dist = std::min<int>(dist, min_dist);
//                }
//            }
//
//            result = std::min<byte_t>(result, min_dist);
//        }
//    }
//
//    return result;
//}

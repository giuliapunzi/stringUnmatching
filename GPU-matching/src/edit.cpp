#include "edit.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <endian.h>

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

#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN 
    chunk_t sample = *((const chunk_t *) oss.str().c_str());
    return get_distance(sample);
#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN
    chunk_t sample = 0;
    char* sample_bytes = (char*) &sample;

    const char* c_str = oss.str().c_str();

    #pragma unroll
    for (auto i = 0; i < CHUNK_SIZE; ++i)
        sample_bytes[CHUNK_SIZE - i - 1] = c_str[i];

    return get_distance(sample);
#else
#error "Endiannes unknown"
#endif
}

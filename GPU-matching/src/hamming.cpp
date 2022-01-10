#include "hamming.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>

using namespace strum;


HammingMatcher::HammingMatcher(const std::string &bytes, byte_t excess)
        : Matcher(bytes, excess), d_bytes_(), distances_() {
    init();
}

HammingMatcher::HammingMatcher(std::string &&bytes, byte_t excess)
        : Matcher(std::move(bytes), excess), d_bytes_(), distances_() {
    init();
}

HammingMatcher::HammingMatcher(HammingMatcher&& matcher) noexcept
        : Matcher(std::move(matcher.bytes_), matcher.excess_),
          d_bytes_(matcher.d_bytes_), distances_(matcher.distances_) {
    matcher.d_bytes_ = nullptr;
    matcher.distances_ = nullptr;
}

HammingMatcher HammingMatcher::from_fasta(const std::string &sequence) {
    std::istringstream iss(sequence);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(iss, oss);
    return HammingMatcher(std::move(oss.str()), excess);
}

HammingMatcher HammingMatcher::from_fasta_file(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(ifs, oss);
    return HammingMatcher(std::move(oss.str()), excess);
}

byte_t HammingMatcher::get_distance(const std::string &fasta) {
    std::istringstream iss(fasta.substr(0, NUM_NUCLEOTIDES));
    std::ostringstream oss;

    io::fasta_to_bytes(iss, oss);
    chunk_t sample = *((const chunk_t *) oss.str().c_str());

    return get_distance(sample);
}

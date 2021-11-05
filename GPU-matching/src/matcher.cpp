#include "matcher.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>

using namespace strum;


Matcher::Matcher(std::string &&bytes, char excess)
        : h_bytes(std::move(bytes)), d_bytes(), length(h_bytes.length()), excess(excess) {
    init();
}

Matcher::Matcher(const std::string& bytes, char excess)
        : h_bytes(bytes), d_bytes(), length(h_bytes.length()), excess(excess) {
    init();
}

Matcher Matcher::from_fasta(const std::string &sequence) {
    std::istringstream iss(sequence);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(iss, oss);
    return Matcher(oss.str(), excess);
}

Matcher Matcher::from_fasta_file(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    std::ostringstream oss;

    auto excess = io::fasta_to_bytes(ifs, oss);
    return Matcher(oss.str(), excess);
}

byte_t Matcher::min_hamming_distance(const std::string &fasta) {
    std::istringstream iss(fasta.substr(0, NUM_NUCLEOTIDES));
    std::ostringstream oss;

    io::fasta_to_bytes(iss, oss);
    auto bytes = oss.str();

    return min_hamming_distance(*((const chunk_t*) bytes.c_str()));
}

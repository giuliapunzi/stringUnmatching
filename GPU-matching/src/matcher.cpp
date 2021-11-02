#include "matcher.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>

using namespace strum;


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

char Matcher::min_hamming_distance(const std::string &fasta) {
    std::istringstream iss(fasta.substr(0, CHUNK_SIZE));
    std::ostringstream oss;

    io::fasta_to_bytes(iss, oss);

    return min_hamming_distance(static_cast<chunk_t>(*oss.str().c_str()));
}


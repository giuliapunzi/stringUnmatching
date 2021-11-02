#include "matcher.hpp"
#include "io.hpp"
#include <sstream>
#include <fstream>

using namespace strum;


Matcher::Matcher(const std::string& bytes, char excess)
        : h_bytes(bytes), d_bytes(), length(bytes.length()), excess(excess) {
    copy_and_expand(reinterpret_cast<const byte_t *>(h_bytes.c_str()), d_bytes, length);
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

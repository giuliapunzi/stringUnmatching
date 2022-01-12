#include "matcher.hpp"
#include <sstream>
#include <utility>
#include <endian.h>

using namespace strum;


Matcher::Matcher(const std::string& bytes, byte_t excess)
        : bytes_(bytes), length_(bytes_.size()), excess_(excess) {};

Matcher::Matcher(std::string&& bytes, byte_t excess)
        : bytes_(std::move(bytes)), length_(bytes_.size()), excess_(excess) {};

byte_t Matcher::get_distance(const std::string &fasta) {
    std::istringstream iss(fasta.substr(0, NUM_NUCLEOTIDES));
    std::ostringstream oss;

    io::fasta_to_bytes(iss, oss);
    const chunk_t sample = *reinterpret_cast<const chunk_t*>(oss.str().c_str());

    return get_distance(be64toh(sample));
}

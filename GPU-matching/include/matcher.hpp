#pragma once

#include <iostream>
#include <vector>
#include "io.hpp"

namespace strum {

using chunk_t = unsigned long long int;
using length_t = unsigned long long int;

constexpr char CHUNK_SIZE = sizeof(chunk_t);
constexpr char NUM_NUCLEOTIDES = CHUNK_SIZE * io::Q;

class Matcher {
private:
    const std::string h_bytes;
    byte_t* d_bytes;
    const length_t length;
    const byte_t excess;

    void init();

public:
    static Matcher from_fasta(const std::string& sequence);
    static Matcher from_fasta_file(const std::string& filename);

    Matcher(const Matcher&) = delete;
    Matcher(Matcher&&) = default;
    explicit Matcher(std::string&& bytes, byte_t excess = 0);
    explicit Matcher(const std::string& bytes, byte_t excess = 0);

    byte_t min_hamming_distance(chunk_t chunk) const;
    byte_t min_hamming_distance(const std::string& fasta) const;

    ~Matcher();
};
}

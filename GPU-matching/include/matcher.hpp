#pragma once

#include <iostream>
#include <vector>
#include "io.hpp"

namespace strum {

using byte_t = unsigned char;
using chunk_t = unsigned long long int;
using length_t = unsigned long long int;

constexpr char CHUNK_SIZE = sizeof(chunk_t);
constexpr char NUM_NUCLEOTIDES = CHUNK_SIZE * io::Q;

class Matcher {
private:
    std::string h_bytes;
    byte_t* d_bytes;
    length_t length;
    char excess = 0;

public:
    static Matcher from_fasta(const std::string& sequence);
    static Matcher from_fasta_file(const std::string& filename);

    Matcher(const Matcher&) = delete;
    Matcher(Matcher&&) = default;
    explicit Matcher(std::string&& bytes, char excess = 0);

    char min_hamming_distance(chunk_t chunk);
    char min_hamming_distance(const std::string& fasta);

    ~Matcher();
};
}

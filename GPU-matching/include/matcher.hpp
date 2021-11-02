#pragma once
#include <iostream>

namespace strum {

using byte_t = unsigned char;
using chunk_t = unsigned long long int;
using length_t = unsigned long long int;

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
    char min_hamming_distance(chunk_t sample);
    ~Matcher();
};

void copy_and_expand(const byte_t* bytes, byte_t* output, length_t length);

}

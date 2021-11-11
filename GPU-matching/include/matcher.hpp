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
    /**
     * Create a @c Matcher from a FASTA sequence.
     *
     * @param sequence FASTA sequence
     * @return @c Matcher object
     */
    static Matcher from_fasta(const std::string& sequence);

    /**
     * Create a @c Matcher from a FASTA file.
     *
     * @param filename Path to the FASTA file
     * @return @c Matcher object
     */
    static Matcher from_fasta_file(const std::string& filename);

    /**
     * Create a new @c Matcher from a @a binarized FASTA sequence.
     *
     * @param bytes Binary sequence
     * @param excess Number of trailing nucleotides to ignore
     */
    explicit Matcher(std::string&& bytes, byte_t excess = 0);

    Matcher(const Matcher&) = delete;
    Matcher(Matcher&&) = default;
    ~Matcher();

    /**
     * Compute the minimum Hamming distance of the template @p chunk.
     *
     * @param chunk Binary template
     * @return Minimum Hamming distance
     */
    byte_t min_hamming_distance(chunk_t chunk) const;


    /**
     * Compute the minimum Hamming distance of the template @p fasta in FASTA
     * format. @p fasta must be of length 32.
     *
     * @param fasta FASTA template
     * @return Minimum Hamming distance
     */
    byte_t min_hamming_distance(const std::string& fasta) const;
};
}

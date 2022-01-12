#pragma once

#include "io.hpp"


namespace strum {

using chunk_t = uint64_t;

constexpr char CHUNK_SIZE = sizeof(chunk_t);
constexpr char NUM_NUCLEOTIDES = CHUNK_SIZE * io::Q;

class Matcher {
protected:
    std::string bytes_;
    size_t length_;
    byte_t excess_;

public:
    Matcher(const Matcher&) = delete;
    Matcher(Matcher&&) = default;

    /**
     * Create a new @c Matcher from a @a binarized FASTA sequence.
     *
     * @param bytes Binary sequence
     * @param excess Number of trailing nucleotides to ignore
     */
    explicit Matcher(const std::string& bytes, byte_t excess = 0);

    /**
     * Create a new @c Matcher from a @a binarized FASTA sequence.
     *
     * @param bytes Binary sequence
     * @param excess Number of trailing nucleotides to ignore
     */
    explicit Matcher(std::string&& bytes, byte_t excess = 0);

    /**
     * Compute the minimum Hamming distance of the template @p fasta in FASTA
     * format. @p fasta must be of length 32.
     *
     * @param fasta FASTA template
     * @return Minimum Hamming distance
     */
    virtual byte_t get_distance(const std::string& fasta);

    /**
     * Compute the minimum distance of the template @p chunk.
     *
     * @param chunk Binary template
     * @return Minimum Hamming distance
     */
    virtual byte_t get_distance(chunk_t chunk) = 0;
};
}

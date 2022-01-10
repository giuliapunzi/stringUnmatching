#pragma once

#include "io.hpp"
#include "matcher.hpp"

namespace strum {

class HammingMatcher : public Matcher {
private:
    byte_t* d_bytes_;
    byte_t* distances_;

    void init();

public:
    static HammingMatcher from_fasta(const std::string& sequence);
    static HammingMatcher from_fasta_file(const std::string& filename);
    HammingMatcher(HammingMatcher&& matcher) noexcept;
    explicit HammingMatcher(const std::string& bytes, byte_t excess = 0);
    explicit HammingMatcher(std::string&& bytes, byte_t excess = 0);

    byte_t get_distance(const std::string& fasta) override;
    byte_t get_distance(chunk_t chunk) override;

    ~HammingMatcher();
};
}

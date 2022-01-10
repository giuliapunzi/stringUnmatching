#pragma once

#include "io.hpp"
#include "matcher.hpp"

namespace strum {
    using mask_t = uint32_t;
    constexpr char MASK_SIZE = sizeof(mask_t);

    class EditMatcher : public Matcher {
    private:
        byte_t* d_bytes_;
        byte_t* distances_;
        size_t num_blocks_;

        void init();

    public:
        static EditMatcher from_fasta(const std::string& sequence);
        static EditMatcher from_fasta_file(const std::string& filename);
        EditMatcher(EditMatcher&& matcher) noexcept;
        explicit EditMatcher(const std::string& bytes, byte_t excess = 0);
        explicit EditMatcher(std::string&& bytes, byte_t excess = 0);

        byte_t get_distance(const std::string& fasta) override;
        byte_t get_distance(chunk_t chunk) override;

        ~EditMatcher() ;
    };
}

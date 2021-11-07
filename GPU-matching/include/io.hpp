#pragma once

#include <iostream>
#include <climits>

namespace strum {
    using byte_t = unsigned char;

    namespace io {
        constexpr byte_t Q = CHAR_BIT >> 1;

        enum Nucleotide : byte_t {
            A = 0x00,
            C = 0x01,
            G = 0x02,
            T = 0x03
        };

        byte_t fasta_to_bytes(std::istream &input, std::ostream &output, bool drop_last = false);
        void bytes_to_fasta(std::istream &input, std::ostream &output, byte_t excess = 0);
    }
}
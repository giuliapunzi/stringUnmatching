#pragma once

#include <iostream>
#include <climits>

namespace strum {
namespace io {
    constexpr char Q = CHAR_BIT >> 1;

    enum Nucleotide : char {
        A = 0x0,
        C = 0x1,
        G = 0x2,
        T = 0x3
    };

    char fasta_to_bytes(std::istream &input, std::ostream &output, bool drop_last = false);
    void bytes_to_fasta(std::istream &input, std::ostream &output, char excess = 0);
}
}
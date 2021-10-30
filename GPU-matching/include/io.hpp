#include <iostream>

#ifndef STRUM_IO_HPP
#define STRUM_IO_HPP

namespace strum::io {
    constexpr char Q = 4;

    enum Nucleotide : char {
        A = 0x0,
        C = 0x1,
        G = 0x2,
        T = 0x3
    };

    char fasta_to_bytes(std::istream& input, std::ostream& output);
    void bytes_to_fasta(std::istream& input, std::ostream& output, char excess = 0);
}

#endif //STRUM_IO_HPP

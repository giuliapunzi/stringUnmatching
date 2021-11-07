#include "io.hpp"

using namespace strum;


byte_t io::fasta_to_bytes(std::istream& input, std::ostream& output, bool drop_last) {
    byte_t key = 0;
    byte_t key_len = 0;
    char code;

    while(!input.get(code).eof()) {
        switch (std::toupper(code))
        {
            case 'A':
                // key |= io::Nucleotide.A;
                break;
            case 'C':
                key |= io::Nucleotide::C;
                break;
            case 'G':
                key |= io::Nucleotide::G;
                break;
            case 'T':
                key |= io::Nucleotide::T;
                break;
            case ';':   // headers and/or comments
            case '>':
                while(!input.get(code).eof() && code != '\n');
            default:
                continue;
        }

        // here only if the current char is A, C, G, T
        if (++key_len == io::Q){    // we have 4 chars packed in a byte
            output.put(key);
            key = 0;
            key_len = 0;            // for the next iteration
        } else {
            key <<= 2;
        }
    }

    if (!drop_last && key_len > 0) {        // put remaining characters
        auto excess = (byte_t) (io::Q - key_len);
        key <<= 2*excess - 2;               // "left align" key
        output.put(key).flush();

        return excess;
    }

    output.flush();
    return 0;
}

void io::bytes_to_fasta(std::istream &input, std::ostream &output, byte_t excess) {
    char key;  // Notice: shifts on signed chars may have signed results!

    while(!input.get(key).eof()) {
        bool is_last = input.peek() == std::char_traits<char>::eof();
        auto num_shifts = is_last && excess? io::Q - excess : io::Q;

        for (auto shift = 0; shift < num_shifts; ++shift, key <<= 2) {
            switch ((byte_t) key >> (CHAR_BIT - 2)) {  // leftmost two bits
                case io::Nucleotide::A:
                    output.put('A');
                    break;
                case io::Nucleotide::C:
                    output.put('C');
                    break;
                case io::Nucleotide::G:
                    output.put('G');
                    break;
                case io::Nucleotide::T:
                    output.put('T');
                    break;
            }
        }
    }

    output.flush();
}

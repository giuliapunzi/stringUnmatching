#pragma once

#include <iostream>
#include <climits>

namespace strum {
    using byte_t = unsigned char;

    namespace io {
        /// @var Number of nucleotides per byte.
        constexpr byte_t Q = CHAR_BIT >> 1;

        /// @var Nucleotide encodings.
        enum Nucleotide : byte_t {
            A = 0x00,
            C = 0x01,
            G = 0x02,
            T = 0x03
        };

        /**
         * Converts a FASTA stream to compressed nucleotide encodings.
         *
         * Accepts a stream of nucleotide characters (i.e., @c 'A', @c 'C',
         * @c 'G', @c 'T', both in lower and upper cases), and ignores the
         * rest. If a line starts with @c ';' or @c '>', ignores also every
         * other character in that line.
         *
         * @param input Input stream of @c char in FASTA format
         * @param output Output stream of bytes, each containing
         *      @c Q nucleotides
         * @param drop_last If `true`, drop last byte if unfilled
         * @return Number of invalid (rightmost) nucleotides in last byte.
         *      Can be a number from 0 to @c Q - 1
         */
        byte_t fasta_to_bytes(std::istream &input, std::ostream &output, bool drop_last = false);

        /**
         * Converts a byte stream to a FASTA string.
         *
         * @param input Input stream of bytes
         * @param output Output stream of FASTA @c char
         * @param excess Number of invalid (rightmost) nucleotides
         *      to ignore in the last byte.
         */
        void bytes_to_fasta(std::istream &input, std::ostream &output, byte_t excess = 0);
    }
}
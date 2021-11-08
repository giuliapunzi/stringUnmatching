#include "catch2/catch.hpp"
#include "matcher.hpp"

#define GOOD_CHAR 'A'
#define BAD_CHAR 'T'
#define MIN_EXP 2
#define MAX_EXP 8
#define BASE 10
#define HAMMING 16

using namespace strum;


TEST_CASE( "Length Test", "[length]" ) {
    const std::string chunk(NUM_NUCLEOTIDES, GOOD_CHAR);
    auto length = 1;

    for (auto exp = 0; exp < MIN_EXP; ++exp, length *= BASE);
    for (auto exp = MIN_EXP; exp < MAX_EXP; ++exp, length *= BASE) {
        SECTION( "Length:\t10^" + std::to_string(exp) ) {
            std::string sequence(length, BAD_CHAR);

            for (auto c = 0; c < HAMMING; ++c) {
                sequence[length - c - 1] = GOOD_CHAR;
            }
            
            auto matcher = Matcher::from_fasta(sequence);
            unsigned int dist = matcher.min_hamming_distance(chunk);

            REQUIRE( dist == HAMMING );
        }
    }
}

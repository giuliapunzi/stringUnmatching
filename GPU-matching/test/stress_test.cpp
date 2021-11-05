#include "catch2/catch.hpp"
#include "matcher.hpp"

#define GOOD_CHAR 'A'
#define BAD_CHAR 'T'
#define LENGTH 1000

using namespace strum;


TEST_CASE( "Stress Test", "[stress]" ) {
    const std::string chunk(NUM_NUCLEOTIDES, GOOD_CHAR);
    const int length = LENGTH;
    std::string str(length, BAD_CHAR);

    for (auto substr_length = 1; substr_length <= NUM_NUCLEOTIDES; ++substr_length) {
        for (auto c = 0; c < substr_length; ++c) {
            str[c] = GOOD_CHAR;
        }

        auto exp_dist = 32 - substr_length;

        SECTION( "Distance:\t" + std::to_string(exp_dist) ) {
            auto shift = 0;

            for (; shift < length - NUM_NUCLEOTIDES; ++shift) {
                SECTION( "Shift:\t" + std::to_string(shift)) {
                    auto matcher = Matcher::from_fasta(str);
                    auto dist = matcher.min_hamming_distance(chunk);

                    REQUIRE( dist == exp_dist );

                    str[shift] = BAD_CHAR;
                    str[shift + substr_length] = GOOD_CHAR;
                }
            }

            for (; shift < length; ++shift) {
                str[shift] = BAD_CHAR;
            }
        }
    }
}

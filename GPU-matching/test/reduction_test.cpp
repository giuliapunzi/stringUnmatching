#include "catch2/catch.hpp"
#include "hamming.hpp"
#include "edit.hpp"

#define GOOD_CHAR 'A'
#define BAD_CHAR 'T'
#define LENGTH 10000000
#define STRIDE 100000

using namespace strum;


TEST_CASE( "Reduction Test", "[reduction]" ) {
    const std::string chunk(NUM_NUCLEOTIDES, GOOD_CHAR);
    const int length = LENGTH;
    const int stride = STRIDE;
    const int exp_dist = NUM_NUCLEOTIDES - 1;
    std::string str(length, BAD_CHAR);

    SECTION( "Hamming Distance" ) {
        for (int pos = 0; pos < length; pos += stride) {
            str[pos] = GOOD_CHAR;

            SECTION( "Position:\t" + std::to_string(pos) ) {
                auto matcher = HammingMatcher::from_fasta(str);
                int dist = matcher.get_distance(chunk);

                REQUIRE( dist == exp_dist );
            }

            str[pos] = BAD_CHAR;
        }
    }

    SECTION( "Edit Distance" ) {
        for (int pos = 0; pos < length; pos += stride) {
            str[pos] = GOOD_CHAR;

            SECTION( "Position:\t" + std::to_string(pos) ) {
                auto matcher = EditMatcher::from_fasta(str);
                int dist = matcher.get_distance(chunk);

                REQUIRE( dist == exp_dist );
            }

            str[pos] = BAD_CHAR;
        }
    }
}

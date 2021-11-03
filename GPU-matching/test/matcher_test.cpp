#include "catch2/catch.hpp"
#include "matcher.hpp"
#include <algorithm>

using namespace strum;

std::vector<std::string> test_strings = {
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
        "ACGTACGTACGTACGTACGTACGTACGTACGT",
        "GATTACAGATTACAGATTACAGATTACAGATT"
};

TEST_CASE( "Test Matcher", "[matcher]" ) {
    SECTION( "Perfect Match" ) {
        for (auto& str: test_strings) {
            auto matcher = Matcher::from_fasta(str);
            auto dist = matcher.min_hamming_distance(str);

            REQUIRE( dist == 0 );
        }
    }

    SECTION( "No Match" ) {
        for (auto& str: test_strings) {
            auto matcher = Matcher::from_fasta(str);
            std::string modified(str);

            std::transform(str.begin(), str.end(), 
                modified.begin(),
                [](char c) { 
                    switch (c)
                    {
                        case 'A':
                            return 'T';
                        case 'C':
                            return 'G';
                        case 'G':
                            return 'C';
                        default:
                            return 'A';
                    };
                });
            
            auto dist = matcher.min_hamming_distance(modified);

            REQUIRE( dist == 32 );
        }
    }

    SECTION( "Partial Match" ) {
        for (auto& str: test_strings) {
            auto matcher = Matcher::from_fasta(str);
            std::string modified(str);

            std::transform(str.begin(), str.end(), 
                modified.begin(),
                [](char c) { 
                    switch (c)
                    {
                        case 'A':
                            return 'T';
                        case 'T':
                            return 'A';
                        default:
                            return c;
                    };
                });

            auto count = std::count_if(str.begin(), str.end(), [](char c){ return c == 'A' || c == 'T'; });
            auto dist = matcher.min_hamming_distance(modified);

            REQUIRE( dist == count );
        }
    }
}

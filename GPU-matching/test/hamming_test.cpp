#include "catch2/catch.hpp"
#include "hamming.hpp"
#include <algorithm>

using namespace strum;


std::vector<std::string> test_strings = {
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
    "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
    "ACGACGACGACGACGACGACGACGACGACGAC",
    "ACGTACGTACGTACGTACGTACGTACGTACGT",
    "GATTACAGATTACAGATTACAGATTACAGATT"};

TEST_CASE("Test Matcher", "[matcher]")
{
    SECTION("Perfect Match")
    {
        for (auto &str : test_strings)
        {
            auto matcher = HammingMatcher::from_fasta(str);
            auto dist = matcher.get_distance(str);

            REQUIRE(dist == 0);
        }
    }

    SECTION("No Match")
    {
        for (auto &str : test_strings)
        {
            auto matcher = HammingMatcher::from_fasta(str);
            std::string modified(str);

            std::transform(str.begin(), str.end(),
                           modified.begin(),
                           [](char c)
                           {
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

            auto dist = matcher.get_distance(modified);

            REQUIRE(dist == 32);
        }
    }

    SECTION("Partial Match")
    {
        for (auto &str : test_strings)
        {
            auto matcher = HammingMatcher::from_fasta(str);

            SECTION("Count 'A's") {
                int count = std::count(str.begin(), str.end(), 'A');
                auto dist = matcher.get_distance(test_strings[0]);
                
                REQUIRE(dist == 32 - count);
            }

            SECTION("Count A/T substitutions") {
                std::string modified(str);
                std::transform(str.begin(), str.end(),
                            modified.begin(),
                            [](char c)
                            {
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

                auto count = std::count_if(str.begin(), str.end(), [](char c)
                                        { return c == 'A' || c == 'T'; });
                auto dist = matcher.get_distance(modified);

                REQUIRE(dist == count);
            }
        }
    }
}

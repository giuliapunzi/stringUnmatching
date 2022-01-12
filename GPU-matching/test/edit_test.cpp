#include "catch2/catch.hpp"
#include "edit.hpp"
#include <algorithm>

using namespace strum;

std::string text = "ACGTACGTACGTACGTA"
    "ACGTACGTACGTACGTACGTACGTACGTACGT"
    "ACGTACGTACGTACGTACGTACGTACGTACGT"
    "ACGTACGTACGTACGTACGTACGTACGTACGT"
    "ACGTACGTACGTACGTACGTACGTACGTACGT"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT"
    "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT"
    "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT"
    "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT"
    "GATTACAGATTACAGATTACAGATTACAGATT"
    "ACAGATTACAGATTACAGATTACAGATTACAG";
    

TEST_CASE("Test Matcher", "[matcher]")
{
    auto matcher = EditMatcher::from_fasta(text);

    SECTION("Perfect Match")
    {
        std::vector<std::string> test_strings = {
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
            "ACGTACGTACGTACGTACGTACGTACGTACGT",
            "GATTACAGATTACAGATTACAGATTACAGATT"};
        
        for (auto &str : test_strings)
        {
            int dist = matcher.get_distance(str);

            REQUIRE(dist == 0);
        }
    }

    SECTION("Insertions")
    {
        std::vector<std::tuple<std::string, int>> test_strings = {
            {"AAAATCCCCTGGGGTTTTTAAAATCCCCTGGG", 5},
            {"AAAAAACCCCCCGGGGGGTTTTTTAAAACCCC", 8},
            {"GATTACAGGAATTTTAACCAAGATTACAGATT", 7},
        };

        for (auto& [str, exp_dist] : test_strings)
        {
            int dist = matcher.get_distance(str);

            REQUIRE(dist == exp_dist);
        }
    }

    SECTION("Deletions")
    {
        std::vector<std::tuple<std::string, int>> test_strings = {
            {"ACGACGACGACGACGACGACGACGACGACGAC", 10},
            {"GATTAGATTAGATTAGATTAGATTAGATTACA", 10},
            {"TTTTAAACCCGGGTTTAAACCCGGGTTTAAAA", 8},
        };

        for (auto& [str, exp_dist] : test_strings)
        {
            int dist = matcher.get_distance(str);

            REQUIRE(dist == exp_dist);
        }
    }

    SECTION("Substitutions")
    {
        std::vector<std::tuple<std::string, int>> test_strings = {
            {"ACCTACCTACCTACCTACCTACCTACCTACCT", 8},
            {"GATTAGAGATTAGAGATTAGAGATTAGAGATT", 4},
            {"AAATAAATAAATAAATAAATAAATAAATAAAT", 8},
        };

        for (auto& [str, exp_dist] : test_strings)
        {
            int dist = matcher.get_distance(str);

            REQUIRE(dist == exp_dist);
        }
    }
}

#include "catch2/catch.hpp"
#include "io.hpp"
#include <sstream>
#include <unordered_set>

using namespace strum;


TEST_CASE( "Test FASTA -> BYTE -> FASTA conversions", "[convert]" ) {
    SECTION( "Even Strings" ) {
        std::vector<std::string> test_strings = {
                "acgt", "aaaa",
                "AAAACCCCGGGGTTTT",
                "aaaaccccggggtttt",
                "acgtACGTTAGTcgta",
                "atatatatcgcgcgcg",
                "aaaaaaaaaaaaaaaa",
        };

        for (auto& str: test_strings) {
            std::istringstream iss_f2b(str);
            std::ostringstream oss_f2b;

            auto excess = io::fasta_to_bytes(iss_f2b, oss_f2b);

            REQUIRE( excess == 0 );

            std::istringstream iss_b2f(oss_f2b.str());
            std::ostringstream oss_b2f;
            io::bytes_to_fasta(iss_b2f, oss_b2f);

            std::transform(str.begin(), str.end(), str.begin(),
                           [](char c) { return std::toupper(c); });
            REQUIRE( str == oss_b2f.str() );
        }
    }

    SECTION( "Uneven Strings" ) {
        std::vector<std::string> test_strings = {
                "AAAACCCCGGGGTT",
                "a", "CCC", "ATGCG", "aaa",
        };

        for (auto& str: test_strings) {
            std::istringstream iss_f2b(str);
            std::ostringstream oss_f2b;

            auto excess = io::fasta_to_bytes(iss_f2b, oss_f2b);

            std::istringstream iss_b2f(oss_f2b.str());
            std::ostringstream oss_b2f;
            io::bytes_to_fasta(iss_b2f, oss_b2f, excess);

            std::transform(str.begin(), str.end(), str.begin(),
                           [](char c) { return std::toupper(c); });
            REQUIRE( str == oss_b2f.str() );
        }
    }

    SECTION( "Skip Characters" ) {
        std::vector<std::string> test_strings = {
                "abcdefghijklmnopqrstuvwxyz",
                "acgt^&&*((4", "a\nc\tg\t\t\tt\n"
        };

        for (auto& str: test_strings) {
            std::istringstream iss_f2b(str);
            std::ostringstream oss_f2b;

            auto excess = io::fasta_to_bytes(iss_f2b, oss_f2b);

            std::istringstream iss_b2f(oss_f2b.str());
            std::ostringstream oss_b2f;
            io::bytes_to_fasta(iss_b2f, oss_b2f, excess);

            std::vector<char> out_chars;
            std::unordered_set<char> peptides = {'A', 'C', 'G', 'T'};

            for (char c: str) {
                c = (char) std::toupper(c);

                if (peptides.count(c)) {
                    out_chars.push_back(c);
                }
            }

            std::string reduced_str(out_chars.begin(), out_chars.end());
            REQUIRE( reduced_str == oss_b2f.str() );
        }
    }

    SECTION( "Skip Headers" ) {
        std::vector<std::string> test_headers = {
                "; GATTACA abcdefghijklmnopqrstuvwxyz []<<<;;;;;>>>\n",
                "> GATTACA abcdefghijklmnopqrstuvwxyz []<<<;;;;;>>>\n",
                "> GATTACA\n; GATTACA\n> GATTACA\n; GATTACA\n"
        };

        std::vector<std::string> test_strings = {
                "GATTACA", "ACGT",
                "AAAACCCCGGGGTTTT",
                "AAAAAAAAAAAAAAAA",
                "CCCCCCCCCCCCCCCC",
                "GGGGGGGGGGGGGGGG",
                "TTTTTTTTTTTTTTTT",
        };

        for (auto& str: test_strings) {
            for (auto& header: test_headers) {
                std::istringstream iss_f2b(header + str);
                std::ostringstream oss_f2b;

                auto excess = io::fasta_to_bytes(iss_f2b, oss_f2b);

                std::istringstream iss_b2f(oss_f2b.str());
                std::ostringstream oss_b2f;
                io::bytes_to_fasta(iss_b2f, oss_b2f, excess);

                REQUIRE(str == oss_b2f.str());
            }
        }
    }
}

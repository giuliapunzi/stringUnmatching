#include <fstream>
#include <string>
#include "matcher.hpp"
#include "cxxopts/cxxopts.hpp"

using namespace strum;


int main(int argc, char *argv[]){
    cxxopts::Options options("hamming",
        "Find the minimum Hamming distance between a template and any substring in a given sequence");
    options.allow_unrecognised_options();
    options.add_options()
            ("s,sequence", "Sequence file name", cxxopts::value<std::string>())
            ("f,fasta", "Read sequence file in FASTA format")
            ("b,binary", "Binary templates")
            ("h,help", "Print usage");

    options.parse_positional({"sequence"});
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string sequence;

    if (result.count("sequence")) {
        auto filename = result["sequence"].as<std::string>();
        std::ifstream ifs(filename, std::ios::binary);

        if (!ifs) {
            std::cerr << "Cannot open file " << filename << std::endl;
            exit(-1);
        }

        sequence.assign(std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>());
    } else {
        std::cerr << "Missing sequence file" << std::endl;
        exit(-1);
    }

    Matcher matcher = result.count("fasta")?
            Matcher::from_fasta(sequence) : Matcher(std::move(sequence));

    bool binary = result.count("binary");
    auto temp_str = result.unmatched();

    auto print_dist = [&](const std::string& temp) {
        auto dist = binary? matcher.min_hamming_distance(static_cast<chunk_t>(*temp.c_str()))
                          : matcher.min_hamming_distance(temp);

        std::cout << static_cast<unsigned>(dist) << std::endl;
    };

    if (temp_str.empty()) {
        for (std::string line; !std::getline(std::cin, line).eof(); ) {
            print_dist(line);
        }
    } else {
        for (auto& temp: temp_str) {
            print_dist(temp);
        }
    }

    return 0;
}

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
            ("t,tsv", "For every template print its FASTA representation and"
                      " the minimum Hamming distance separated by a tab.")
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

    const Matcher& matcher = result.count("fasta")?
            Matcher::from_fasta(sequence) : Matcher(std::move(sequence));

    auto args = result.unmatched();
    bool binary = result.count("binary") && args.empty();
    bool tsv = result.count("tsv");

    if (binary) {
        for (chunk_t chunk; !std::cin.read((char*) &chunk, CHUNK_SIZE).eof(); ) {
            unsigned int dist = matcher.min_hamming_distance(chunk);

            if (tsv) {
                std::istringstream iss(std::string((char*) &chunk, CHUNK_SIZE));
                io::bytes_to_fasta(iss, std::cout);
                std::cout << '\t';
            }

            std::cout << dist << std::endl;
        }

        return 0;
    } 

    auto log = [tsv, &matcher](const std::string& line) {
        unsigned int dist = matcher.min_hamming_distance(line);

        if (tsv)
            std::cout << line << '\t';

        std::cout << dist << std::endl;
    };

    if (args.empty()) {
        for (std::string line; !std::getline(std::cin, line).eof(); ) 
            log(line);
    } else {
        for (auto& arg: args) 
            log(arg);
    }

    return 0;
}

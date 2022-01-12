#include <fstream>
#include <string>
#include "hamming.hpp"
#include "edit.hpp"
#include "cxxopts/cxxopts.hpp"
#include "endian.h"

using namespace strum;


int main(int argc, char *argv[]){
    cxxopts::Options options("hamming",
                             "Find the minimum distance between a template and any substring in a given sequence");
    options.allow_unrecognised_options();
    options.add_options()
            ("s,sequence", "Sequence file name", cxxopts::value<std::string>())
            ("f,fasta", "Read sequence file in FASTA format")
            ("b,binary", "Binary templates")
            ("t,tsv", "For every template print its FASTA representation and"
                      " the minimum distance separated by a tab.")
            ("e,edit", "Find the minimum edit distance instead of the Hamming distance")
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

    auto use_edit = result.count("edit");
    std::unique_ptr<Matcher> matcher;

    if (result.count("fasta")) {
        if (use_edit)
            matcher = std::make_unique<EditMatcher>(EditMatcher::from_fasta(sequence));
        else
            matcher = std::make_unique<HammingMatcher>(HammingMatcher::from_fasta(sequence));
    } else {
        if (use_edit)
            matcher = std::make_unique<EditMatcher>(std::move(sequence));
        else
            matcher = std::make_unique<HammingMatcher>(std::move(sequence));
    }

    auto args = result.unmatched();
    bool binary = result.count("binary") && args.empty();
    bool tsv = result.count("tsv");

    if (binary) {
        chunk_t chunk;
        char (&chunk_bytes)[CHUNK_SIZE] = *reinterpret_cast<char(*)[CHUNK_SIZE]>(&chunk);

        while (!std::cin.read(chunk_bytes, CHUNK_SIZE).eof()) {
            auto dist = matcher->get_distance(be64toh(chunk));

            if (tsv) {
                std::istringstream iss(std::string(chunk_bytes, CHUNK_SIZE));
                io::bytes_to_fasta(iss, std::cout);
                std::cout << '\t';
            }

            std::cout << (unsigned short) dist << std::endl;
        }

        return 0;
    }

    auto log = [tsv, &matcher](const std::string& line) {
        unsigned int dist = matcher->get_distance(line);

        if (tsv)
            std::cout << line << '\t';

        std::cout << (unsigned short) dist << std::endl;
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
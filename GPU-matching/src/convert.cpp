#include <fstream>
#include "io.hpp"
#include "cxxopts.hpp"

using namespace strum;


int main(int argc, char *argv[]){
    cxxopts::Options options("convert", "Packs nucleotides in groups of four, encoded in one byte");
    options.allow_unrecognised_options();
    options.add_options()
            ("f,file", "File name", cxxopts::value<std::string>())
            ("i,inverse", "Convert bytes to FASTA")
            ("e,excess", "Drop last <E> nucleotides", cxxopts::value<int>()->default_value("0"))
            ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto excess = (char) (result["excess"].as<int>());

    std::function<char(std::istream&, std::ostream&)>
            inverse_map = [excess](std::istream &input, std::ostream &output) -> char {
        io::bytes_to_fasta(input, output, excess);
        return 0;
    };

    auto stream_map = result.count("inverse") ? inverse_map : &io::fasta_to_bytes;

    if (result.count("file")) {
        auto filename = result["file"].as<std::string>();
        std::ifstream input(filename, std::ios::binary);

        return stream_map(input, std::cout);
    }

    if (result.unmatched().empty()) {
        return stream_map(std::cin, std::cout);
    }

    if (result.unmatched().size() > 1) {
        for (const std::string& arg: result.unmatched()) {
            std::istringstream iss(arg);
            excess = stream_map(iss, std::cout);
            std::cout << "\t" << excess << std::endl;
        }

        return 0;
    }

    std::istringstream iss(result.unmatched().back());
    return stream_map(iss, std::cout);
}

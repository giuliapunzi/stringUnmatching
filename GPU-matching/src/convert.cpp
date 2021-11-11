#include <fstream>
#include "io.hpp"
#include "cxxopts/cxxopts.hpp"

using namespace strum;


int main(int argc, char *argv[]){
    cxxopts::Options options("convert", "Packs nucleotides in groups of four, encoded in one byte");
    options.allow_unrecognised_options();
    options.add_options()
            ("f,file", "File name", cxxopts::value<std::string>())
            ("i,inverse", "Convert bytes to FASTA")
            ("d,drop_last", "Drop last un-filled byte")
            ("e,excess", "Drop last <E> nucleotides", cxxopts::value<int>()->default_value("0"))
            ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto excess = (char) (result["excess"].as<int>());
    bool drop_last = result["drop_last"].count();

    std::function<byte_t(std::istream&, std::ostream&)>
        forward_map = [drop_last](std::istream &input, std::ostream &output) -> byte_t {
            return io::fasta_to_bytes(input, output, drop_last);
        },
        inverse_map = [excess](std::istream &input, std::ostream &output) -> byte_t {
            io::bytes_to_fasta(input, output, excess);
            return 0;
        };

    auto stream_map = result.count("inverse") ? inverse_map : forward_map;

    if (result.count("file")) {
        auto filename = result["file"].as<std::string>();
        std::ifstream input(filename, std::ios::binary);

        return stream_map(input, std::cout);
    }

    if (result.unmatched().empty()) {
        return stream_map(std::cin, std::cout);
    }

    std::ostringstream oss;
    std::copy(result.unmatched().begin(), result.unmatched().end(),
              std::ostream_iterator<std::string>(oss));

    std::istringstream iss(oss.str());
    return stream_map(iss, std::cout);
}

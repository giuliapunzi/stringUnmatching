// RG-July2021: compile with option -std=c++20

#include <iostream>
#include <fstream>
#include "mapfile.hpp"

using namespace std;

/* Q-GRAMS */

constexpr auto Q = 4;    // it fits a byte
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 

void find_Q_grams(const char * filename, const char * outfilename){
    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file(filename, textlen);      

    ofstream fout;
    fout.open(outfilename, ios::binary | ios::out);

    for(uint64_t i = 0; i < textlen; i++){ // auto i is a bug here
        auto key = text[i];
        for (auto j =0; j < Q; j++){
            switch ((key >> (Q-1+Q-1)) & 0x3)
            {
            case 0:
                fout << "A";
                break;
            case 1:
                fout << "C";
                break;
            case 2:
                fout << "G";
                break;
            case 3:
                fout << "T";
                break;
            default:
                ;
            }
            key <<= 2;            // shift two bits to the left
        }
    }
    fout.close();
    unmap_file(text, textlen);
}



int main(int argc, char *argv[]){
    if (argc != 3){
        cout << "Usage: " + string(argv[0]) + " IN_packed_atcg_filename OUT_text_filename " << endl << flush;
        exit(255);
    }

    find_Q_grams(argv[1], argv[2]);

    return 0;
}
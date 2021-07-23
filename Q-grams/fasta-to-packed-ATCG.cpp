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

    //uint8_t key = 0;  // 4 chars from { A, C, G, T } packed as a 8-bit unsigned integer
    char key = 0;  // 4 chars from { A, C, G, T } packed as a 8-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;

    while(i < textlen){
        switch (toupper(text[i]))
        {
        case 'A':
            // key |= 0x0;
            break;
        case 'C':
            key |= 0x1;
            break;
        case 'G':
            key |= 0x2;
            break;
        case 'T':
            key |= 0x3;
            break;
        case '\n':
            skip = true;
            break;
        case '>':
        case ';':
            while( i < textlen && text[i] != '\n') i++;
            // key = 0;
            // key_len = 0;
            skip = true;
            break;
        default:
            // key = 0;
            // key_len = 0;
            skip = true;
            break;
        }
        i++;
        if (skip){ 
            skip = false; 
            continue;
        }
        // here only if the current char is A, C, G, T
        if (++key_len == Q){  // we have 4 chars packed in a byte
            fout.write(&key, 1);
            key_len = 0;        // for the next iteration
        }
        key <<= 2;            // shift two bits to the left
    }

    fout.close();
    unmap_file(text, textlen);
}



int main(int argc, char *argv[]){
    if (argc != 3){
        cout << "Usage: " + string(argv[0]) + " IN_fasta_filename OUT_packed_atcg_filename" << endl << flush;
        exit(255);
    }

    find_Q_grams(argv[1], argv[2]);

    return 0;
}
// RG-July2021: compile with option -std=c++20

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>

// for mmap:
#include "mapfile.hpp"

using namespace std;


/* Q-GRAMS */

constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
constexpr auto N_CLASSES = 6545;   // = (35 choose 3) for Q = 32 and Sigma = 4
uint16_t global_counter = 0; // < N_CLASSES
uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t
unordered_map<uint32_t, uint16_t> class_to_ID;
vector <uint64_t> Parikh_class[N_CLASSES];

void print_Q_gram(uint64_t gram){
    char s[Q+1];
    s[Q] = '\0';
    for (auto i=Q-1; i >= 0; i--, gram >>= 2){
        switch (gram & 0x3)
        {
        case 0:
            s[i] = 'A';
            break;
        case 1:
            s[i] = 'C';
            break;
        case 2:
            s[i] = 'G';
            break;
        case 3:
            s[i] = 'T';
            break;
        default:
            break;
        }
    }
    cout << string(s) << endl << flush;
}

void check_Q_gram(uint64_t gram){
    uint32_t item =  *reinterpret_cast<uint32_t *>(char_counter);
    if (!class_to_ID.contains(item) ){
        class_to_ID[item] = global_counter++;
    }  
    Parikh_class[class_to_ID[item]].push_back(gram);
}


void find_Q_grams(const char * filename){
    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file(filename, textlen);      

    for (auto i =0; i < 4; i++) char_counter[i] = 0;
    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;

    while(i < textlen){
        switch (toupper(text[i]))
        {
        case 'A':
            // key |= 0x0;
            char_counter[0]++;
            break;
        case 'C':
            key |= 0x1;
            char_counter[1]++;
            break;
        case 'G':
            key |= 0x2;
            char_counter[2]++;
            break;
        case 'T':
            key |= 0x3;
            char_counter[3]++;
            break;
        case '\n':
            skip = true;
            break;
        case '>':
        case ';':
            while( i < textlen && text[i] != '\n') i++;
            key = 0;
            key_len = 0;
            for (auto i =0; i < 4; i++) char_counter[i] = 0;
            skip = true;
            break;
        default:
            key = 0;
            key_len = 0;
            for (auto i =0; i < 4; i++) char_counter[i] = 0;
            skip = true;
            break;
        }
        i++;
        if (skip){ 
            skip = false; 
            continue;
        }
        // here only if the current char is A, C, G, T
        if (++key_len == Q){
            key_len = Q-1;        // for the next iteration
            check_Q_gram( key );  // & mask when Q < 32
            char_counter[(key >> (Q-1+Q-1)) & 0x3]--;  // exiting char
        }
        key <<= 2;            // shift two bits to the left
    }
    unmap_file(text, textlen);
}

void clean_dump_classes(){
    for( auto i: class_to_ID ){
        auto v = Parikh_class[ i.second ];
        sort( v.begin(), v.end() );
        v.erase( unique( v.begin(), v.end() ), v.end() );
        // TBD: dump to a file
        if (v.size() > 0 ){
            ofstream fout;
            string outfilename = "file_Parikh_" + to_string(i.first); // we need to recover the Parish vector to build the outfilename
            fout.open(outfilename, ios::binary | ios::out);
            for (auto& e : v)
                fout.write(reinterpret_cast<char *>(&e), sizeof(uint64_t)); 
            fout.close();
        }
        // total += v.size();
        // cout << v.size() << endl;
        // for (const auto& e : v)
        //     print_Q_gram(e); 
        // cout << endl;
    }
    //cout << "total " << total << endl;
}

int main(int argc, char *argv[]){
    if (argc != 2){
        cout << "Usage: " + string(argv[0]) + " <FASTA_filename>" << endl << flush;
        exit(255);
    }

    find_Q_grams(argv[1]);

    clean_dump_classes();

    // for (const auto& gram: Q_hash) {
    //     print_Q_gram(gram);
    // }

    // for (auto e : Parikh_classes) {
    //     uint32_t temp = e.first;
    //     uint8_t * tempa = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&temp));
    //     for (auto i = 0; i < 4; i++)
    //         cout << static_cast<uint64_t>(tempa[i]) << ' ';
    //     cout << "\t : " << e.second << "\n";
    // }

    return 0;
}
// RG-July2021: compile with option -mbmi2 -std=c++2a

#include <iostream>
#include <fstream>
#include <cassert>
#include <bitset>
#include <immintrin.h>

// for mmap:
#include "mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "class_partitions_6545.h"

using namespace std;

/* Q-GRAMS */

constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t

constexpr auto MASK_WEIGHT = 14;  // number of 1s

constexpr auto UNIVERSE_SIZE = 268435456;    // 4^14 = 268435456

bitset<UNIVERSE_SIZE> universe_bitvector;  // 33 554 432 bytes

__attribute__((always_inline)) uint64_t qgram_to_index(uint64_t gram, uint64_t mask){
    return _pext_u64(gram & mask, mask);
}

__attribute__((always_inline)) uint64_t index_to_qgram(uint64_t index, uint64_t mask){
    return _pdep_u64(index & mask, mask);
}

void process_mask(uint64_t mask, const char * outfilename){
    assert( __popcount(mask) == 14 );

    // initialize bit vector
    universe_bitvector.reset();

    // scan all files and sets the bits corresponding to each masked q-gram
    for (auto i = 0; i < N_CLASSES; i++){
        uint32_t suffix = Parikh_class_partition[i];
        // open filename as an array of uint64_t
        string name = "file_Parikh_" + to_string(suffix);
        size_t gramlen = 0;   
        const char * temp = map_file(name.c_str(), gramlen);      
        const uint64_t * gram = reinterpret_cast<const uint64_t *>(temp);      
        for(size_t i = 0; i < gramlen/8; i++){
            universe_bitvector[ qgram_to_index(gram[i], mask)] = true;
        }
        unmap_file(temp, gramlen);
    }

    // scan the bit vector and populate the complement 
    ofstream fout;
    fout.open(outfilename, ios::binary | ios::out);
    for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
        if (!(universe_bitvector[i])){
            uint64_t t = index_to_qgram( i, mask);
            fout.write(reinterpret_cast<char *>(&t), sizeof(uint64_t)); 
        }
    }
    fout.close();
}


int main(int argc, char *argv[]){
    if (argc != 2){
        cout << "Usage: " + string(argv[0]) + " complement_filename" << endl << flush;
        exit(255);
    }

    uint64_t mask = 0x3FFF;

    process_mask( mask, argv[1]);
    
    return 0;
}
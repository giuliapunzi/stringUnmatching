// RG-July2021: compile with option -mbmi2 -std=c++2a

#include <iostream>
#include <fstream>
#include <cassert>
#include <bitset>
#include <immintrin.h>

// for mmap:
#include "mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
//#include "class_partitions_6545.h"

using namespace std;

/* Q-GRAMS */

constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t

constexpr auto MASK_WEIGHT = 14;  // number of 1s

constexpr auto UNIVERSE_SIZE = 268435456;    // 4^14 = 268435456

bitset<UNIVERSE_SIZE> universe_bitvector;  // 33 554 432 bytes

//__attribute__((always_inline)) 
inline uint64_t qgram_to_index(uint64_t gram, uint64_t mask){
    return _pext_u64(gram & mask, mask);
}

//__attribute__((always_inline)) 
uint64_t index_to_qgram(uint64_t index, uint64_t mask){
    return _pdep_u64(index, mask);
}

string qgram_to_string(uint64_t gram){
  string s = "";
  for (auto i=Q-1; i >= 0; i--, gram >>= 2){
        switch (gram & 0x3)
        {
        case 0:
            s = "A" + s;
            break;
        case 1:
            s = "C" + s;
            break;
        case 2:
            s = "G" + s;
            break;
        case 3:
            s = "T" + s;
            break;
        default:
            break;
        }
    }
  return s;
}


string qgram_to_string_mask(uint64_t gram, uint64_t mask){
  string s = "";
  for (auto i=Q-1; i >= 0; i--, gram >>= 2, mask >>= 2){
    if (mask & 0x3){ 
        switch (gram & 0x3)
        {
        case 0:
            s = "A" + s;
            break;
        case 1:
            s = "C" + s;
            break;
        case 2:
            s = "G" + s;
            break;
        case 3:
            s = "T" + s;
            break;
        default:
            break;
        }
    }
  }
  return s;
}

string qgram_to_string_bug(uint64_t gram){
  string s = "";
  for (auto i=Q-1; i >= 0; i--, gram <<= 2){
        switch (gram & 0xC000000000000000)
        {
        case 0:
            s += "A";
            break;
        case 1:
            s += "C";
            break;
        case 2:
            s += "G";
            break;
        case 3:
            s += "T";
            break;
        default:
            break;
        }
    }
  return s;
}


string qgram_to_string_mask_bug(uint64_t gram, uint64_t mask){
  string s = "";
  for (auto i=Q-1; i >= 0; i--, gram <<= 2, mask <<= 2){
    if (mask & 0xC000000000000000){ 
        switch (gram & 0xC000000000000000)
        {
        case 0:
            s += "A";
            break;
        case 1:
            s += "C";
            break;
        case 2:
            s += "G";
            break;
        case 3:
            s += "T";
            break;
        default:
            break;
        }
    }
  }
  return s;
}

void process_mask(const char * filename, uint64_t mask, const char * outfilename){
    assert( __builtin_popcountll(mask) == MASK_WEIGHT );

    // initialize bit vector
    universe_bitvector.reset();

    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file(filename, textlen);      

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
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
    	    uint64_t index = qgram_to_index(key, mask);
            //cout << qgram_to_string(key) << " " << qgram_to_string_mask(key, mask) << " " << std::hex << key << " " <<  std::hex << (key & mask) << " " <<  std::hex << index << std::dec << endl;
            universe_bitvector[ index ] = true;
        }
        key <<= 2;            // shift two bits to the left
    }
    unmap_file(text, textlen);

    cout << "found " << universe_bitvector.count() << " qgrams" << endl << flush;

    // scan the bit vector and populate the complement 
    ofstream fout;
    fout.open(outfilename, ios::binary | ios::out);
    uint64_t counter = 0;
    for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
        if (!(universe_bitvector[i])){
            counter++;
            uint64_t t = index_to_qgram( i, mask);
            fout.write(reinterpret_cast<char *>(&t), sizeof(uint64_t)); 
        }
    }
    fout.close();
    cout << "found " << counter << " complement qgrams out of " << UNIVERSE_SIZE << endl << flush;
    assert(UNIVERSE_SIZE == universe_bitvector.count() + counter);
}


int main(int argc, char *argv[]){
    if (argc != 3){
        cout << "Usage: " + string(argv[0]) + " input_text_filename complement_filename" << endl << flush;
        exit(255);
    }

    uint64_t mask = 0x3FFF;

    process_mask( argv[1], mask, argv[2]);
    
    return 0;
}

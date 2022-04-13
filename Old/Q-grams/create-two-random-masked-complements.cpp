// RG-July2021: compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <fstream>
#include <cassert>
#include <bitset>
#include <immintrin.h>

// for mmap:
#include "mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

/* Q-GRAMS */

//constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t

constexpr auto MASK_WEIGHT = 28;  // number of 1s, twice the number of selected chars (as the alphabet is 4)

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

void process_mask(uint64_t mask, const string outfilename){
    assert( __builtin_popcountll(mask) == MASK_WEIGHT );

    // initialize bit vector
    universe_bitvector.reset();

    // scan all files and sets the bits corresponding to each masked q-gram
    for (auto i = 0; i < N_CLASSES; i++){
        uint32_t suffix = Parikh_class_partition[i];
        string name = "../data/FILES/file_Parikh_" + to_string(suffix);
        // open name as a sequence of uint64_t
        ifstream fin;
        fin.open(name, ios::binary | ios::in);
        uint64_t gram;
        while (true){
            fin.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
            if (!fin) break;
            universe_bitvector[ qgram_to_index(gram, mask)] = true;
        }
        fin.close();
        cout << "*" << flush;
    }
    cout << endl << endl << flush;

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
    // if (argc != 2){
    //     cout << "Usage: " + string(argv[0]) + " complement_filename" << endl << flush;
    //     exit(255);
    // }

    srand(time(NULL));
    // Store all the items in a list, shuffle that list, then choose the first 14+14 items from that list.
    int positions[32]; 
    for(int i=0; i < 32; i++)
        positions[i] = i;
    
    // Set i to the last item's index
    int i = 32-1;
    while (i > 0){
        // Choose an item ranging from the first item up to the item given in i. 
        // Note that the item at i+1 is excluded.
        int j = rand() % (i + 1);
         // Swap item at index i with item at index j;
         // in this case, i and j may be the same
        int tmp = positions[i];
        positions[i] = positions[j];
        positions[j] = tmp;
         // Move i so it points to the previous item
        i = i - 1;
    }

    // first mask are the first 14 positions
    int firstmaskpos[14];
    for(int i=0; i< 14; i++)
        firstmaskpos[i] = 2*positions[i];

    sort(firstmaskpos, firstmaskpos+14);
    cout << "First mask pos are ";
    for(int i=0; i< 14; i++)
        cout << firstmaskpos[i] << ", ";
    cout << endl;

    uint64_t g1 = 0b11;
    for(int i=1; i< 14; i++){
        // cout << "g1 is " << bitset<64>(g1) << endl << flush;
        g1 <<= (firstmaskpos[i] - firstmaskpos[i-1]);
        g1 |= 0b11;
    }
        
    g1 <<= (64 - 2 - firstmaskpos[13]);
    cout << "g1 is " << bitset<64>(g1) << endl << flush;

    // second mask are the next 14 positions
    int secondmaskpos[14];
    for(int i=0; i< 14; i++)
        secondmaskpos[i] = 2*positions[i+14];

    sort(secondmaskpos, secondmaskpos+14);
    cout << "Second mask pos are ";
    for(int i=0; i< 14; i++)
        cout << secondmaskpos[i] << ", ";
    cout << endl;

    uint64_t g2 = 0b11;
    for(int i=1; i< 14; i++){
        // cout << "g2 is " << bitset<64>(g2) << endl << flush;
        g2 <<= (secondmaskpos[i] - secondmaskpos[i-1]);
        g2 |= 0b11;
    }
        
    g2 <<= (64 - 2 - secondmaskpos[13]);
    cout << "g2 is " << bitset<64>(g2) << endl << flush;


    if(g2 & g1){
        cout << "Functions are not disjoint!"<< endl;
        exit(255);
    }

    // TODO: LINK THE TWO FILES 
    process_mask( g1, "../data/complementaries/complementary" + to_string(g1)); //argv[1]);
    cout << "Finished processing first mask" << endl << flush;

    process_mask( g2, "../data/complementaries/complementary" + to_string(g2));
    cout << "Finished processing second mask" << endl << flush;
    
    return 0;
}

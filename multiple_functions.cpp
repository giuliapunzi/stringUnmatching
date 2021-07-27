// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <stdlib.h>
#include <immintrin.h> // to use pdep, pext

using namespace std;

constexpr auto Q = 32;
constexpr int N_tests = 3; // number of tests
constexpr int N_completions = 256; // number of completions for each template

constexpr int N_hash_fctns = 3;  // number of hash functions 
constexpr int target_size = 14; // target space size of hash functions

// g is the array of hash functions, of size N_hash_fctns
// csets is an array of arrays of complementary sets, of size N_hash_fctns * 
// csets_sizes is an array of ints, with the sizes of the corresponding complementary sets
void sort_according_to_masks(const uint64_t* g, uint64_t** csets, uint64_t* csets_sizes)
{
    // for each function, find the overlap with the previous and sort its corresponding array
    for(int i = 1; i< N_hash_fctns; i++)
    {
        uint64_t currentfctn = g[i];
        uint64_t overlapmask = currentfctn;
        uint64_t truemask = 0;

        for(int j = 0; j<i ; j++)
            overlapmask &= g[j];
        
        // when overlapmask, currentfctn are differ, it is because the former is 0 and the latter is 1
        // performing xor finds the truemask of currentfctn positions which are not in overlap
        truemask = currentfctn^overlapmask;

        // sort the corresponding array according to the masks
        sort(csets[i], csets[i]+csets_sizes[i], [=](uint64_t x, uint64_t y) {
            if((x & overlapmask) > (y&overlapmask)) // order according to the overlapping positions first
		        return false;

	        if((x & overlapmask) < (y&overlapmask)) // order according to the overlapping positions first
		        return true;
	
	        return (x & truemask) < (y&truemask); // if overlapping positions are equal, order according to the rest
        });
    }
}


int main()
{
    return 0;
}
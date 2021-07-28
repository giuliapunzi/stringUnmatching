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
constexpr int N_hash_fctns = 4;  // number of hash functions 
constexpr int target_size = 14; // target space size of hash functions



// given an uint64_t, print it in A,C,G,T alphabet
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




// g is the array of hash functions, of size N_hash_fctns
// csets is an array of arrays of complementary sets, of size N_hash_fctns * 
// csets_sizes is an array of ints, with the sizes of the corresponding complementary sets
void sort_according_to_masks(const uint64_t* g, uint64_t** csets, uint64_t* csets_sizes) // DEBUGGED
{
    // for each function, find the overlap with the previous and sort its corresponding array
    for(int i = 1; i< N_hash_fctns; i++)
    {
        // cout << "Considering function " << i+1 << " given by " << bitset<64>(g[i]) << endl << flush;

        uint64_t currentfctn = g[i];
        uint64_t overlapmask = 0;
        uint64_t truemask = 0;

        for(int j = 0; j<i ; j++)
            overlapmask |= (g[i] & g[j]);

        
        // when overlapmask, currentfctn are differ, it is because the former is 0 and the latter is 1
        // performing xor finds the truemask of currentfctn positions which are not in overlap
        truemask = currentfctn^overlapmask;

        // cout << "Overlapmask is " << bitset<64>(overlapmask) << endl << flush;
        // cout << "True mask is " << bitset<64>(truemask) << endl << flush;

        // sort the corresponding array according to the masks
        sort(csets[i], csets[i]+csets_sizes[i], [&](uint64_t x, uint64_t y) {
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
    uint64_t g[N_hash_fctns];
    g[0] = 0b0011001111000000001100001100110000000000110000000011000011110000;
    g[1] = 0b1111001100110011000011000000001100001100000000001111000000000000;
    g[2] = 0b0000110000001111001100110000000011000000110011000000000011001100;
    g[3] = 0b1100110000000000110000110000110000000000001100001100001100110011;

    
    uint64_t set0[] = {0b0001000010000000000100000100000000000000110000000001000001110000, 0b0011001011000000000000001000010000000000010000000011000000100000, 0b0010001101000000000100000000110000000000100000000001000011100000};
    uint64_t set1[] = {0b1111001000110000000001000000000100001100000000001000000000000000, 0b0101001000110000000011000000001000001000000000000011000000000000, 0b1011000000010011000000000000000100001000000000001001000000000000, 0b1100000100000010000010000000000000001100000000000101000000000000};
    uint64_t set2[] = {0b0000100000000011000100100000000000000000010001000000000010001100, 0b0000100000001001000100100000000000000000100000000000000011001000, 0b0000110000000100001000010000000001000000110001000000000000000000};
    uint64_t set3[] = {0b0000100000000000100000010000000000000000000100000100001100000001, 0b1100010000000000000000000000100000000000000100000100001000110010};

    uint64_t* csets[N_hash_fctns] = {set0, set1, set2, set3};


    uint64_t csets_sizes[N_hash_fctns];
    csets_sizes[0] = 3;
    csets_sizes[1] = 4;
    csets_sizes[2] = 3;
    csets_sizes[3] = 2;

    cout << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
    cout << endl;


    cout << "Starting complementary sets: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
    {
        cout << "Set " << i+1 << ": ";
        for(int j = 0; j< csets_sizes[i] ; j++)
        {
            cout << bitset<64>(csets[i][j]) << "=";
            print_Q_gram(csets[i][j]);
        }
        cout << endl << flush;
    }
        
    cout << endl;


    sort_according_to_masks(g, csets, csets_sizes);

    cout << "Complementary sets after sorting: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
    {
        cout << "Set " << i+1 << ": ";
        for(int j = 0; j< csets_sizes[i] ; j++)
        {
            cout << bitset<64>(csets[i][j]) << "=";
            print_Q_gram(csets[i][j]);
        }
        cout << endl << flush;
    }
        
    cout << endl;


    return 0;
}
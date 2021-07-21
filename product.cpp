// compile with option -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form
#include <stdlib.h>

using namespace std;

constexpr auto Q = 32;
constexpr int N_missing = 4; // Q - 2m
constexpr int N_tests = 2;

// array of 256 completions we will need to check
uint64_t completions[256];

uint8_t text[] = {"ACAGCTTTTTGCGTATCTGGGCGCTATGCATGCTTAGGCTATCGGGCGCGCGCGATTATGCGCGCTA"};


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


__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}


void check(uint8_t * text, uint64_t textlen, int* mindist) // min dist will be filled with minimum distances
{
    for(int i=0; i<256; i++)
        mindist[i]=Q+1;

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer

    for(uint64_t i = 0; i< Q-1; i++)
    {
        switch (text[i])
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
        }

        key<<=2;
    }

    uint64_t i;
    for( ;i < textlen; i++){
        switch (text[i])
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
        }

        for(int j=0; j<256;j++)
        {
            int dist = Hamming_distance(key, completions[j]);
            if(dist < mindist[j])
                mindist[j] = dist;
        }

        key <<= 2;
    }
}


// INPUT: template given as uint64, array of 4 positions that need to be filled (shift from the right necessary to have them as last two)
// gtemplate needs to be zero in the missing pairs of positions.
// OUTPUT: completion to uint64 in missing positions
void complete (uint64_t gtemplate, int* missing) // DEBUGGED
{
    // since the free positions of g are 4, we need numbers from 00000000 to 11111111 (0 to 255)
    // we will perform all these completions
    for(uint64_t curr = 0; curr < 256; curr++) 
    {
        // cout << endl << "Completing with " << curr << ", which is " << bitset<64>(curr) << endl << flush;
        uint64_t compmask = 0;
        uint64_t tcurr = curr;

        for(int i = 1; i<= N_missing; i++, tcurr >>= 2) // possible optimization: unroll the loop
        {
            compmask = compmask | (tcurr & 0b11);
            compmask <<= missing[i]-missing[i-1];
        }

        // cout << "Mask is " << bitset<64>(compmask) << endl << flush;
        completions[curr] = gtemplate | compmask;

        // cout << "Current completion is " << bitset<64>(completions[curr]) << endl << flush;
    }

    
    cout << "Completions array is: {";

    for(int i = 0; i<256; i++)
    {
        cout << bitset<64>(completions[i]) << ", \t";
    }
    cout << "}"<< endl << flush;
}


// INPUT: two non-overlapping hash functions g1,g2 (bit masks of 64 bit (represented as a uint64 each), 
// with 11 at the pair of positions the function projects at) and two arrays of 
// uint64 c1, c2 of sizes n1,n2 representing the complementary sets of the codomains of the hash functions
// OUTPUT: the product set of c1,c2
void sample_product_set(const uint64_t g1, uint64_t * c1,  int n1, const uint64_t g2,  uint64_t * c2, int n2) // DEBUGGED
{
    uint64_t cg = ~(g1 | g2);  // cg is the complementary of positions of hash functions


    int missing[N_missing+1];
    int pos = Q-1;
    int index = 0; 

    cout << endl << "Complementary of g is " << bitset<64>(cg) << endl << flush;

    // compute array of missing positions (how much shift to the left for next missing position) 0011000011001100 len 16, missing = [4,10,14]
    for(int i = 0; i< 64; i+=2, cg <<= 2)
    {
        if(cg & 0xC000000000000000)
            missing[index++] = i;  
    }

    missing[N_missing] = 64-2; // fixed

    cout << "Missing positions are: ";
    for(int i=0; i<=N_missing; i++)
        cout << "\t" << missing[i];
    cout << endl << flush;

    for(int test=0; test< N_tests; test++)
    {
        int i = rand() % n1;
        int j = rand() % n2;

        cout << "Indices chosen: i=" << i<< "; j="<< j << endl <<flush;

        // c1[i] and c2[j] are two uint64_t representing two M-grams we wish to compute the product of
        // first, set the two elements to zero outside g1, g2 (apply the mask with bitwise AND)
        // bitwise OR is now sufficient to produce an element in the direct product            
        uint64_t gtemplate = (c1[i] & g1) | (c2[j] & g2);

        cout << "c1[i] = " << bitset<64>(c1[i]) << endl << flush;
        cout << "c2[j] = " << bitset<64>(c2[j]) << endl << flush;

        cout << "Template is " << bitset<64>(gtemplate) << endl << flush;
        // complete(gtemplate, missing);

        // check()
    }
}



int main()
{
    uint64_t g1 = 0xF00F0F0F00F0F0F0; // alternating 14 pairs
    uint64_t g2 = 0x0F00F0F0FF000F0F; // 14 pairs in complementary of g1

    cout << "g1 is " << bitset<64>(g1) << "; g2 is " << bitset<64>(g2) << endl << flush;

    // initialize complementaries and their sizes
    int n1 = 3;
    int n2 = 2;

    // uint64_t c1[n1] = {0b0001001001110010011010011011, 0b0011011000110100011001100010, 0b1100110011001101100111000100}; // ACAGCTAGCGGCGT, ATCGATCACGCGAG, TATATATCGCTACA
    uint64_t c1[n1] = {0b0001000000000010000001110000001000000000011000001001000010110000, 0b0011000000000110000000110000010000000000011000000110000000100000, 0b1100000000001100000011000000110100000000100100001100000001000000}; // ACAGCTAGCGGCGT, ATCGATCACGCGAG, TATATATCGCTACA

    // note: since c2 corresponds to the positions from 29-56, need to add 28 zeroes at the end
    // uint64_t c2[n2] = {0b11000001101111011000001101110000000000000000000000000000, 0b10100111101001110010011111010000000000000000000000000000}; // TAACGTTCGAATCT, GGCTGGCTAGCTTC
    uint64_t c2[n2] = {0b0000110000000000000100001011000011011000000000000000001100000111, 0b0000101000000000011100001010000001110010000000000000011100001101}; // TAACGTTCGAATCT, GGCTGGCTAGCTTC 

    // print qgrams in complementaries
    cout << "c1: ";
    for(int i =0; i<n1; i++)
    {
        cout << bitset<64>(c1[i]) << "=";
        print_Q_gram(c1[i]);
    }
        
    cout << "c2: ";
    for(int i =0; i<n2; i++)
    {
        cout << bitset<64>(c2[i]) << "=";
        print_Q_gram(c2[i]);
    }

    // int dist = Hamming_distance(c1[0], c1[1]);
    // cout << "Distance between c1[0], c1[1] is " << dist << endl << flush;
    // cout << "Distance between c1[0], c1[2] is " << Hamming_distance(c1[0], c1[2]) << endl << flush;
    // cout << "Distance between c1[1], c1[2] is " << Hamming_distance(c1[1], c1[2]) << endl << flush;
    // cout << "Distance between c2[0], c2[1] is " << Hamming_distance(c2[0], c2[1]) << endl << flush;

    sample_product_set(g1, c1, n1, g2, c2, n2);

    return 0;
}
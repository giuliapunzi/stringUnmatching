#include <iostream>
#include <bit> 
#include <stdlib.h>

using namespace std;

constexpr auto Q = 32;
constexpr int N_missing = 4; // Q - 2m
constexpr int N_tests = 100;

// array of 256 completions we will need to check
uint64_t completions[256];


__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y)
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return popcount(diff);
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
void complete (uint64_t gtemplate, int* missing) 
{
    // since the free positions of g are 4, we need numbers from 00000000 to 11111111 (0 to 255)
    // we will perform all these completions
    for(uint64_t curr = 0; curr < 256; curr++) 
    {
        uint64_t compmask = 0;
        uint64_t tcurr = curr;

        for(int i = 1; i<= N_missing; i++, tcurr >>= 2) // possible optimization: unroll the loop
        {
            compmask = compmask | (tcurr & 0b11);
            compmask <<= missing[i]-missing[i-1];
        }

        completions[curr] = gtemplate | compmask;
    }
}


// INPUT: two non-overlapping hash functions g1,g2 (bit masks of 64 bit (represented as a uint64 each), 
// with 11 at the pair of positions the function projects at) and two arrays of 
// uint64 c1, c2 of sizes n1,n2 representing the complementary sets of the codomains of the hash functions
// OUTPUT: the product set of c1,c2
void sample_product_set(const uint64_t g1, uint64_t * c1,  int n1, const uint64_t g2,  uint64_t * c2, int n2)
{
    uint64_t cg = ~(g1 | g2);  // cg is the complementary of positions of hash functions

    int missing[N_missing+1];
    int pos = Q-1;
    int index = 0; 

    // compute array of missing positions (how much shift to the left for next missing position) 0011000011001100 len 16, missing = [4,10,14]
    for(int i = 0; i< 64; i+=2, cg <<= 2)
    {
        if(cg & 0xC000000000000000)
            missing[index++] = i;  
    }

    missing[N_missing] = 64;

    for(int test=0; test< N_tests; test++)
    {
        int i = rand() % n1;
        int j = rand() % n2;

        // c1[i] and c2[j] are two uint64_t representing two M-grams we wish to compute the product of
        // first, set the two elements to zero outside g1, g2 (apply the mask with bitwise AND)
        // bitwise OR is now sufficient to produce an element in the direct product            
        uint64_t gtemplate = (c1[i] & g1) | (c2[j] & g2);
        complete(gtemplate, missing);

        // check()
    }
}


int main()
{
    return 0;
}
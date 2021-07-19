#include <iostream>

using namespace std;

constexpr auto Q = 32;


// INPUT: template given as uint64, with mask marking the positions that have been set
// OUTPUT: completion to uint64 in unmarked positions
void complete (uint64_t template, uint64_t mask)
{
    template = template & g;

    // perform all completions; find the positions not belonging to g and fill in all possible ways
    // we need to perform an OR with a random mask, which is zero over g
    uint64_t cg = ~g;

    // since the free positions of g are 4, we need numbers from 0000 to 1111 (0 to 15)
    // we will perform all these completions
    for(int curr = 0; curr < 15; curr++) // Q - size of g = 4
    {
        // int last = curr & 0b11;
        // int first = curr & 0b1100;
    }
}


// INPUT: two non-overlapping hash functions g1,g2 (bit masks of 64 bit (represented as a uint64 each), 
// with 11 at the pair of positions the function projects at) and two arrays of 
// uint64 c1, c2 of sizes n1,n2 representing the complementary sets of the codomains of the hash functions
// OUTPUT: the product set of c1,c2
void product_set(const uint64_t g1, const uint64_t g2, uint64_t * c1, uint64_t * c2, int n1, int n2)
{
    uint64_t g = g1 | g2;  // g is the union of positions of hash functions

    for(int i=0; i< n1; i++)
    {
        for(int j=0; j<n2; j++)
        {
            // c1[i] and c2[j] are two uint64_t representing two M-grams we wish to compute the product of
            // first, set the two elements to zero outside g1, g2 (apply the mask with bitwise AND)
            // bitwise OR is now sufficient to produce an element in the direct product            
            uint64_t template = (c1[i] & g1) | (c2[j] & g2);
            complete(template, g);
        }
    }

    return;
}


int main()
{
    return 0;
}
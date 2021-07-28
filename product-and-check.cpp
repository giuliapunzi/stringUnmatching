// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <fstream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <stdlib.h>
#include <immintrin.h> // to use pdep, pext

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "./Q-grams/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int N_missing = 4; // Q - 2m
constexpr int N_tests = 3; // number of products to compute
constexpr int N_completions = 256; // number of completions of each product

// array of N_completions (between 1 and 256) completions we will check
uint64_t completions[N_completions]; 
uint64_t mindist[N_completions];



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


// given two uint64_t, compute their Hamming distance
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}


// text is now a file composed of uint64_t

 // the check will taking one byte of the text at a time, where the byte contains 4 chars
 // it scans every Qgram of the text, comparing it to every template and keeping an array 
 // mindist of minimum distances
 // filename is the name of the file containing the Qgrams
// void check (uint8_t * text, uint64_t textlen, int* mindist)  // DEBUGGED
void check ()
{
    // re-initialize distances for every template
    for(int templindex = 0; templindex < N_completions; templindex++)
        mindist[templindex] = Q+1; //Hamming_distance(completions[templindex], key); 


    // scan all files and sets the bits corresponding to each masked q-gram
    for (auto i = 0; i < N_CLASSES; i++){
        uint32_t suffix = Parikh_class_partition[i];
        string name = "file_Parikh_" + to_string(suffix);
        // open name as a sequence of uint64_t
        ifstream fin;
        fin.open(name, ios::binary | ios::in);
        uint64_t gram;
        while (true){
            fin.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
            if (!fin) break;
            // compute distance for each Qgram of the file
            for(int j=0; j<N_completions;j++){
                int dist = Hamming_distance(gram, completions[j]);
                if(dist < mindist[j])
                    mindist[j] = dist;
            }
        }
        fin.close();
    }

    return;
}


// INPUT: two non-overlapping hash functions g1,g2 (bit masks of 64 bit (represented as a uint64 each), 
// with 11 at the pair of positions the function projects at) and two filenames for the complementary sets
// OUTPUT: the product set of c1,c2
void sample_product_set(const char* textname, const uint64_t g1, const char * filename1, const uint64_t g2, const char * filename2) //uint64_t * c1,  int n1, const uint64_t g2,  uint64_t * c2, int n2) // DEBUGGED
{
    uint64_t cg = ~(g1 | g2);  // cg is the complementary of positions of hash functions
    int pos = Q-1;
    int index = 0; 
    cout << endl << "Complementary of g is " << bitset<64>(cg) << endl << flush;
    srand (time(NULL));

    ifstream complementary1, complementary2;
    complementary1.open(filename1, ios::binary | ios::in);
    complementary2.open(filename2, ios::binary | ios::in);
    for(int test=0; test< N_tests; test++) // for now, go in order
    {
        uint64_t gram1, gram2;
        complementary1.read(reinterpret_cast<char *>(&gram1), sizeof(uint64_t)); 
        complementary2.read(reinterpret_cast<char *>(&gram2), sizeof(uint64_t));
        
        if (!complementary1 || !complementary2) break;
         
        uint64_t gtemplate = (gram1 & g1) | (gram2 & g2);

        // fill completions array
        for(uint64_t curr = 0; curr < N_completions; curr++) 
            completions[curr] = gtemplate | _pdep_u64(curr, cg);

        check();

        // cout << "Printing min distances for the " << N_completions << " completions: " << flush;
        // for(int dd = 0; dd< N_completions; dd++)
        //     cout << mindist[dd] << " " << flush;

        // cout << endl << flush;
    }

    complementary1.close();
    complementary2.close();
}


int main()
{
    // randomize g1,g2
    srand(time(NULL));


    // sample_product_set(g1, c1, n1, g2, c2, n2);

    return 0;
}
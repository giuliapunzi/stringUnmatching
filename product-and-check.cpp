// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <stdlib.h>
#include <immintrin.h> // to use pdep, pext

using namespace std;

constexpr auto Q = 32;
constexpr int N_missing = 4; // Q - 2m
constexpr int N_tests = 3; // number of products to compute
constexpr int N_completions = 256; // number of completions of each product

// array of N_completions (between 1 and 256) completions we will check
uint64_t completions[N_completions];

// text used for debugging
uint8_t text[] = {0b00010010, 0b01111111, 0b11111001, 0b10110011, 0b01111010, 0b10011001, 0b11001110, 0b01001110, 0b01111100, 0b10100111, 0b00110110, 0b10100110, 0b01100110, 0b01100011, 0b11001110, 0b01100110, 0b01110010};
constexpr int textlen = 17;

void simple_check(int * mindist); // used for debugging


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



 // the check will taking one byte of the text at a time, where the byte contains 4 chars
 // it scans every Qgram of the text, comparing it to every template and keeping an array 
 // mindist of minimum distances
void check (uint8_t * text, uint64_t textlen, int* mindist)  // DEBUGGED
{
    for(int i=0; i<N_completions; i++) // mindist has the same size as completions
        mindist[i]=Q+1;

    uint64_t key = 0;  

    // setup first Q characters by hand
    // for Q=32, first 8 elements of text array (8*8=64)
    key |= text[0];
    key = (key<<8) | text[1];
    key = (key<<8) | text[2];
    key = (key<<8) | text[3];
    key = (key<<8) | text[4];
    key = (key<<8) | text[5];
    key = (key<<8) | text[6];
    key = (key<<8) | text[7];

    // cout << "First key is " << bitset<64>(key) << endl << flush;


    // initialize distances for every template
    for(int templindex = 0; templindex < N_completions; templindex++)
        mindist[templindex] = Hamming_distance(completions[templindex], key); 


    // for every element of text (every uint8_t), 3 shift and three Qgrams
    for(uint64_t i = 8;i < textlen; i++)
    {
        // cout << "Iteration no. " << i << " for text integer " << bitset<8>(text[i]) << endl << flush; 
        key <<= 2;
        key |= ((text[i] >> 6) & 0b11); // after being shifted by two, key gets an OR with the last two bits of the current uint8 shifted by six       

        // cout << "key is " << bitset<64>(key) << endl << flush;
        for(int j=0; j<N_completions;j++)
        {
            int dist = Hamming_distance(key, completions[j]);
            if(dist < mindist[j])
                mindist[j] = dist;
        }

        key <<= 2;
        key |= ((text[i] >> 4) & 0b11); // after being shifted by two, key gets an OR with the last two bits of the current uint8 shifted by four       

        // cout << "key is " << bitset<64>(key) << endl << flush;

        for(int j=0; j<N_completions;j++)
        {
            int dist = Hamming_distance(key, completions[j]);
            if(dist < mindist[j])
                mindist[j] = dist;
        }

        key <<= 2;
        key |= ((text[i] >> 2) & 0b11); // after being shifted by two, key gets an OR with the last two bits of the current uint8 shifted by two       

        // cout << "key is " << bitset<64>(key) << endl << flush;

        for(int j=0; j<N_completions;j++)
        {
            int dist = Hamming_distance(key, completions[j]);
            if(dist < mindist[j])
                mindist[j] = dist;
        }

        key <<= 2;
        key |= (text[i] & 0b11); // after being shifted by two, key gets an OR with the last two bits of the current uint8        

        // cout << "key is " << bitset<64>(key) << endl << flush;

        for(int j=0; j<N_completions;j++)
        {
            int dist = Hamming_distance(key, completions[j]);
            if(dist < mindist[j])
                mindist[j] = dist;
        }
    }

    // // now, for every element of text (every uint8_t), 3 shift and three Qgrams
    // for(uint64_t i = 4;i < textlen; i++)
    // {  
    //     for(int j=0; j<256;j++)
    //     {
    //         int dist = Hamming_distance(key, completions[j]);
    //         if(dist < mindist[j])
    //             mindist[j] = dist;
    //     }

    //     key <<= 2;
    // }

    return;
}


// INPUT: template given as uint64, array of 4 positions that need to be filled (shift from the right necessary to have them as last two)
// gtemplate needs to be zero in the missing pairs of positions.
// OUTPUT: completion to uint64 in missing positions
void complete (uint64_t gtemplate, int* missing, uint64_t cmask) // DEBUGGED WITH PDEP
{
    // since the free positions of g are 4, we need numbers from 00000000 to 11111111 (0 to 255)
    // we will perform all these completions
    for(uint64_t curr = 0; curr < N_completions; curr++) 
    {
        // cout << endl << "Completing with " << curr << ", which is " << bitset<64>(curr) << endl << flush;
        // uint64_t compmask = 0;
        // uint64_t tcurr = curr;

        // for(int i = 1; i<= N_missing; i++, tcurr >>= 2) // possible optimization: unroll the loop
        // {
        //     compmask = compmask | (tcurr & 0b11);
        //     compmask <<= missing[i]-missing[i-1];
        // }

        // cout << "Mask for first completion is " << bitset<64>(compmask) << endl << flush;
        // completions[curr] = gtemplate | compmask;
        // cout << "Current completion is " << bitset<64>(completions[curr]) << endl << flush;

        // cout << "Depositing bits of curr according to cmask yields: " << bitset<64>(_pdep_u64(curr, cmask)) << endl << flush;
        // cout << "Completion with pdep is " << bitset<64>(gtemplate | _pdep_u64(curr, cmask)) << endl << flush;
        // cout << "Are the two completions equal? " << (completions[curr] == (gtemplate | _pdep_u64(curr, cmask))) << endl << endl << flush;
        completions[curr] = gtemplate | _pdep_u64(curr, cmask);
    }

    
    // cout << "Completions array is: {";

    // for(int i = 0; i<N_completions; i++)
    // {
    //     cout << bitset<64>(completions[i]) << ", \t";
    // }
    // cout << "}"<< endl << flush;

    cout << endl;
}


// INPUT: two non-overlapping hash functions g1,g2 (bit masks of 64 bit (represented as a uint64 each), 
// with 11 at the pair of positions the function projects at) and two arrays of 
// uint64 c1, c2 of sizes n1,n2 representing the complementary sets of the codomains of the hash functions
// OUTPUT: the product set of c1,c2
void sample_product_set(const uint64_t g1, uint64_t * c1,  int n1, const uint64_t g2,  uint64_t * c2, int n2) // DEBUGGED
{
    uint64_t cg = ~(g1 | g2);  // cg is the complementary of positions of hash functions


    int missing[N_missing+1]; // don't need missing array; we can use pdep
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
    cout << endl << endl << flush;

    srand (time(NULL));

    for(int test=0; test< N_tests; test++)
    {
        int i = rand() % n1;
        int j = rand() % n2;

        cout << "Indices chosen: i=" << i<< "; j="<< j << endl <<flush;

        // c1[i] and c2[j] are two uint64_t representing two M-grams we wish to compute the product of
        // first, set the two elements to zero outside g1, g2 (apply the mask with bitwise AND)
        // bitwise OR is now sufficient to produce an element in the direct product            
        uint64_t gtemplate = (c1[i] & g1) | (c2[j] & g2);

        // cout << "c1[i] = " << bitset<64>(c1[i]) << endl << flush;
        // cout << "c2[j] = " << bitset<64>(c2[j]) << endl << flush;

        cout << "Template is " << bitset<64>(gtemplate) << endl << flush;

        complete(gtemplate, missing, ~(g1 | g2));

        int mindist[N_completions];
        check(text, textlen, mindist);

        // cout << "Printing min distances for the " << N_completions << " completions: " << flush;
        // for(int dd = 0; dd< N_completions; dd++)
        //     cout << mindist[dd] << " " << flush;

        // cout << endl << flush;

        simple_check(mindist);
        cout << endl << flush;
    }
}


int char_dist(string x, string y) // used for debugging
{
    if(x.size() != y.size())
        return -1;

    int dist = 0;
    for(int i = 0; i < x.size(); i++)
    {
        if(x[i] != y[i])
            dist++;
    }
    return dist;
}

void simple_check(int * mindist) // used for debugging
{    
    // setup input (both completions and text)
    string string_compl[N_completions];
    for(int i= 0; i<N_completions; i++)
    {
        uint64_t gram = completions[i];

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

        string_compl[i] = string(s);
    }

    // cout << "Completions are: ";
    // for(int i = 0; i<N_completions ; i++)
    //     cout << string_compl[i] << ", " << flush;
    // cout << endl;


    char string_text[] = {"ACAGCTTTTTGCGTATCTGGGCGCTATGCATGCTTAGGCTATCGGGCGCGCGCGATTATGCGCGCTAG"};
    int string_textlen = 68;

    int string_mindist[N_completions];
    for(int i = 0; i< N_completions; i++)
        string_mindist[i] = Q+1;

    // cout << "Before starting, min distances: ";
    // for(int i = 0; i < N_completions; i++)
    //     cout << string_mindist[i] << " ";
    // cout << endl << flush;

    for(int i = 0; i < string_textlen-Q+1; i++)
    {
        string current_Qgram = "";

        // get the current Qgram
        for(int j = 0; j<Q; j++)
            current_Qgram = current_Qgram + string_text[i+j];

        // cout << "Current Qgram: " << current_Qgram << endl << flush;

        // test the distance of current Qgram to others
        for(int j = 0; j < N_completions; j++)
        {
            int distance = char_dist(string_compl[j], current_Qgram);
            if(distance < string_mindist[j])
                string_mindist[j] = distance;
        }
    }

    // cout << "Min distances when seen as strings are ";
    // for(int i = 0; i < N_completions; i++)
    //     cout << string_mindist[i] << " ";
    // cout << endl << flush;

    bool same = true;
    for(int i = 0; i<N_completions; i++)
    {
        if(string_mindist[i] != mindist[i])
            same = false;
    }

    if(same)
        cout << "OK: The two vectors are equal" << endl << flush;
    else 
        cout << "KO: The two vectors differ!" << endl << flush;

    return;
}


int main()
{
    // randomize g1,g2
    srand(time(NULL));

    // uint64_t g1 = 0xF00F0F0F00F0F0F0; // alternating 14 pairs
    // uint64_t g2 = 0x0F00F0F0FF000F0F; // 14 pairs in complementary of g1

    uint64_t g1 = 0b0011000011000011000011110011001100110000111100001111000011110000;
    uint64_t g2 = 0b1100001100110000111100000000110011001111000011000000111100001111;

    cout << "g1 is " << bitset<64>(g1) << "; g2 is " << bitset<64>(g2) << endl << flush;

    // initialize complementaries and their sizes
    int n1 = 6; 
    int n2 = 4;

    // uint64_t c1[n1] = {0b0001001001110010011010011011, 0b0011011000110100011001100010, 0b1100110011001101100111000100}; // ACAGCTAGCGGCGT, ATCGATCACGCGAG, TATATATCGCTACA
    uint64_t c1[n1] = {0b0001000000000010000001110000001000000000011000001001000010110000, 0b0011000000000110000000110000010000000000011000000110000000100000, 0b1100000000001100000011000000110100000000100100001100000001000000, 0b0101001110000100010001010011001000111000011001001001000010110100, 0b0011000100010110001100100001110000011100011001000110001100100100, 0b1100001001001101000111000110110101000010100100001100111001000111}; // ACAGCTAGCGGCGT, ATCGATCACGCGAG, TATATATCGCTACA

    // note: since c2 corresponds to the positions from 29-56, need to add 28 zeroes at the end
    // uint64_t c2[n2] = {0b11000001101111011000001101110000000000000000000000000000, 0b10100111101001110010011111010000000000000000000000000000}; // TAACGTTCGAATCT, GGCTGGCTAGCTTC
    uint64_t c2[n2] = {0b0000110000000000000100001011000011011000000000000000001100000111, 0b0000101000000000011100001010000001110010000000000000011100001101, 0b1000110001110001000100001011010011011001001001000111001100000111, 0b0000101000110010010100111010010101110010000110000100011100101101}; // TAACGTTCGAATCT, GGCTGGCTAGCTTC 

    // print qgrams in complementaries
    // cout << "c1: ";
    // for(int i =0; i<n1; i++)
    // {
    //     cout << bitset<64>(c1[i]) << "=";
    //     print_Q_gram(c1[i]);
    // }
        
    // cout << "c2: ";
    // for(int i =0; i<n2; i++)
    // {
    //     cout << bitset<64>(c2[i]) << "=";
    //     print_Q_gram(c2[i]);
    // }

    // int dist = Hamming_distance(c1[0], c1[1]);
    // cout << "Distance between c1[0], c1[1] is " << dist << endl << flush;
    // cout << "Distance between c1[0], c1[2] is " << Hamming_distance(c1[0], c1[2]) << endl << flush;
    // cout << "Distance between c1[1], c1[2] is " << Hamming_distance(c1[1], c1[2]) << endl << flush;
    // cout << "Distance between c2[0], c2[1] is " << Hamming_distance(c2[0], c2[1]) << endl << flush;

    sample_product_set(g1, c1, n1, g2, c2, n2);

    // simple_check();
    return 0;
}
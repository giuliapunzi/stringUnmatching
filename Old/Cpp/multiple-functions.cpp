// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <immintrin.h> // to use pdep, pext

// for mmap:
#include "../script/mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "../script/class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int N_tests = 3; // number of tests
constexpr int N_completions = 256; // number of completions for each template
constexpr int N_hash_fctns = 1;  // number of hash functions 
constexpr int target_size = 14; // target space size of hash functions
// constexpr int MAX_complement_size = 150000;
vector<uint64_t> csets_array[N_hash_fctns]; // array of vectors for complementary sets
// uint64_t csets_sizes[N_hash_fctns];

uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t

constexpr auto MASK_WEIGHT = 28;  // number of 1s, twice the number of selected chars (as the alphabet is 4)

constexpr auto UNIVERSE_SIZE = 268435456;    // 4^14 = 268435456

bitset<UNIVERSE_SIZE> universe_bitvector;  // 33 554 432 bytes



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


//__attribute__((always_inline)) 
inline uint64_t qgram_to_index(uint64_t gram, uint64_t mask){
    return _pext_u64(gram & mask, mask);
}

//__attribute__((always_inline)) 
uint64_t index_to_qgram(uint64_t index, uint64_t mask){
    return _pdep_u64(index, mask);
}

void process_multiple_masks(uint64_t* mask_array, string filename){
    for(int maskindex = 0; maskindex < N_hash_fctns; maskindex++){
        uint64_t mask = mask_array[maskindex]; // current mask

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

        // we know that the complement will have size UNIVERSE_SIZE - count()
        // uint64_t comp_size = UNIVERSE_SIZE - universe_bitvector.count();
        // uint64_t* current_complement = new uint64_t[comp_size];

        vector<uint64_t> current_complement;

        // scan the bit vector and populate the complement as an array 
        ofstream fout;
        fout.open(filename, ios::binary | ios::out);
        uint64_t counter = 0;
        for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
            if (!(universe_bitvector[i])){
                uint64_t t = index_to_qgram( i, mask);
                fout.write(reinterpret_cast<char *>(&t), sizeof(uint64_t)); 
                // current_complement[counter] = t;
                current_complement.push_back(t);
                counter++;
            }
        }
        fout.close();

        cout << "found " << counter << " complement qgrams out of " << UNIVERSE_SIZE << endl << flush;

        cout << "Checking if all complements are correct: "<< flush;

        ifstream fin;
        fin.open(filename, ios::binary | ios::in);
        uint64_t gram;
        vector<uint64_t>::iterator compl_it = current_complement.begin();
        while (true){
            fin.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
            if (!fin) break;

            if(compl_it == current_complement.end()){
                cout << "POINTER ERROR!!" << endl << flush;
                break;
            }

            if(*compl_it != gram)
                cout << "ELEMENT ERROR!" << endl << flush;

            compl_it++; 
        }
        cout << endl;
        
        fin.close();

        csets_array[maskindex] = current_complement;
        // csets_sizes[maskindex] = comp_size;

        assert(UNIVERSE_SIZE == universe_bitvector.count() + counter);

    }

    return;
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
    // uint64_t g1 = 58318922431328316;  
    // uint64_t g2 = 878429593903304643;
    uint64_t g[N_hash_fctns];
    g[0] = {878429593903304643};

    cout << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
    cout << endl;

    // uint64_t** csets_array = new uint64_t*[N_hash_fctns];
    


    process_multiple_masks(g, "testfile");


    cout << "Starting complementary sets: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
    {
        cout << "Set " << i+1 << "of size "<< csets_array[i].size() << ": ";
        for(vector<uint64_t>::iterator it = csets_array[i].begin(); it!= csets_array[i].end() ; it++)
        {
            if(((it-csets_array[i].begin())%10000) == 0){
                cout << bitset<64>(*it) << "=";
                print_Q_gram(*it);
            }
        }
        cout << endl << flush;
    }
        
    cout << endl;


    // // sort_according_to_masks(g, csets, csets_sizes);

    // cout << "Complementary sets after sorting: " << endl << flush;
    // for(int i = 0; i< N_hash_fctns; i++)
    // {
    //     cout << "Set " << i+1 << ": ";
    //     for(int j = 0; j< csets_sizes[i] ; j++)
    //     {
    //         cout << bitset<64>(csets[i][j]) << "=";
    //         print_Q_gram(csets[i][j]);
    //     }
    //     cout << endl << flush;
    // }
        
    // cout << endl;

    // delete[] csets_array;


    return 0;
}
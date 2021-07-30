// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <pair>
#include <immintrin.h> // to use pdep, pext

// for mmap:
#include "../script/mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "../script/class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int N_tests = 3; // number of tests
constexpr int N_completions = 10; // number of completions for each template
constexpr int N_hash_fctns = 6;  // number of hash functions 
constexpr int target_size = 14; // target space size of hash functions
// constexpr int MAX_complement_size = 150000;
vector<uint64_t> compl_array[N_hash_fctns]; // array of vectors for complementary sets
vector<uint64_t> global_outcome;

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

void process_multiple_masks(uint64_t* mask_array){
    // TODO we can later remove this loop
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

        // vector<uint64_t> current_complement;
        compl_array[maskindex].clear();
        // scan the bit vector and populate the complement vector 
        uint64_t counter = 0;
        for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
            if (!(universe_bitvector[i])){
                uint64_t t = index_to_qgram( i, mask);
                // current_complement[counter] = t;
                compl_array[maskindex].push_back(t);
                counter++;
            }
        }

        cout << "found " << counter << " complement qgrams out of " << UNIVERSE_SIZE << endl << flush;
        // compl_array[maskindex] = current_complement;

        assert(UNIVERSE_SIZE == universe_bitvector.count() + counter);

    }

    return;
}



// g is the array of hash functions, of size N_hash_fctns
// csets is an array of arrays of complementary sets, of size N_hash_fctns * 
// compl_sizes is an array of ints, with the sizes of the corresponding complementary sets
void sort_according_to_masks(const uint64_t* g) // DEBUGGED
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

        // sort the corresponding vector according to the masks
        sort(compl_array[i].begin(), compl_array[i].end(), [&](uint64_t x, uint64_t y) {
            if((x & overlapmask) > (y&overlapmask)) // order according to the overlapping positions first
		        return false;

	        if((x & overlapmask) < (y&overlapmask)) // order according to the overlapping positions first
		        return true;
	
	        return (x & truemask) < (y&truemask); // if overlapping positions are equal, order according to the rest
        });
    }
}



// search in complementary set number compl_index for the interval with values equal to key over the mask overlapmask
pair<int,int> complementary_search(uint64_t template, int compl_index, uint64_t mask){
    vector<uint64_t> &curr_vector = compl_array[compl_index];
    int pos_of_equality = -1;

    // vector<uint64_t>::iterator beg = curr_vector.begin();
    // vector<uint64_t>::iterator end = curr_vector.end();
    int beg_pos = 0;
    int end_pos = curr_vector.size()-1;

    while (beg_pos <= end_pos && pos_of_equality < 0) {
        int mid = (beg_pos + end_pos)/2;
        
        if (((*(curr_vector.begin() + mid)) & mask) == (template & mask))
            pos_of_equality = mid;
        else if (((*(curr_vector.begin() + mid)) & mask) > (template & mask))
            end_pos = mid - 1;
        else
            beg_pos = mid + 1;
    }
  
    if(pos_of_equality == -1) // element does not occur
        return make_pair(-1,-1);
    
    cout << "Found pos of equality " << pos_of_equality << endl << flush;

    // now, we want to find the extremes of the interval
    int beg_equality = pos_of_equality;
    int end_equality = pos_of_equality;

    while(beg_equality-1 >= 0 && ((*(curr_vector.begin() + beg_equality-1)) & overlapmask) == (key & overlapmask)) // keep going unless we go past the beginning, or find a different value
        beg_equality--;
        

    while(end_equality+1 < curr_vector.size() && ((*(curr_vector.begin() + end_equality+1)) & overlapmask) == (key & overlapmask)) // keep going unless we go past the beginning, or find a different value
        end_equality++;

    return make_pair(beg_equality, end_equality);
    
}



void rec_compute_templates(uint64_t candidate, int function_index, const uint64_t* g, uint64_t* redmasks){
    // for now, we use a std function: [interval.first, interval.second) is the interval
    pair<vector<uint64_t>::iterator,vector<uint64_t>::iterator> interval = equal_range(compl_array[function_index].begin(), compl_array[function_index].end(), candidate, [=](uint64_t x, uint64_t y) {
		    return (x & redmasks[function_index]) < (y & redmasks[function_index]);
        }); // complementary_search(candidate, function_index, redmasks[function_index]);

    // if the range was empty, we have first= second 
    if(interval.first == interval.second) //-1 || interval.second == -1)
        return;

    if(function_index == N_hash_fctns -1){
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++){ //vector<uint64_t>::iterator it = compl_array[function_index].begin() + interval.first; it!= compl_array[function_index].begin() + interval.second +1; it++){
            assert(((*it) & redmasks[function_index])==(candidate & redmasks[function_index]));
            global_outcome.push_back(((*it) & g[function_index]) | candidate);
        }
    }
    else{
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++ ){ //compl_array[function_index].begin() + interval.first; it!= compl_array[function_index].begin() + interval.second +1; it++){
            assert(((*it) & redmasks[function_index])==(candidate & redmasks[function_index]));
            rec_compute_templates(((*it) & g[function_index]) | candidate, function_index+1, g, redmasks);
        }
            
    }
}


void compute_templates(const uint64_t *g){
    // TODO: sort functions according to the number of elements of the complementary 
    uint64_t redmasks[N_hash_fctns];

    // for the first function, no overlap
    redmasks[0] = 0;
    uint64_t overlapmask = g[0];
    
    // for each function, find the overlap with the previous and sort its corresponding array
    for(int i = 1; i< N_hash_fctns; i++)
    {
        // cout << "Considering function " << i+1 << " given by " << bitset<64>(g[i]) << endl << flush;
        uint64_t currentfctn = g[i]; 
        redmasks[i] = overlapmask & currentfctn;
        overlapmask |= currentfctn;

        // sort the corresponding vector of complementaries according to the masks
        sort(compl_array[i].begin(), compl_array[i].end(), [=](uint64_t x, uint64_t y) {
		    return (x & redmasks[i]) < (y & redmasks[i]);
        });

    }

    // start a recursive computation for every element of the first complementary set
    for(auto &x : compl_array[0])
        rec_compute_templates(x, 1, g, redmasks);

    // Now, file dump + complete and check
}


int main()
{
    uint64_t g[N_hash_fctns];
    srand(111);

    // Crete random functions
    int pos[target_size];
    int filled = 0;
    while(filled < target_size){
        int new_pos = rand() % 32;
        // TODO
    }
        

    cout << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
    cout << endl;    


    process_multiple_masks(g);

    sort_according_to_masks(g);




    return 0;
}
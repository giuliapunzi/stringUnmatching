// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <vector>
// #include <utility> // for pair
#include <immintrin.h> // to use pdep, pext

// for mmap:
#include "../Q-grams/mapfile.hpp"

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "../Q-grams/class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int N_tests = 3; // number of tests
constexpr int N_completions = 256; // number of completions for each template
constexpr int N_hash_fctns = 4;  // number of hash functions 
constexpr int target_size = 10; // target space size of hash functions
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

        compl_array[maskindex] = current_complement;
        // csets_sizes[maskindex] = comp_size;

        assert(UNIVERSE_SIZE == universe_bitvector.count() + counter);

    }

    return;
}


// search in complementary set number compl_index for the interval with values equal to key over the mask overlapmask
pair<int,int> complementary_search(uint64_t key, int compl_index, uint64_t overlapmask){
    vector<uint64_t> curr_vector = compl_array[compl_index];
    int compl_size = curr_vector.size();
    int pos_of_equality = -1;

    // vector<uint64_t>::iterator beg = curr_vector.begin();
    // vector<uint64_t>::iterator end = curr_vector.end();
    int beg_pos = 0;
    int end_pos = compl_size-1;

    while (beg_pos <= end_pos && pos_of_equality < 0) {
        int mid = beg_pos + (end_pos - beg_pos) / 2;
        
        if (((*(curr_vector.begin() + mid)) & overlapmask) == (key & overlapmask))
            pos_of_equality = mid;
  
        if (((*(curr_vector.begin() + mid)) & overlapmask) > (key & overlapmask))
            beg_pos = mid + 1;
  
        else
            end_pos = mid - 1;
    }
  
    if(pos_of_equality == -1) // element does not occur
        return make_pair(-1,-1);
    
    cout << "Found pos of equality " << pos_of_equality << endl << flush;

    // now, we want to find the extremes of the interval
    int beg_equality = pos_of_equality;
    int end_equality = pos_of_equality;

    while(beg_equality-1 >= 0 && ((*(curr_vector.begin() + beg_equality-1)) & overlapmask) == (key & overlapmask)) // keep going unless we go past the beginning, or find a different value
        beg_equality--;

    if(end_equality+1 < compl_size){
        cout << "at pos " << end_equality+1 << " we will have: " << endl;
        cout << "element of list unmasked: "  << bitset<64>((*(curr_vector.begin() + end_equality+1))) << " and masked: "  << bitset<64>((*(curr_vector.begin() + end_equality+1)) & overlapmask) << endl;
        cout << " key is unmasked: " << bitset<64>(key) << " and masked: " << bitset<64>(key & overlapmask) << endl << endl << flush;
    }
        

    while(end_equality+1 < compl_size && ((*(curr_vector.begin() + end_equality+1)) & overlapmask) == (key & overlapmask)) // keep going unless we go past the beginning, or find a different value
        end_equality++;

    
    cout << "recall the overlap mask " << bitset<64>(overlapmask) << endl << flush;

    return make_pair(beg_equality, end_equality);
}


// g is the array of hash functions, of size N_hash_fctns
// csets is an array of arrays of complementary sets, of size N_hash_fctns * 
// csets_sizes is an array of ints, with the sizes of the corresponding complementary sets
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

        // sort the corresponding array according to the masks
        sort(compl_array[i].begin(), compl_array[i].end(), [&](uint64_t x, uint64_t y) {
            if((x & overlapmask) > (y&overlapmask)) // order according to the overlapping positions first
		        return false;

	        if((x & overlapmask) < (y&overlapmask)) // order according to the overlapping positions first
		        return true;
	
	        return (x & truemask) < (y&truemask); // if overlapping positions are equal, order according to the rest
        });
    }
}





void rec_compute_templates(uint64_t candidate, int function_index, const uint64_t* g, uint64_t* redmasks){
    // for now, we use a std function: [interval.first, interval.second) is the interval
    pair<vector<uint64_t>::iterator,vector<uint64_t>::iterator> interval = equal_range(compl_array[function_index].begin(), compl_array[function_index].end(), candidate, [=](uint64_t x, uint64_t y) {
		    return (x & redmasks[function_index]) < (y & redmasks[function_index]);
        }); // complementary_search(candidate, function_index, redmasks[function_index]);

    cout << "Inside recursive call with index " << function_index << "; we have candidate " << bitset<64>(candidate) << " which occurs in range " << interval.first - compl_array[function_index].begin() << ", " << interval.second - compl_array[function_index].begin() << endl << endl;

    // if the range was empty, we have first= second 
    if(interval.first == interval.second) //-1 || interval.second == -1)
        return;

    if(function_index == N_hash_fctns -1){
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++){ //vector<uint64_t>::iterator it = compl_array[function_index].begin() + interval.first; it!= compl_array[function_index].begin() + interval.second +1; it++){
            assert(((*it) & redmasks[function_index])==(candidate & redmasks[function_index]));
            global_outcome.push_back(((*it)& g[function_index]) | candidate);
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
        // cout << "Considering function " << i << " given by " << bitset<64>(g[i]) << endl << flush;
        uint64_t currentfctn = g[i]; 
        redmasks[i] = overlapmask & currentfctn;
        // cout << "and between " << bitset<64>(overlapmask) << " and " << bitset<64>(currentfctn) << endl << endl << flush;

        overlapmask |= currentfctn;

        // sort the corresponding vector of complementaries according to the masks
        sort(compl_array[i].begin(), compl_array[i].end(), [=](uint64_t x, uint64_t y) {
		    return (x & redmasks[i]) < (y & redmasks[i]);
        });

    }

    cout << "redmasks: ";
    for(int i = 0; i<N_hash_fctns; i++)
        cout << bitset<64>(redmasks[i]) << ", ";
    cout << endl << endl;

    // start a recursive computation for every element of the first complementary set
    for(auto &x : compl_array[0])
        rec_compute_templates(x, 1, g, redmasks);

    // Now, file dump + complete and check
    cout << "Global outcomes (" << global_outcome.size() << "): ";
    for(auto x : global_outcome)
        cout << ", " << bitset<64>(x);
    cout << endl;
}



int main()
{
    // uint64_t g1 = 58318922431328316;  
    // uint64_t g2 = 878429593903304643;
    // uint64_t g[N_hash_fctns];
    // g[0] = {878429593903304643};

    uint64_t g[N_hash_fctns];
    g[0] = 0b0011001111000000001100001100110000000000110000000011000011110000;
    g[1] = 0b1111001100110011000011000000001100001100000000001111000000000000;
    g[2] = 0b0000110000001111001100110000000011000000110011000000000011001100;
    g[3] = 0b1100110000000000110000110000110000000000001100001100001100110011;

    
    vector<uint64_t> set0 = {0, 0b1111111111111111111111111111111111111111111111111111111111111111, 0b0001000010000000000100000100000000000000110000000001000001110000, 0b0011001011000000000000001000010000000000010000000000000000100000, 0b0010001101000000000100000000110000000000100000000001000011100000};
    vector<uint64_t> set1 = {0b1100000011111111111111111111111111111111111111111100111111111111,    0b0011001100000000000000000000000000000000000000000011000000000000, 0b1111001000110000000001000000000100001100000000001000000000000000, 0b0101001000110000000011000000001000001000000000000011000000000000, 0b1011000000010011000000000000000100001000000000001001000000000000, 0b1111000000110000000001000000000100001100000000001001000000000000};
    vector<uint64_t> set2 = {0b1111111111111111111111111111111111111111111111111111111111111111, 0, 0b0000100000000011000100100000000000000000010001000000000010001100, 0b0000100000001001000100100000000000000000100000000000000011001000, 0b0000110000000100001000010000000001000000110001000000000000000000};
    vector<uint64_t> set3 = {0b1111111111111111111111111111111111111111111111111111111111111111, 0, 0b0000100000000000100000010000000000000000000100000100001100000001, 0b1100010000000000000000000000100000000000000100000100001000110010};

    // uint64_t* csets[N_hash_fctns] = {set0, set1, set2, set3};

    compl_array[0] = set0;
    compl_array[1] = set1;
    compl_array[2] = set2;
    compl_array[3] = set3;



    cout << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
    cout << endl;

    // uint64_t** compl_array = new uint64_t*[N_hash_fctns];
    
    


    // process_multiple_masks(g, "testfile");


    cout << "Starting complementary vectors: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
    {
        cout << "Set " << i+1 << " of size "<< compl_array[i].size() << ": ";
        for(vector<uint64_t>::iterator it = compl_array[i].begin(); it!= compl_array[i].end() ; it++)
        {
            // if(((it-compl_array[i].begin())%10000) == 0){
                cout << bitset<64>(*it) << "=";
                print_Q_gram(*it);
            // }
        }
        cout << endl << flush;
    }
        
    cout << endl;


    sort_according_to_masks(g);

    cout << "Complementary sets after sorting: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
    {
        cout << "Set " << i+1 << ": ";
        for(int j = 0; j< compl_array[i].size() ; j++)
        {
            cout << bitset<64>(compl_array[i][j]) << "=";
            print_Q_gram(compl_array[i][j]);
        }
        cout << endl << flush;
    }
        
    cout << endl;


    // pair<int,int> search = complementary_search(0b0000100000000001000100100000001000000000100000001111000011000000, 2, 0b0000000000000011001100000000000000000000110000000000000011000000);
    cout << "Searching for " << bitset<64>(0b0000100000000000001000100000001000000000110000001111000000000000) << " in the third set." << endl << flush;
    // cout << "Result of the first search: " << search.first << ", " << search.second << endl << flush;

    pair<vector<uint64_t>::iterator,vector<uint64_t>::iterator> interval = equal_range(compl_array[2].begin(), compl_array[2].end(), 0b0000100000000001000100100000001000000000100000001111000011000000, [=](uint64_t x, uint64_t y) {
		    return (x & 0b0000000000000011001100000000000000000000110000000000000011000000) < (y & 0b0000000000000011001100000000000000000000110000000000000011000000);
        });
    cout << "With standard: " << interval.first-compl_array[2].begin() << ", " << interval.second-compl_array[2].begin() << endl << endl;

    // search = complementary_search(0b1011000001000100110000001010001010100001001010101001101000010101, 1, 0b0011001100000000000000000000000000000000000000000011000000000000);
    cout << "Searching for " << bitset<64>(0b1011000101000100110000001010001010100001001010101000101000010101) << " in the second set." << endl << flush;
    // cout << "Result of the second search: " << search.first << ", " << search.second << endl << flush;

    uint64_t key = 0b1111000000101010010010101010000010101010101010001001010011111010;
    uint64_t mask = g[1] & g[0];
    interval = equal_range(compl_array[1].begin(), compl_array[1].end(), key, [=](uint64_t x, uint64_t y) {
		    return (x & mask) < (y & mask);
        });
    cout << "With standard: " << interval.first-compl_array[1].begin() << ", " << interval.second-compl_array[1].begin() << endl << endl;

    // cout << "Masked compl array: ";
    // for(auto x : compl_array[1])
    //     cout << bitset<64>(x & mask) << ", ";
    // cout << endl;


    compute_templates(g);

    return 0;
}
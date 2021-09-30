// compile with option -mbmi2 -std=c++2a
#include <iostream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <immintrin.h> // to use pdep, pext
#include <ctime> // for elapsed time
#include <algorithm>
#include <cstring>

#include <unordered_map>

using namespace std;

constexpr auto maxQ = 32; // max value of Q as we use 64 bit integers
constexpr auto Q = 5; // strings will be represented as 64 bit integers, stored in the last (least significative) 2*Q positions

constexpr int N_hash_fctns = 2;  // number of hash functions 
constexpr int target_size = 3; // target space size of hash functions
constexpr auto MASK_WEIGHT = 2*target_size;  // number of 1s, twice the number of selected chars (as the alphabet is 4)

constexpr auto UNIVERSE_SIZE = 64; // NO: 4^target_size! 1024; //268435456;    // 4^14 = 268435456

bitset<UNIVERSE_SIZE> universe_bitvector_array[N_hash_fctns];

vector<uint64_t> compl_array[N_hash_fctns]; // array of vectors for complementary sets
vector<uint64_t> global_outcome; // global outcome will be the final 

constexpr int SEED = 13; //19; //227; // 87; // 11

constexpr int MIN_DIST = 9;

const char * text = "ACGATATATGCTACGACTGCGCGCGGCGCGCGATCGATGCTAGCGCTATAGCTAGTCGCGCGCGCGGCGCGGGGGGGGGGGGGGGGGGGGGGATTATATATAGTCGATCGATGCTAGCATGCTCGTGCGGATATTATATATCGTCGTACGTAGCTACGTAGCTAGCTGATCGATGCTAGTCGCGGCGCGCGATCGATCGATCGATCGATCGATCGTACGTAGTGCATGCTAGTTTAATATCGTACGTCTCTCTGCAGGAGAGTCGATCGTGCATTGTACGTAGCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATATCGTAGCTACGTACTGGCGCGCGCGCGGCTATTATCGATCTACTACGTCGGCGCATAGCGTAGCTAGGCGATCGAGGCGGCTAGCTAGCTACTAGTTAGCGGCGAGTAGTCGATCGACGTAGGCGATGCTAGCATCGGCGGGGGGGCGATCGTATATTTATATACCCCGGCAGGAGAGGGGAGAGAAAAAAATATTATATATTATCGATCGTACGTAGCTACGTAGCGCGCGCGATCTAGCATCTCGCGGCGCG"; //= map_file("./halfY.txt", textlen); //= map_file("./data/all_seqs.fa", textlen);   


// given an uint64_t, print it in A,C,G,T alphabet its last Q characters
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
inline uint64_t index_to_qgram(uint64_t index, uint64_t mask){
    return _pdep_u64(index, mask);
}
 

// given two uint64_t, compute their Hamming distance ============= CHANGED ===============
// __attribute__((always_inline)) 
int Hamming_distance(uint64_t x, uint64_t y, uint64_t Qmask) // DEBUGGED 
{
    uint64_t diff = (x^y); 
    diff &= Qmask;
    diff |= (diff << 1); 
    diff &= 0xAAAAAAAAAAAAAAAA;

    return __builtin_popcountll(diff); 
}



// extracts Qgrams from text, inserts them in a binary file 
void extract_Q_grams(){
    // map file
    size_t textlen = strlen(text);   
    // const char * text = map_file("./data/all_seqs.fa", textlen); 
    
    ofstream fout, foutACGT;
    fout.open("./DEBUGall" + to_string(Q) + "grams_repetitions", ios::binary | ios::out);
    foutACGT.open("./DEBUG" + to_string(Q) + "grams_ACGT", ios::out);

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;
    auto count_situation = 0;

    while(i < textlen){
        switch (toupper(text[i]))
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
        case '\n':
            skip = true;
            break;
        case '>':
        case ';':
            while( i < textlen && text[i] != '\n') i++;
            key = 0;
            key_len = 0;
            skip = true;
            break;
        default:
            key = 0;
            key_len = 0;
            skip = true;
            break;
        }
        i++;
        if (skip){ 
            skip = false; 
            continue;
        }
        // here only if the current char is A, C, G, T
        if (++key_len == Q){
            key_len = Q-1;        // for the next iteration
            
            // push out to output
            fout.write(reinterpret_cast<char *>(& key), sizeof(uint64_t)); 
            count_situation++;

            uint64_t gram = key;

            // write to ACGT output
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
            foutACGT << string(s) << endl << flush;

        }
        key <<= 2;            // shift two bits to the left

        if(count_situation == 10000000) 
            cout << "*" << flush;
    }

    fout.close();
    // unmap_file(text, textlen);
    cout << "Total count: " << count_situation << endl;
}



// ==================================== CHANGED! ==================================
// directly from main file instead of through parikh classes
// need to open main file and fill the bitvector when parsing it
void process_multiple_masks(uint64_t* mask_array){
    // bitset<UNIVERSE_SIZE> universe_bitvector_array[N_hash_fctns] = {};  // 33 554 432 bytes * N_hash_fctns, about 200MB for 6 hash fctns
    // vector<bitset<UNIVERSE_SIZE>> universe_bitvector_array;

    // TODO we can later remove this loop
    for(int maskindex=0; maskindex < N_hash_fctns; maskindex++) {
        // initialize bit vector
        universe_bitvector_array[maskindex].reset();
    }

    cout << "Bitvector initialized!" << endl << flush;
    
    // map file (for debug, use a string)
    size_t textlen = 0;   
    textlen = strlen(text);
    // cout << "It is " << text << endl << flush;

    // for (auto i =0; i < 4; i++) char_counter[i] = 0;
    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;
    auto count_situation = 0;

    while(i < textlen){
        switch (toupper(text[i]))
        {
        case 'A':
            // key |= 0x0;
            // char_counter[0]++;
            break;
        case 'C':
            key |= 0x1;
            // char_counter[1]++;
            break;
        case 'G':
            key |= 0x2;
            // char_counter[2]++;
            break;
        case 'T':
            key |= 0x3;
            // char_counter[3]++;
            break;
        case '\n':
            skip = true;
            break;
        case '>':
        case ';':
            while( i < textlen && text[i] != '\n') i++;
            key = 0;
            key_len = 0;
            // for (auto i =0; i < 4; i++) char_counter[i] = 0;
            skip = true;
            break;
        default:
            key = 0;
            key_len = 0;
            // for (auto i =0; i < 4; i++) char_counter[i] = 0;
            skip = true;
            break;
        }
        i++;
        if (skip){ 
            skip = false; 
            continue;
        }
        // here only if the current char is A, C, G, T
        if (++key_len == Q){
            key_len = Q-1;        // for the next iteration
            // check_Q_gram( key );  // & mask when Q < 32
            // char_counter[(key >> (Q-1+Q-1)) & 0x3]--;  // exiting char
            
            // insert into all mask arrays
            for(int maskindex=0; maskindex < N_hash_fctns; maskindex++) {
                if(!universe_bitvector_array[maskindex][qgram_to_index(key, mask_array[maskindex])])
                    universe_bitvector_array[maskindex][qgram_to_index(key, mask_array[maskindex])] = true;
            
            }

            count_situation++;

        }
        key <<= 2;            // shift two bits to the left

        if(count_situation == 10000000) 
            cout << "*" << flush;
    }

    cout << "found " << universe_bitvector_array[0].count() << " qgrams for first function " << endl << flush;
    cout << "found " << universe_bitvector_array[1].count() << " qgrams for second function " << endl << flush;

    // TODO we can later remove this loop
    for(int maskindex=0; maskindex < N_hash_fctns; maskindex++) {
        // initialize bit vector
        universe_bitvector_array[maskindex].reset();
    }

    // instead of scanning the text, let us go through the binary file of Qgrams (with repetitions)
    ifstream inputQgrams;
    inputQgrams.open("./DEBUGall" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    uint64_t gram;
    while (true){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;

        // insert into all mask arrays
        for(int maskindex=0; maskindex < N_hash_fctns; maskindex++) {
            if(!universe_bitvector_array[maskindex][qgram_to_index(gram, mask_array[maskindex])])
                universe_bitvector_array[maskindex][qgram_to_index(gram, mask_array[maskindex])] = true;
        }

        count_situation++;

        if(count_situation == 10000000) 
            cout << "*" << flush;
    }
    inputQgrams.close();

    cout << endl;
    cout << "From file, found " << universe_bitvector_array[0].count() << " qgrams for first function " << endl << flush;
    cout << "From file, found " << universe_bitvector_array[1].count() << " qgrams for second function " << endl << flush;


    // unmap_file(text, textlen);
    cout << endl << flush;

    // last for loop populates the complementary for each function
    for(int maskindex = 0; maskindex < N_hash_fctns; maskindex++){
        cout << "found " << universe_bitvector_array[maskindex].count() << " qgrams for " << maskindex << "th function." << endl << flush;

        compl_array[maskindex].clear();

        // scan the bit vector and populate the complement vector 
        uint64_t counter = 0;
        for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
            if (!(universe_bitvector_array[maskindex][i])){
                uint64_t t = index_to_qgram( i, mask_array[maskindex]); // PROBLEM HERE
                cout << "Adding index i=" << i << ", corresponding to qgram " << bitset<64>(t) << endl << flush;
                // current_complement[counter] = t;
                compl_array[maskindex].push_back(t);
                counter++;
            }
        }

        cout << "found " << counter << " complement qgrams out of " << UNIVERSE_SIZE << endl << flush;
        // compl_array[maskindex] = current_complement;

        assert(UNIVERSE_SIZE == universe_bitvector_array[maskindex].count() + counter);

    }

    return;
}


// ======================================================================


// g is the array of hash functions, of size N_hash_fctns
// csets is an array of arrays of complementary sets, of size N_hash_fctns * 
// compl_sizes is an array of ints, with the sizes of the corresponding complementary sets
void sort_according_to_masks(const uint64_t* g) 
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
pair<int,int> complementary_search(uint64_t Qgram_template, int compl_index, uint64_t mask){
    vector<uint64_t> &curr_vector = compl_array[compl_index];
    int pos_of_equality = -1;

    // vector<uint64_t>::iterator beg = curr_vector.begin();
    // vector<uint64_t>::iterator end = curr_vector.end();
    int beg_pos = 0;
    int end_pos = curr_vector.size()-1;

    while (beg_pos <= end_pos && pos_of_equality < 0) {
        int mid = (beg_pos + end_pos)/2;
        
        if (((*(curr_vector.begin() + mid)) & mask) == (Qgram_template & mask))
            pos_of_equality = mid;
        else if (((*(curr_vector.begin() + mid)) & mask) > (Qgram_template & mask))
            end_pos = mid - 1;
        else
            beg_pos = mid + 1;
    }
  
    if(pos_of_equality == -1) // element does not occur
        return make_pair(-1,-1);
    
    cout << "Found pos of equality " << pos_of_equality << endl << flush;

    // now, we want to find the extremes of the interval
    uint64_t beg_equality = pos_of_equality;
    uint64_t end_equality = pos_of_equality;

    while(beg_equality-1 >= 0 && ((*(curr_vector.begin() + beg_equality-1)) & mask) == (Qgram_template & mask)) // keep going unless we go past the beginning, or find a different value
        beg_equality--;
        

    while(end_equality+1 < curr_vector.size() && ((*(curr_vector.begin() + end_equality+1)) & mask) == (Qgram_template & mask)) // keep going unless we go past the beginning, or find a different value
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
            cout << "Found a template \t" << flush;
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


    for(auto x : compl_array[1])
    {
        // print_Q_gram(x);
        // cout << bitset<64>(x) << endl;
    }

        

    // for each function, find the overlap with the previous and sort its corresponding array
    for(int i = 1; i< N_hash_fctns; i++)
    {
        // cout << "Considering function " << i+1 << " given by " << bitset<64>(g[i]) << endl << flush;
        uint64_t currentfctn = g[i]; 
        redmasks[i] = overlapmask & currentfctn;
        overlapmask |= currentfctn;

        auto currentred = redmasks[i];

        // sort the corresponding vector of complementaries according to the masks
        sort(compl_array[i].begin(), compl_array[i].end(), [&](uint64_t &x, uint64_t &y) {
		    return (x & currentred) < (y & currentred);
        });

    }

    cout << "=======================================" << endl << flush;

    for(auto x : compl_array[1]){
        // cout << bitset<64>(x) << endl;
        // print_Q_gram(x);
    }

    // start a recursive computation for every element of the first complementary set
    for(auto &x : compl_array[0])
        rec_compute_templates(x, 1, g, redmasks);

    cout << endl << flush;

    // File dump of results
    // ofstream outputfile, binaryout; 
    // outputfile.open("../exp_results/" +to_string(N_hash_fctns) + "MultFunctSeed" + to_string(SEED), ios::app);
    // binaryout.open("../exp_results/" +to_string(N_hash_fctns) + "MultFunctBinarySeed"  + to_string(SEED), ios::binary | ios::app);
    // outputfile << "Test with " << N_hash_fctns << " functions: " << endl;
    // outputfile << "Functions g: " << endl << flush;
    // for(int i = 0; i< N_hash_fctns; i++)
    //     outputfile << "g_" << i << ": " << bitset<64>(g[i]) << endl;
    // outputfile << endl << endl;

    // outputfile << "Templates to check are " << global_outcome.size() << ": " << endl;
    cout << "Templates found are " << global_outcome.size() <<  endl << flush;

    for(uint64_t i = 0; i < global_outcome.size(); i++){
        uint64_t templ = global_outcome[i];
        cout << bitset<64>(templ) << " = "; 
        print_Q_gram(templ);
    //     binaryout.write(reinterpret_cast<char *>(&templ), sizeof(uint64_t)); 
    //     outputfile << bitset<64>(templ) << ", "; //print_Q_gram(templ);
    }
    cout << endl;
    // outputfile << endl << endl;

    // binaryout.close();
    // outputfile.close();
}



// ==================================== CHANGED! ==================================
// FUNCTIONS WILL HAVE THE FIRST (LEFTMOST, MOST SIGNIFICANT) POSITIONS EQUAL TO ZERO
// this is because when filling the keys for the text, we fill them inserting from the right.
void build_functions(uint64_t* g){ // DEBUGGED
    // srand(SEED);
    srand(time(NULL));

    bool all_covered = false; // all_covered is now different: need and with last maxQ-Q pos
    while( !all_covered ){
        // Create random functions in last (least significant) 2*Q positions
        for(int i=0; i<N_hash_fctns; i++){
            int pos[target_size];
            pos[0] = rand() % Q;
            int filled = 1;
            bool new_position = true;
            while(filled < target_size){
                int candidate_pos = rand() % Q;
                new_position = true;
                for(int j=0; j<filled; j++){
                    if(pos[j] == candidate_pos)
                        new_position = false;
                }

                if(new_position){
                    pos[filled] = candidate_pos;
                    filled++;
                }
            }

            sort(pos, pos + target_size);

            // cout << "Array of random positions is: ";
            // for(int j=0; j<target_size; j++)
            //     cout << pos[j] << ", ";
            // cout << endl;

            // for(int j=0; j<target_size; j++)
            //     pos[j] =2*pos[j];

            // cout << "Array of doubled random positions is: ";
            // for(int j=0; j<target_size; j++)
            //     cout << pos[j] << ", ";
            // cout << endl;

            // build the correspondin uint64 number and assign it to g[i]
            uint64_t currg = 0b11;
            for(int j=1; j< target_size; j++){
                // cout << "g1 is " << bitset<64>(currg) << endl << flush;
                currg <<= (2*pos[j] - 2*pos[j-1]);
                currg |= 0b11;
            }

            // cout << "Before final shift: " << bitset<64>(currg) << endl << flush;

            // final shift needs to be changed: still shifting by Q allows us to have the most significant bits set to 0
            currg <<= (2*Q - 2 - 2*pos[target_size-1]);
            g[i] = currg;

            // cout << "After final shift: " << bitset<64>(g[i]) << endl << flush;

            assert( __builtin_popcountll(g[i]) == MASK_WEIGHT );
        }

        // cout << "Functions g: " << endl << flush;
        // for(int i = 0; i< N_hash_fctns; i++)
        //     cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
        // cout << endl; 

        // compute the complementary to check if all positions are covered

        // tail has the first 2(maxQ-Q) bits set to 0
        uint64_t tail = 0;
        for(int i=0; i < maxQ-Q; i++)
        {
            tail <<=2;
            tail |= 0b11;
        }
        tail <<= 2*Q;
            

        // cout << "Tail is " << bitset<64>(tail) << endl;

        // check whether any g has positions in the tail, that is, the and is 0
        for(int i=0; i< N_hash_fctns; i++)
            assert((tail & g[i]) == 0);
        

        uint64_t cg = g[0];
        for(int i=1; i<N_hash_fctns; i++)
            cg |= g[i];
        
        cg |= tail;

        // cout << "Complementary is " << bitset<64>(~cg) << endl;

        // if( ~cg == 0)
            all_covered = true; // for now uncovered positions
    }
}


// ======================================================================


// CHECK NEEDS TO BE CHANGED
void check (uint64_t Qmask)
{
    uint64_t templ_size = global_outcome.size();
    cout << "Candidates are " << global_outcome.size() << ": " << endl << flush;
    for(uint64_t i = 0; i < global_outcome.size(); i++){
        uint64_t templ = global_outcome[i];
        cout << bitset<64>(templ) << " = "; 
        print_Q_gram(templ);
    }
    cout << endl << endl << flush;

    if(global_outcome.size() == 0)
        return;

    // initialize distances' vector
    // int mindist[templ_size];
    vector<int> mindist;
    for(uint64_t templindex = 0; templindex < templ_size; templindex++)
        mindist.push_back(Q+1);
        // mindist[templindex] = Q+1; //Hamming_distance(completions[templindex], key); 

    cout << endl << "Printing min distances before starting: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        cout << mindist[dd] << " " << flush;
    cout << endl;

    // instead of scanning the text, let us go through the binary file of Qgrams (with repetitions)
    ifstream inputQgrams;
    inputQgrams.open("./DEBUGall" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    uint64_t gram;
    while (true){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;

        // update distances
        for(uint64_t j=0; j<templ_size; j++){
            int dist = Hamming_distance(gram, global_outcome[j], Qmask);
            if(dist < mindist[j])
                mindist[j] = dist;
        }
    }
    inputQgrams.close();


    // ofstream outputfile; 
    // outputfile.open("../exp_results/" +to_string(N_hash_fctns) +"MultFunctSeed" + to_string(SEED), ios::app);

    // cout << endl << "Printing min distances for the " << templ_size << " templates: " << flush;
    // for(uint64_t dd = 0; dd< templ_size; dd++)
    //     cout << mindist[dd] << " " << flush;

    // outputfile << endl << "Printing min distances for the " << templ_size << " templates: " << flush;
    // for(uint64_t dd = 0; dd< templ_size; dd++)
    //     outputfile << mindist[dd] << " " << flush;
    // outputfile << endl;

    uint64_t max_dist_index = 0;
    for(uint64_t i=0; i<global_outcome.size(); i++){
        if(mindist[i] > mindist[max_dist_index])
            max_dist_index = i;
    }
    cout << "Max minimum distance of " << mindist[max_dist_index] << " reached by gram " << bitset<64>(global_outcome[max_dist_index]) << endl;


    // outputfile << "Max minimum distance of " << mindist[max_dist_index] << " reached by gram " << bitset<64>(global_outcome[max_dist_index]) << endl;

    // outputfile << endl << endl << flush;

    // outputfile.close();

    // if(mindist[max_dist_index] >= MIN_DIST){
    //     // create a file containing all the far Qgrams
    //     ofstream goodgrams;
    //     goodgrams.open("../exp_results/" + to_string(Q) + "gramsDist" + to_string(MIN_DIST), ios::binary | ios::out | ios::app );

    //     for(uint64_t i = 0; i < global_outcome.size(); i++){
    //         if(mindist[i] == mindist[max_dist_index])
    //             goodgrams.write(reinterpret_cast<char *>(&(global_outcome[i])), sizeof(uint64_t)); 
    //     }
    //     goodgrams.close();
    // }
    
    return;
}



int main() // NEED TO MAKE SURE THAT THE CORRESPONDING QGRAM FILE EXISTS!
{
    uint64_t g[N_hash_fctns];
    clock_t begin = clock();
    build_functions(g);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    
    fstream checkfile;
    checkfile.open("./DEBUGall" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    if(!checkfile.is_open())
    {
        throw logic_error("Preprocessing not done correctly!");
        return 0;
    }
    checkfile.close();

    
    extract_Q_grams();

    

    // Qmask has the first 2(maxQ-Q) bits set to 0, the last 2Q set to 1
    uint64_t Qmask = 0b11;
    for(int i=0; i < Q-1; i++)
    {
        Qmask <<=2;
        Qmask |= 0b11;
    }
    cout << "Mask Qmask is " << bitset<64>(Qmask) << endl << flush;

    cout << "Functions found in " << elapsed_secs << " seconds are: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i << ": " << bitset<64>(g[i]) << endl;
    cout << endl; 

    // cout << "Computing distance between 0000000000000000000000000101011011011000001011100100001011111001 and 0000000000000000000000000001001010011001001010100100011011101000" << endl;
    // // x = 0b0000000000000000000000000101011011011000001011100100001011111001
    // uint64_t x = 0b1000001110000000000000000101011011011000001011100100001011111001;
    // uint64_t y = 0b0000100000000111100000000001001010011001001010100100011011101000;

    // cout << "x = " << bitset<64>(x) << endl << "y = " << bitset<64>(y) << endl;
    // cout << Hamming_distance( x,y,Qmask )  << endl;
    
    
    /*
    Functions g:
    g0: 0011000000110000001111001100111100111100110000110000001100110011
    g1: 0011111111001111110000110000110000001100000011000000000011111100
    g2: 0011001100111100000000001111110000001111000000001111001100111100
    g3: 0000001100110000000000111100111111111111001100001111000000000011
    g4: 1100000011001111001100000000001100000011001111001111111100001100
    g5: 0000001100001100110000001100000011111111000000111100001100111111
    */

    // test bitset
    // bitset<100000000> bittest[N_hash_fctns]; // CANNOT MAKE IT OF UNIVERSE_SIZE: problem at 100 000 000. 
    
    begin = clock();
    process_multiple_masks(g); // ABOUT 20 MINS WITH 6 MASKS
    end = clock();
    // elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Masks processed in "<< elapsed_secs <<" time; complementary sets have sizes: " << flush; // Masks processed; complementary sets have sizes: |C0|= 38831      |C1|= 164169    |C2|= 86982   |C3|= 212690     |C4|= 114491    |C5|= 126248
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "|C" << i << "|= " << compl_array[i].size() << "\t " << flush;
    cout << endl<< flush; 

    // sort_according_to_masks(g); // couple of minutes


    // cout << "Masks have been sorted" << endl << flush;

    // begin = clock();
    compute_templates(g);
    // end = clock();
    // elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // for(auto x : compl_array[0])
    //     print_Q_gram(x);


    cout << "End of template computation, which took " << elapsed_secs << " seconds. " << endl << flush;

    // if(global_outcome.size() == 0)
    //     return 0;

    // begin = clock();
    check(Qmask);
    // end = clock();
    // elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // cout << "End of check, which took " << elapsed_secs << " seconds. " << endl << flush;

    return 0;
}
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

#include <unordered_map>


// for mmap: DOES NOT WORK ON PC 
#include "../script/mapfile.hpp"


using namespace std;

constexpr auto maxQ = 32; // max value of Q as we use 64 bit integers
constexpr auto Q = 20; // strings will be represented as 64 bit integers, stored in the first (most significative) 2*Q positions

constexpr int N_hash_fctns = 6;  // number of hash functions 
constexpr int target_size = 14; // target space size of hash functions
constexpr auto MASK_WEIGHT = 2*target_size;  // number of 1s, twice the number of selected chars (as the alphabet is 4)

vector<uint64_t> compl_array[N_hash_fctns]; // array of vectors for complementary sets
vector<uint64_t> global_outcome;

constexpr int SEED = 13; //19; //227; // 87; // 111
constexpr int MIN_DIST = 9;

constexpr auto UNIVERSE_SIZE = 268435456;    // 4^target_size = 268435456
bitset<UNIVERSE_SIZE> universe_bitvector_array[N_hash_fctns];


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
inline uint64_t index_to_qgram(uint64_t index, uint64_t mask){
    return _pdep_u64(index, mask);
}
 

// given two uint64_t, compute their Hamming distance only on the Qmasked positions
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y, uint64_t Qmask) 
{
    uint64_t diff = (x^y); 
    diff &= Qmask;
    diff |= (diff << 1); 
    diff &= 0xAAAAAAAAAAAAAAAA;

    return __builtin_popcountll(diff); 
}



// process masks goes through a pre-constructed binary file of all input Qgrams (with repetitions) 
// the function fills the complementary sets array 
void process_multiple_masks(uint64_t* mask_array){
    
    for(int maskindex=0; maskindex < N_hash_fctns; maskindex++) {
        // empty bit vector
        universe_bitvector_array[maskindex].reset();
    }

    cout << "Bitvector initialized!" << endl << flush;
    
    // instead of scanning the text, let us go through the binary file of Qgrams (with repetitions)
    ifstream inputQgrams;
    inputQgrams.open("../data/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
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
    cout << endl<<flush;

    // last for loop populates the complementary for each function
    for(int maskindex = 0; maskindex < N_hash_fctns; maskindex++){
        cout << "found " << universe_bitvector_array[maskindex].count() << " qgrams for " << maskindex << "th function." << endl << flush;

        compl_array[maskindex].clear();

        // scan the bit vector and populate the complement vector 
        uint64_t counter = 0;
        for (uint64_t i = 0; i < UNIVERSE_SIZE; i++){
            if (!(universe_bitvector_array[maskindex][i])){
                uint64_t t = index_to_qgram( i, mask_array[maskindex]);
                // cout << "Adding index i=" << i << ", corresponding to qgram " << bitset<64>(t) << endl << flush;
                compl_array[maskindex].push_back(t);
                counter++;
            }
        }

        cout << "Found " << counter << " complement qgrams out of " << UNIVERSE_SIZE << endl << flush;
        assert(UNIVERSE_SIZE == universe_bitvector_array[maskindex].count() + counter);
    }

    return;
}


void rec_compute_templates(uint64_t candidate, int function_index, const uint64_t* g, uint64_t* redmasks){
    // for now, we use a std function: [interval.first, interval.second) is the interval
    pair<vector<uint64_t>::iterator,vector<uint64_t>::iterator> interval = equal_range(compl_array[function_index].begin(), compl_array[function_index].end(), candidate, [=](uint64_t x, uint64_t y) {
		    return (x & redmasks[function_index]) < (y & redmasks[function_index]);
        }); 

    // if the range was empty, we have first= second 
    if(interval.first == interval.second)
        return;

    if(function_index == N_hash_fctns -1){
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++){ 
            assert(((*it) & redmasks[function_index])==(candidate & redmasks[function_index]));
            global_outcome.push_back(((*it) & g[function_index]) | candidate);
            cout << "Found a template \t" << flush;
        }
    }
    else{
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++ ){ 
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

    cout << endl << flush;

    // file dump of results
    ofstream outputfile, binaryout; 
    outputfile.open("../exp_results/Q=" + to_string(Q) + "/" + to_string(N_hash_fctns) + "FunctSeed" + to_string(SEED), ios::app);
    binaryout.open("../exp_results/Q=" + to_string(Q) + "/" + to_string(N_hash_fctns) + "FunctBinarySeed"  + to_string(SEED), ios::binary | ios::app);
    outputfile << "Test with " << N_hash_fctns << " functions: " << endl;
    outputfile << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        outputfile << "g_" << i << ": " << bitset<64>(g[i]) << endl;
    outputfile << endl << endl;

    outputfile << "Templates to check are " << global_outcome.size() << ": " << endl;
    cout << "Templates found are " << global_outcome.size() <<  endl << flush;

    for(uint64_t i = 0; i < global_outcome.size(); i++){
        uint64_t templ = global_outcome[i];
        cout << bitset<64>(templ) << " = "; 
        print_Q_gram(templ);
        binaryout.write(reinterpret_cast<char *>(&templ), sizeof(uint64_t)); 
        outputfile << bitset<64>(templ) << ", "; 
    }
    outputfile << endl << endl;

    binaryout.close();
    outputfile.close();
}


// FUNCTIONS WILL HAVE THE FIRST (LEFTMOST, MOST SIGNIFICANT) POSITIONS EQUAL TO ZERO
// this is because when filling the keys for the text, we fill them inserting from the right.
void build_functions(uint64_t* g){ 
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


            // build the correspondin uint64 number and assign it to g[i]
            uint64_t currg = 0b11;
            for(int j=1; j< target_size; j++){
                currg <<= (2*pos[j] - 2*pos[j-1]);
                currg |= 0b11;
            }

            // final shift needs to be changed: still shifting by Q allows us to have the most significant bits set to 0
            currg <<= (2*Q - 2 - 2*pos[target_size-1]);
            g[i] = currg;

            assert( __builtin_popcountll(g[i]) == MASK_WEIGHT );
        }

    
        // compute the complementary to check if all positions are covered
        // tail has the first 2(maxQ-Q) bits set to 0
        uint64_t tail = 0;
        for(int i=0; i < maxQ-Q; i++)
        {
            tail <<=2;
            tail |= 0b11;
        }
        tail <<= 2*Q;
            

        // check whether any g has positions in the tail, that is, the and is 0
        for(int i=0; i< N_hash_fctns; i++)
            assert((tail & g[i]) == 0);
        

        uint64_t cg = g[0];
        for(int i=1; i<N_hash_fctns; i++)
            cg |= g[i];
        
        cg |= tail;

        // cout << "Complementary is " << bitset<64>(~cg) << endl;

        if( ~cg == 0)
            all_covered = true; // for now uncovered positions
    }
}



// again, check goes through the binary file of all Qgrams
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
    vector<int> mindist;
    for(uint64_t templindex = 0; templindex < templ_size; templindex++)
        mindist.push_back(Q+1);

    cout << endl << "Printing min distances before starting: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        cout << mindist[dd] << " " << flush;
    cout << endl;

    // instead of scanning the text, let us go through the binary file of Qgrams (with repetitions)
    ifstream inputQgrams;
    inputQgrams.open("../data/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
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


    ofstream outputfile; 
    outputfile.open("../exp_results/Q=" + to_string(Q) + "/" +to_string(N_hash_fctns) +"FunctSeed" + to_string(SEED), ios::app);

    cout << endl << "Printing min distances for the " << templ_size << " templates: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        cout << mindist[dd] << " " << flush;

    outputfile << endl << "Printing min distances for the " << templ_size << " templates: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        outputfile << mindist[dd] << " " << flush;
    outputfile << endl;

    uint64_t max_dist_index = 0;
    for(uint64_t i=0; i<global_outcome.size(); i++){
        if(mindist[i] > mindist[max_dist_index])
            max_dist_index = i;
    }
    cout << "Max minimum distance of " << mindist[max_dist_index] << " reached by gram " << bitset<64>(global_outcome[max_dist_index]) << endl;

    outputfile << "Max minimum distance of " << mindist[max_dist_index] << " reached by gram " << bitset<64>(global_outcome[max_dist_index]) << endl;

    outputfile << endl << endl << flush;

    outputfile.close();

    if(mindist[max_dist_index] >= MIN_DIST){
        // create a file containing all the far Qgrams
        ofstream goodgrams;
        goodgrams.open("../exp_results/Q=" + to_string(Q) + "/" + to_string(Q) + "gramsDist" + to_string(MIN_DIST), ios::binary | ios::out | ios::app );

        for(uint64_t i = 0; i < global_outcome.size(); i++){
            if(mindist[i] == mindist[max_dist_index])
                goodgrams.write(reinterpret_cast<char *>(&(global_outcome[i])), sizeof(uint64_t)); 
        }
        goodgrams.close();
    }
    
    return;
}




int main()
{
    uint64_t g[N_hash_fctns];

    fstream checkfile;
    checkfile.open("../data/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    if(!checkfile.is_open())
    {
        throw logic_error("Preprocessing not done correctly!");
        return 0;
    }
    checkfile.close();


    clock_t begin = clock();
    build_functions(g);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Functions found in " << elapsed_secs << " seconds are: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i << ": " << bitset<64>(g[i]) << endl;
    cout << endl; 

    // Qmask has the first 2(maxQ-Q) bits set to 0, the last 2Q set to 1
    uint64_t Qmask = 0b11;
    for(int i=0; i < Q-1; i++)
    {
        Qmask <<=2;
        Qmask |= 0b11;
    }
    cout << "Mask Qmask is " << bitset<64>(Qmask) << endl << flush;

    begin = clock();
    process_multiple_masks(g); // ABOUT 20 MINS WITH 6 MASKS
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Masks processed in "<< elapsed_secs <<" time; complementary sets have sizes: " << flush; // Masks processed; complementary sets have sizes: |C0|= 38831      |C1|= 164169    |C2|= 86982   |C3|= 212690     |C4|= 114491    |C5|= 126248
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "|C" << i << "|= " << compl_array[i].size() << "\t " << flush;
    cout << endl<< flush; 


    begin = clock();
    compute_templates(g);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "End of template computation, which took " << elapsed_secs << " seconds. " << endl << flush;

    if(global_outcome.size() == 0)
        return 0;

    begin = clock();
    check(Qmask);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "End of check, which took " << elapsed_secs << " seconds. " << endl << flush;

    return 0;
}
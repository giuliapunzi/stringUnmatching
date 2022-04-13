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

// for mmap:
#include "../Q-grams/mapfile.hpp"


using namespace std;

constexpr auto Q = 32;
// constexpr int N_tests = 3; // number of tests
// constexpr int N_completions = 10; // number of completions for each template
constexpr int N_hash_fctns = 6;  // number of hash functions 
constexpr int target_size = 11; // target space size of hash functions
// constexpr int MAX_complement_size = 150000;
vector<uint64_t> compl_array[N_hash_fctns]; // array of vectors for complementary sets
vector<uint64_t> global_outcome;

constexpr int SEED = 13; // seed for random functions is fixed for trial reproducibility; may be changed as needed

// constexpr int MIN_DIST = 9;

constexpr auto MASK_WEIGHT = 22;  // number of 1s, twice the number of selected chars (as the alphabet is 4)
constexpr auto UNIVERSE_SIZE = 4194304; // this is 4^target_size 

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



// given two uint64_t, compute their Hamming distance
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}



void process_multiple_masks(uint64_t* mask_array, char* input_file_name){
    // TODO we can later remove this loop
    for(int maskindex = 0; maskindex < N_hash_fctns; maskindex++){
        uint64_t mask = mask_array[maskindex]; // current mask

        assert( __builtin_popcountll(mask) == MASK_WEIGHT );

        // initialize bit vector
        universe_bitvector.reset();

        size_t textlen = 0;   
        // scan all files and sets the bits corresponding to each masked q-gram
        const char * text = map_file(input_file_name, textlen); // map_file("../data/Blood/Blood.nuc.fsa", textlen);      

        uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
        auto key_len = 0;
        uint64_t i = 0;  // bug if we use auto :(
        auto skip = false;

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
                universe_bitvector[ qgram_to_index(key, mask)] = true;                
            }
            key <<= 2;            // shift two bits to the left
            
        }

        unmap_file(text, textlen);

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
            // cout << "Found a template \t" << flush;
        }
    }
    else{
        for(vector<uint64_t>::iterator it = interval.first; it != interval.second; it++ ){ //compl_array[function_index].begin() + interval.first; it!= compl_array[function_index].begin() + interval.second +1; it++){
            assert(((*it) & redmasks[function_index])==(candidate & redmasks[function_index]));
            rec_compute_templates(((*it) & g[function_index]) | candidate, function_index+1, g, redmasks);
        }
            
    }
}


void compute_templates(const uint64_t *g, char* output_log, char* output_templates_file){
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

    // File dump of results
    ofstream outputfile, binaryout; 
    outputfile.open(output_log, ios::app); //("../exp_results/Blood/" +to_string(N_hash_fctns) + "Funct" + to_string(target_size) + "TargetSeed" + to_string(SEED), ios::app);
    binaryout.open(output_templates_file, ios::binary | ios::app); //("../exp_results/Blood/" +to_string(N_hash_fctns) + "FunctBinary" + to_string(target_size) +"TargetSeed"  + to_string(SEED), ios::binary | ios::app);
    outputfile << "Test with " << N_hash_fctns << " functions: " << endl;
    outputfile << "Functions g: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        outputfile << "g_" << i << ": " << bitset<64>(g[i]) << endl;
    outputfile << endl << endl;

    outputfile << "Templates to check are " << global_outcome.size() << ": " << endl;
    cout << "Templates found are " << global_outcome.size() <<  endl << flush;

    for(uint64_t i = 0; i < global_outcome.size(); i++){
        uint64_t templ = global_outcome[i];
        binaryout.write(reinterpret_cast<char *>(&templ), sizeof(uint64_t)); 
        outputfile << bitset<64>(templ) << ", "; //print_Q_gram(templ);
    }
    outputfile << endl << endl;

    binaryout.close();
    outputfile.close();
}



void build_functions(uint64_t* g){
    srand(SEED);

    bool all_covered = false; 
    while( !all_covered ){
        // Create random functions
        for(int i=0; i<N_hash_fctns; i++){
            int pos[target_size];
            pos[0] = rand() % 32;
            int filled = 1;
            bool new_position = true;
            while(filled < target_size){
                int candidate_pos = rand() % 32;
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
                // cout << "g1 is " << bitset<64>(g1) << endl << flush;
                currg <<= (2*pos[j] - 2*pos[j-1]);
                currg |= 0b11;
            }

            currg <<= (2*Q - 2 - 2*pos[target_size-1]);
            g[i] = currg;

            assert( __builtin_popcountll(g[i]) == MASK_WEIGHT );
        }

        // cout << "Functions g: " << endl << flush;
        // for(int i = 0; i< N_hash_fctns; i++)
        //     cout << "g" << i+1 << ": " << bitset<64>(g[i]) << endl;
        // cout << endl; 

        // compute the complementary to check if all positions are covered
        uint64_t cg = g[0];
        for(int i=1; i<N_hash_fctns; i++)
            cg |= g[i];
        
        // cout << "Complementary is " << bitset<64>(~cg) << endl;

        if( ~cg == 0)
            all_covered = true;
    }
}


void template_generation(char* input_filename, char* output_log_filename, char* output_templates_filename)
{
    uint64_t g[N_hash_fctns];
    clock_t begin = clock();
    build_functions(g);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Functions found in " << elapsed_secs << " seconds are: " << endl << flush;
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "g" << i << ": " << bitset<64>(g[i]) << endl;
    cout << endl; 


    begin = clock();
    process_multiple_masks(g, input_file_name); // ABOUT 20 MINS WITH 6 MASKS
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Masks processed in "<< elapsed_secs <<" time; complementary sets have sizes: " << flush; // Masks processed; complementary sets have sizes: |C0|= 38831      |C1|= 164169    |C2|= 86982   |C3|= 212690     |C4|= 114491    |C5|= 126248
    for(int i = 0; i< N_hash_fctns; i++)
        cout << "|C" << i << "|= " << compl_array[i].size() << "\t " << flush;
    cout << endl<< flush; 

    sort_according_to_masks(g); // couple of minutes

    cout << "Masks have been sorted" << endl << flush;

    begin = clock();
    compute_templates(g, output_log_filename, output_templates_filename);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "End of template computation, which took " << elapsed_secs << " seconds. " << endl << flush;

    if(global_outcome.size() == 0)
        return;

    return;
}



int main(){
    char* input_filename = "input.fsa"; // NAME OF THE FASTA INPUT FILE
    char* output_log_filename = "output_log"; // NAME OF THE OUTPUT LOG FILE WHERE VARIOUS INFO IS APPENDED
    char* output_templates_filename = "templates.bin"; // NAME OF THE OUTPUT BINARY FILE CONTAINING ALL TEMPLATES FOUND

    template_generation(input_filename, output_log_filename, output_templates_filename);
    
    return 0;
}

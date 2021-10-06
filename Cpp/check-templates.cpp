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

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
// #include "../script/class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int MIN_DIST = 9;

// given two uint64_t, compute their Hamming distance
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}


void check (uint64_t* templates, int8_t* mindist, int64_t length)
{
    ifstream inputQgrams;
    inputQgrams.open("../data/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    uint64_t gram;
    int64_t N_deleted = 0;
    
    while (N_deleted < length){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;

        // compute distance for each Qgram of the file 
        for(uint64_t j=0; j<length; j++){
            if(mindist[j] >= 0){ // template has not been deleted yet 
                int dist = Hamming_distance(gram, templates[j]);

                if(dist < mindist[j]){
                    if(dist >= MIN_DIST){
                        mindist[j] = dist;
                    }
                    else{
                        mindist[j] = -mindist[j];
                        N_deleted++;
                        cout << "Size of templates: " << length - N_deleted << endl << flush;
                    }
                }

            }
                
        }
        
    }
    inputQgrams.close();


    ofstream outputfile; 
    outputfile.open("../exp_results/blocks/BlockEnumerationCheck1", ios::app);

    cout << endl << "Printing min distances for the " << length << " templates: " << flush;
    for(uint64_t dd = 0; dd< length; dd++){
        if(mindist[dd] > 0 )
            cout << mindist[dd] << " " << flush;
    }

    outputfile << endl << "Printing min distances for the " << length << " templates: " << flush;
    for(uint64_t dd = 0; dd< length; dd++){
        if(mindist[dd] > 0 )
            outputfile << mindist[dd] << " " << flush;
    }
    outputfile << endl;

    uint64_t max_dist_index = 0;
    for(uint64_t i=0; i<length; i++){
        if(mindist[i] > mindist[max_dist_index])
            max_dist_index = i;
    }

    outputfile << "Max minimum distance of " << mindist[max_dist_index] << " reached by gram " << bitset<64>(templates[max_dist_index]) << endl;
    outputfile << endl << endl << flush;

    outputfile.close();

    if(mindist[max_dist_index] >= MIN_DIST){
        // create a file containing all the far Qgrams
        ofstream goodgrams;
        goodgrams.open("../exp_results/blocks/QgramsDist" + to_string(MIN_DIST), ios::binary | ios::out | ios::app );

        for(uint64_t i = 0; i < length; i++){
            if(mindist[i] >0)
                goodgrams.write(reinterpret_cast<char *>(&(templates[i])), sizeof(uint64_t)); 
        }
        goodgrams.close();
    }
    
    return;
}


int main(){
    ifstream binaryin;
    binaryin.open("../exp_results/BinaryTest", ios::binary | ios::in);
    binaryin.seekg (0, binaryin.end);
    int64_t length = binaryin.tellg()/sizeof(uint64_t);
    cout << "Length: " << length << endl << flush;

    binaryin.seekg (0, binaryin.beg);

    uint64_t templates[length];
    int8_t mindist[length] = {Q+1}; // negative for deleted elements

    uint64_t gram;
    uint64_t counter = 0;
    int l = 0;
    while (l< length){
        binaryin.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!binaryin) break;

        template[l++] = gram;
    }
    length = l;

    clock_t begin = clock();
    check(templates, mindist, length);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Check 1 performed in " << elapsed_secs << " seconds" << endl << flush;

    binaryin.close();

    return 0;
}

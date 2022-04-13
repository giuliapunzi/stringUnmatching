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
#include <random>
# include <chrono>

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
// #include "../script/class_partitions_6545.h" // "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;
constexpr int MIN_DIST = 9;
constexpr int64_t length = 600015;

// given two uint64_t, compute their Hamming distance
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}




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




void rand_check (uint64_t* templates, uint8_t* mindist)
{
    ifstream inputQgrams;
    inputQgrams.open("../data/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::in);
    uint64_t gram;
    // int64_t N_deleted = 0;
    int64_t counter = 0;
    int64_t total_count = 0;
    
    while(true){ // (N_deleted < length){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;

        // compute distance for each Qgram of the file 
        // #pragma omp parallel for 
        for(int64_t j=0; j<length; j++){
            // if(mindist[j] >= 0){ // template has not been deleted yet 
            int dist = Hamming_distance(gram, templates[j]);

            if(dist < mindist[j]){
                // if(dist >= MIN_DIST){
                mindist[j] = dist;
                // }
                // add pragma critical
                // else{
                //     mindist[j] = -mindist[j];
                //     // #pragma omp critical
                //     // {
                //     ++N_deleted;
                //     cout << "Size of templates: " << length - N_deleted << endl << flush;
                //     // }
                // }
            }
                
        }

        counter++;
        total_count++;

        // every 100 million, print some output
        if(counter == 10000000){
            cout << "Parsed " << total_count << endl << flush;
            counter = 0;
        }
        
    }
    inputQgrams.close();


    ofstream outputfile; 
    outputfile.open("../exp_results/blocks/RandomCheck", ios::app);

    cout << endl << "Printing min distances for the " << length << " templates: " << flush;
    for(int64_t dd = 0; dd< length; dd++){
        if(mindist[dd] > 0 )
            cout << (int)(mindist[dd]) << " " << flush;
    }
    cout << endl << endl << flush;

    outputfile << endl << "Printing min distances for the " << length << " templates: " << flush;
    for(int64_t dd = 0; dd< length; dd++){
        if(mindist[dd] > 0 )
            outputfile << (int)(mindist[dd]) << " " << flush;
    }
    outputfile << endl;

    uint64_t max_dist_index = 0;
    for(int64_t i=0; i<length; i++){
        if(mindist[i] > mindist[max_dist_index])
            max_dist_index = i;
    }
    
    cout << "Max minimum distance of " << (int)(mindist[max_dist_index]) << " reached by gram " << bitset<64>(templates[max_dist_index]) << endl << endl;
    
    outputfile << "Max minimum distance of " << (int)(mindist[max_dist_index]) << " reached by gram " << bitset<64>(templates[max_dist_index]) << endl;
    outputfile << endl << endl << flush;

    outputfile.close();

    // if(mindist[max_dist_index] >= MIN_DIST){
    //     // create a file containing all the far Qgrams
    //     ofstream goodgrams;
    //     goodgrams.open("../exp_results/blocks/QgramsDist" + to_string(MIN_DIST), ios::binary | ios::out | ios::app );

    //     for(int64_t i = 0; i < length; i++){
    //         if(mindist[i] >0)
    //             goodgrams.write(reinterpret_cast<char *>(&(templates[i])), sizeof(uint64_t)); 
    //     }
    //     goodgrams.close();
    // }
    
    return;
}


int main(){
    uint64_t templates[length];
    uint8_t mindist[length]; // NOT WORKING = {Q+1}; // negative for deleted elements

    for (int64_t j = 0; j < length; j++)
    {
        mindist[j] = Q+1;

        // if(j % 100 == 0)
        //     cout << "Mindist for j=" << j << " is " << (int)(mindist[j]) << endl << flush;

        // create random string to fill templates[i]
        std::random_device rd;
        std::mt19937::result_type seed = rd() ^ (
        (std::mt19937::result_type) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() +
        (std::mt19937::result_type) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        );
        std::mt19937 gen(seed);

        uint64_t rand_gram = 0;
        for(int i = 0; i < Q; i++, rand_gram <<= 2)
        {
            std::uniform_int_distribution<unsigned> distrib(0, 3); // 3 included
            int rand_char = distrib(gen); //rand()%4;
            switch (rand_char)
            {
            case 0:
                break;
            case 1:
                rand_gram |= 0b01;
                break;
            case 2:
                rand_gram |= 0b10;
                break;
            case 3:
                rand_gram |= 0b11;
                break;
            default:
                break;
            }
        }
        
        templates[j] = rand_gram;
        if(j%25000 == 0)
        {
            cout << "Random string: " << flush;
            print_Q_gram(rand_gram);
        }

    }

    cout << endl;

    // cout << "Printing mindistances before starting: " << endl << flush;
    // for(int64_t dd = 0; dd< length; dd++){
    //     if(mindist[dd] > 0 )
    //         cout << (int)(mindist[dd]) << " " << flush;
    // }
    // cout << endl << endl;
    
    ofstream binaryout;
    binaryout.open("../exp_results/Blood/random32grams", ios::binary);
    for(int64_t i=0; i< length; i++){
        binaryout.write(reinterpret_cast<char *>(&templates[i]), sizeof(uint64_t));
    }

    binaryout.close();

    clock_t begin = clock();
    // rand_check(templates, mindist);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // cout << "Random check performed in " << elapsed_secs << " seconds" << endl << flush;

    return 0;
}


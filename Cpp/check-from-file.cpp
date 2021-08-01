// compile with option -mbmi2 -O3 -std=c++2a
#include <iostream>
#include <fstream>
#include <bit> // for popcount
#include <bitset> // to print in binary form 
#include <stdlib.h>
#include <vector>
#include <immintrin.h> // to use pdep, pext

// for Parikh classes Parikh_class_partition[N_CLASSES] where N_CLASSES = 6545 = (35 choose 3)
#include "../script/class_partitions_6545.h"

using namespace std;

constexpr auto Q = 32;

vector<uint64_t> candidates;
// vector<uint64_t> mindist;

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
 // filename is the name of the file containing the Qgrams
// void check (uint8_t * text, uint64_t textlen, int* mindist)  // DEBUGGED
void check_from_file (string filename)
{
    ifstream inputfile;
    inputfile.open(filename, ios::binary | ios::in);
    uint64_t curr_template;
    while (true){
        inputfile.read(reinterpret_cast<char *>(&curr_template), sizeof(uint64_t)); 
        if (!inputfile) break;
        // add current object to vector
        candidates.push_back(curr_template);
    
    }
    inputfile.close();

    uint64_t templ_size = candidates.size();
    cout << "Candidates are " << templ_size << ": " << endl << flush;
    for(uint64_t i = 0; i < candidates.size(); i++){
        uint64_t templ = candidates[i];
        cout << bitset<64>(templ) << ", "; //print_Q_gram(templ);
    }
    cout << endl << flush;

    // initialize distances' vector
    // int mindist[templ_size];
    vector<int> mindist;
    for(uint64_t templindex = 0; templindex < templ_size; templindex++)
        mindist.push_back(Q+1);
        // mindist[templindex] = Q+1; //Hamming_distance(completions[templindex], key); 

    cout << endl << "Printing min distances before starting: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        cout << mindist[dd] << " " << flush;

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
            // compute distance for each Qgram of the file 
            for(uint64_t j=0; j<templ_size; j++){
                int dist = Hamming_distance(gram, candidates[j]);

                if(dist < mindist[j])
                    mindist[j] = dist;
            }
        }
        fin.close();
        cout << "*" << flush;
    }

    cout << endl << "Printing min distances for the " << templ_size << " templates: " << flush;
    for(uint64_t dd = 0; dd< templ_size; dd++)
        cout << mindist[dd] << " " << flush;


    return;
}



int main()
{
    check_from_file("../exp_results/TestMultFunctTemplates");

    return 0;
}
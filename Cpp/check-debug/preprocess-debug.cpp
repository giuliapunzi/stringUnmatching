#include <iostream>
#include <fstream>
#include <vector>
#include "mapfile.hpp"

using namespace std;

/* preprocess the text and store it in a binary file of uint64, where the 2Q least significative bits correspond to the Qgrams */

constexpr auto Q = 32;   
constexpr auto maxQ = 32;

void extract_Q_grams(){
    // map file
    size_t textlen = 0;   
    const char * text = map_file("../data/Blood/Blood.nuc.fsa", textlen); 

    ofstream fout;
    fout.open("../data/Blood/all" + to_string(Q) + "grams_repetitions", ios::binary | ios::out);
     

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;
    uint64_t count_situation = 0; // same as for i?
    uint64_t total_count = 0;

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
            total_count++;

        }
        key <<= 2;            // shift two bits to the left

        if(count_situation == 10000000){
            cout << "*" << flush;
            count_situation = 0;
        } 
    }

    fout.close();
    unmap_file(text, textlen);

    cout << "Total count: " << total_count << endl;
}



int main(int argc, char *argv[]){
    extract_Q_grams();

    return 0;
}
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

#include "mapfile.hpp"

using namespace std;

constexpr auto Q = 32;
int MIN_DIST;

// given two uint64_t, compute their Hamming distance
__attribute__((always_inline)) int Hamming_distance(uint64_t x, uint64_t y) // DEBUGGED
{
    uint64_t diff = ~(x^y);
    diff &= (diff << 1);
    diff &= 0xAAAAAAAAAAAAAAAA;

    return Q - popcount(diff); // I counted where they are equal, subtract it from Q to find the difference
}


/* preprocess the text (input_file_name) and store it in a binary file (Qgrams_filename) of uint64 */
void extract_Q_grams(const char* input_filename, const char* Qgrams_filename){
    // map file
    size_t textlen = 0;   
    const char * text = map_file(input_filename, textlen);  

    ofstream fout;
    fout.open(Qgrams_filename, ios::binary | ios::out); 
     

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




void check (uint64_t* templates, uint8_t* mindist, int64_t length, const char* Qgrams_filename, const char* output_log_filename, const char* good_templates_filename)
{
    ifstream inputQgrams;
    inputQgrams.open(Qgrams_filename, ios::binary | ios::in);
    uint64_t gram;
    int64_t N_deleted = 0;
    int64_t counter = 0;
    int64_t total_count = 0;
    
    while (N_deleted < length){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;

        // compute distance for each Qgram of the file 
        // #pragma omp parallel for 
        for(int64_t j=0; j<length; j++){
            if(mindist[j] > 0){ // template has not been deleted yet 
                int dist = Hamming_distance(gram, templates[j]);

                if(dist < mindist[j]){
                    if(dist >= MIN_DIST){
                        mindist[j] = dist;
                    }
                    // add pragma critical
                    else{
                        mindist[j] = -mindist[j];
                        // #pragma omp critical
                        // {
                        ++N_deleted;
                        cout << "Size of templates: " << length - N_deleted << endl << flush;
                        // }
                    }
                }

            }

                
        }

        counter++;
        total_count++;

        // every 100 million, print some output
        if(counter == 100000000){
            cout << "Parsed " << total_count << endl << flush;
            counter = 0;
        }
        
    }
    inputQgrams.close();


    ofstream outputfile; 
    outputfile.open(output_log_filename, ios::app);

    cout << endl << "Printing min distances for the " << length << " templates: " << flush;
    for(int64_t dd = 0; dd< length; dd++){
        if(mindist[dd] > 0 )
            cout << (int)(mindist[dd]) << " " << flush;
    }

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

    outputfile << "Max minimum distance of " << (int)(mindist[max_dist_index]) << " reached by gram " << bitset<64>(templates[max_dist_index]) << endl;
    outputfile << endl << endl << flush;

    outputfile.close();

    if(mindist[max_dist_index] >= MIN_DIST){
        // create a file containing all the far Qgrams
        ofstream goodgrams;
        goodgrams.open(good_templates_filename, ios::binary | ios::out | ios::app );

        for(int64_t i = 0; i < length; i++){
            if(mindist[i] >0)
                goodgrams.write(reinterpret_cast<char *>(&(templates[i])), sizeof(uint64_t)); 
        }
        goodgrams.close();
    }
    
    return;
}


int map_and_check(int minimum_dist, const char* input_filename, const char* Qgrams_filename, const char* templates_filename, const char* output_log_filename, const char* good_templates_filename){
    MIN_DIST = minimum_dist;

    ifstream binaryin;
    binaryin.open(templates_filename, ios::binary | ios::in);
    binaryin.seekg (0, binaryin.end);
    int64_t length = binaryin.tellg()/sizeof(uint64_t);
    cout << "Length: " << length << endl << flush;

    binaryin.seekg (0, binaryin.beg);


    uint64_t* templates = new uint64_t[length];
    uint8_t* mindist = new uint8_t[length];

    for (int64_t i = 0; i < length; i++)
    {
        mindist[i] = Q+1;
    }
    

    uint64_t gram;
    int l = 0;
    while (l< length){
        binaryin.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!binaryin) break;

        templates[l++] = gram;
    }


    // before the check, we need to map the input
    extract_Q_grams(input_filename, Qgrams_filename);


    clock_t begin = clock();
    check(templates, mindist, length, Qgrams_filename, output_log_filename, good_templates_filename);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Check performed in " << elapsed_secs << " seconds" << endl << flush;

    binaryin.close();


    delete[] templates;
    delete[] mindist;

    return 0;
}


int main(){
    int minimum_distance = 9;
    const char* input_filename = "input.fsa"; // NAME OF THE FASTA FILE CONTAINING THE INPUT DATASET 
    const char* Qgrams_filename = "Qgrams.bin"; // NAME OF THE BINARY FILE THAT WILL BE FILLED WITH THE QGRAMS OF THE INPUT FILE
    const char* templates_filename = "templates.bin"; // NAME OF THE BINARY FILE CONTAINING ALL TEMPLATES TO BE CHECKED 
    const char* output_log_filename = "output_log"; // NAME OF OUTPUT TXT LOG FILE THAT WILL BE APPENDED WITH VARIOUS INFORMATION ABOUT THE TRIAL
    const char* good_templates_filename = "good_templates.bin"; // NAME OF OUTPUT BINARY FILE THAT WILL BE FILLED WITH ALL INPUT TEMPLATES THAT ARE AT DISTANCE AT LEAST MIN_DIST FROM THE WHOLE INPUT  
    
    map_and_check(minimum_distance, input_filename, Qgrams_filename, templates_filename, output_log_filename, good_templates_filename);

    return 0;
}
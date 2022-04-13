#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// for mmap:
#include "mapfile.hpp"

using namespace std;

void find_no_genomes(){
    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file("../data/Gastrointestinal_tract/Gastrointestinal_tract.nuc.fsa", textlen);      
    uint64_t count = 0;

    for (uint64_t i = 0; i < textlen; i++)
    {
        if(text[i] == '>')
            count++;
    }
    
    unmap_file(text, textlen);

    cout << "Total count of genomes is " << count << endl;
}


int main(int argc, char *argv[]){
    // if (argc != 2){
    //     cout << "Usage: " + string(argv[0]) + " <FASTA_filename>" << endl << flush;
    //     exit(255);
    // }

    find_no_genomes();

    return 0;
}
// RG-July2021: compile with option -std=c++20

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <map>
#include <unordered_map>

// for mmap:
#include <sys/mman.h>
#include <sys/stat.h>

using namespace std;

/* Q-GRAMS */

constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
std::unordered_set<uint64_t> Q_hash;

// uint8_t char_counter[4] __attribute__ ((aligned (4)));  // invariant: char_counter[i] <= Q < 256, and sum_ i char_counter[i] = Q. char_counter[] is seen as uint32_t
// unordered_map<uint32_t, uint64_t> Parikh_classes;

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

void check_Q_gram(uint64_t gram){
    if (!Q_hash.contains( gram )){
        Q_hash.insert( gram );
        // Parikh_classes[*reinterpret_cast<uint32_t *>(char_counter)]++;    
    }
}


const char* map_file(const char* fname, size_t& length);
void unmap_file(const char* addr, size_t length);

void find_Q_grams(const char * filename){
    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file(filename, textlen);      

    // for (auto i =0; i < 4; i++) char_counter[i] = 0;
    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;
    uint64_t counter = 0;

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
            check_Q_gram( key );  // & mask when Q < 32
            // char_counter[(key >> (Q-1+Q-1)) & 0x3]--;  // exiting char
            counter++;
        }
        key <<= 2;            // shift two bits to the left

        if(counter == 1000000)
            cout << "Processed 1 million" << endl << flush;
    }
    unmap_file(text, textlen);
}

/* MMAP */

void handle_error(const char* msg) {
    perror(msg); 
    exit(255);
}

FILE *fd;

const char* map_file(const char* fname, size_t& length)
{
    fd = fopen(fname, "r");
    if (fd == NULL)
        handle_error("open");

    // obtain file size
    struct stat sb;
    if (fstat(fileno(fd), &sb) == -1)
        handle_error("fstat");

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_SHARED, fileno(fd), 0u));
    if (addr == MAP_FAILED)
        handle_error("mmap");

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}

void unmap_file(const char* addr, size_t length)
{
    munmap(const_cast<char *>(addr), length);
    fclose(fd);
}


int main(int argc, char *argv[]){
    if (argc != 2){
        cout << "Usage: " + string(argv[0]) + " <FASTA_filename>" << endl << flush;
        exit(255);
    }

    find_Q_grams(argv[1]);

    cout << "Found " << Q_hash.size() << " " << Q << "grams" << endl << flush; 
    
    ofstream fout;
    fout.open("../data/Blood/distinct" + to_string(Q) + "grams", ios::binary | ios::out);
    

    // print distinct chars to output
    // for (const auto gram: Q_hash) {    
    for (auto it = Q_hash.begin(); it != Q_hash.end(); ++it) {
        uint64_t gram = *it;
        fout.write(reinterpret_cast<char *>(&gram), sizeof(uint64_t));
    }

    fout.close();

    // for (auto e : Parikh_classes) {
    //     uint32_t temp = e.first;
    //     uint8_t * tempa = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&temp));
    //     for (auto i = 0; i < 4; i++)
    //         cout << static_cast<uint64_t>(tempa[i]) << ' ';
    //     cout << "\t : " << e.second << "\n";
    // }

    // for (auto e : Parikh_classes) {
    //     cout << e.second << "\n";
    // }

    return 0;
}
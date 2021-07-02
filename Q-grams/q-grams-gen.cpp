// RG-July2021: compile with option -std=c++20

#include <iostream>
#include <unordered_set>

// for mmap:
#include <sys/mman.h>
#include <sys/stat.h>

using namespace std;


/* Q-GRAMS */

constexpr auto Q = 32;    // max val is 32 as we pack a Q-gram into a 64-bit word uint64_t
                          // 2-bit encoding: A = 00, C = 01, G = 10, T = 11 
std::unordered_set<uint64_t> Q_hash;


void print_Q_gram(uint64_t gram){
    char s[Q+1];
    s[Q] = '\0';
    for (auto i=Q-1; i >= 0; i--, gram >>= 2){
        switch (gram & 0x11)
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
    if (!Q_hash.contains( gram ))
        Q_hash.insert( gram );        
}


const char* map_file(const char* fname, size_t& length);
void unmap_file(const char* addr, size_t length);

void find_Q_grams(const char * filename){
    // open filename as an array text of textlen characters
    size_t textlen = 0;   
    const char * text = map_file(filename, textlen);      

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    auto i = 0;
    auto skip = false;

    while(i < textlen){
        switch (toupper(text[i]))
        {
        case 'A':
            key |= 0x00;
            break;
        case 'C':
            key |= 0x01;
            break;
        case 'G':
            key |= 0x10;
            break;
        case 'T':
            key |= 0x11;
            break;
        case '\n':
            skip = true;
            break;
        case '>':
        case ';':
            while( i < textlen && text[i] != '\n') i++;
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
        if (++key_len == Q){
            key_len = Q-1;        // for the next iteration
            check_Q_gram( key );
            key <<= 2;            // shift two bits to the left
        }
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

    for (const auto& gram: Q_hash) {
        print_Q_gram(gram);
    }

    return 0;
}
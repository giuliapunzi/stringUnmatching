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


using namespace std;

constexpr auto Q = 32;
constexpr int MIN_DIST = 2;

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



void extract_Q_grams(){
    ifstream fin;
    fin.open("debug_text.txt", ios::in);

    ofstream fout;
    fout.open("binary_debug_text", ios::binary | ios::out);
     

    uint64_t key = 0;  // 32 chars from { A, C, G, T } packed as a 64-bit unsigned integer
    auto key_len = 0;
    uint64_t i = 0;  // bug if we use auto :(
    auto skip = false;
    uint64_t total_count = 0;

    while (true){
            char c;
            fin >> c;
            if (!fin) break;
            
            switch (toupper(c))
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
                while(!fin && c != '\n') i++;
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
                
                cout << key << '\t';

                // push out to output
                fout.write(reinterpret_cast<char *>(& key), sizeof(uint64_t)); 
                total_count++;
            }
            key <<= 2;            // shift two bits to the left

    }

    fin.close();
    fout.close();

    cout << endl << "Total count: " << total_count << endl;
}




void check (uint64_t templates_A[], uint64_t templates_B[], int8_t mindist_A[], int8_t mindist_B[], int64_t length)
{
    uint64_t* p;
    uint64_t* q; 
    p = templates_A;
    q = templates_B;
    
    int8_t* distp;
    int8_t* distq;
    distp = mindist_A;
    distq = mindist_B;

    cout << endl << "Printing starting distances for the " << length << " templates: " << flush;
    for(uint64_t dd = 0; dd< length; dd++){
        cout << (int) distp[dd] << " " << flush;
    }

    ifstream inputQgrams;
    inputQgrams.open("binary_debug_text", ios::binary | ios::in);
    uint64_t gram;
    int64_t lenp = length;
    int64_t lenq = 0;
    
    while (lenp > 0){
        inputQgrams.read(reinterpret_cast<char *>(&gram), sizeof(uint64_t)); 
        if (!inputQgrams) break;
        // cout << "Considering template " << bitset<64>(p[j]) << endl;
        
        // compute distance for each Qgram of the file 
        for(uint64_t j=0; j<lenp; j++){
            int dist = Hamming_distance(gram, p[j]);

            
            if(dist < distp[j]){
                if(dist >= MIN_DIST){
                    distq[lenq] = dist;
                    q[lenq++] = p[j];
                }
                else{
                    cout << "Templates left: " << lenp - (j + 1 - lenq) << endl << flush;
                }
            }
            else{
                distq[lenq] = distp[j];
                q[lenq++] = p[j];
            }
        }

        uint64_t* temp = p;
        p = q;
        q = temp;

        int8_t* tempd = distp;
        distp = distq;
        distq = tempd;

        lenp = lenq;
        lenq=0;
    }
    inputQgrams.close();


    cout << endl << "Printing min distances for the " << lenp << " templates: " << flush;
    for(uint64_t dd = 0; dd< lenp; dd++){
        cout << (int) distp[dd] << " " << flush;
    }
    cout <<endl;

    uint64_t max_dist_index = 0;
    for(uint64_t i=0; i<lenp; i++){
        if(distp[i] > distp[max_dist_index])
            max_dist_index = i;
    }

    cout << "Max minimum distance of " << (int) distp[max_dist_index] << " reached by gram " << bitset<64>(p[max_dist_index]) << endl;
    cout << endl << endl << flush;
    
    return;
}


int main(){
    int64_t length = 5;
    cout << "Length: " << length << endl << flush;

    uint64_t templates_A[length];
    uint64_t templates_B[length];
    int8_t mindist_A[length]; // negative for deleted elements
    int8_t mindist_B[length];

    uint64_t gram= 0b00000000000000000000000000000000;
    uint64_t counter = 0;
    int l = 0;
    while (l< length){
        templates_A[l] = gram;
        mindist_A[l++] = 33;
        
        // consider gram shifted by four, with a T added
        gram <<=4;
        gram |= 0b11;
    }
    length = l;

    cout << "Templates are: " << endl;
    for (int i = 0; i < length; i++)
    {
        print_Q_gram(templates_A[i]);
    }
    cout << endl;
     
    // extract_Q_grams();

    cout << "Now starting check: " << endl;

    clock_t begin = clock();
    check(templates_A, templates_B, mindist_A, mindist_B, length);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Check 2 performed in " << elapsed_secs << " seconds" << endl << flush;

    return 0;
}

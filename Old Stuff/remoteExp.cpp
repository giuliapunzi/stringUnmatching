#include <iostream> 
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <set>
#include <algorithm> 
#include <cmath>
#include <time.h>
#include <stdexcept>

using namespace std;

// alphabet constant vector
const vector<char> alph = {'A', 'C', 'G', 'T'};


// random projection function which, given an input integer L representing the size of the origin space,
// and an input integer m representing the size of the target space, it outputs a vector of m integers
// extracted randomly from [0,L-1], each representing a projection onto one coordinate. 
// NOTE: if the m integers have repetitions, they are removed.
// Furthermore, output vector is sorted.
vector<int> randomProjections(int L, int m)
{
    int val;
    vector<int> g = {};
    vector<int>::iterator it;

    for(int i = 0; i<m; i++)
    {
        val = rand()%L;
        it = find(g.begin(), g.end(), val);
        
        // add element only if not present already
        if(it == g.end())
            g.push_back(val);
    } 

    sort(g.begin(), g.end());

    return g;
}


// hamming distance between two strings
int hdist(string x, string y)
{
    if(x.size() != y.size())
        return -1;

    int dist = 0;

    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] != y[i])
            dist++;
    }
    
    return dist;
}


// generates a random string of given length L
string randomString(int L)
{
    string s = "";
    for (int i = 0; i < L; i++)
        s.push_back(alph[rand() % alph.size()]);
    
    return s;
}



// check takes a string query, and int distance r and the input set and checks whether 
// the query is at distance at least r from the whole input.
bool check(string q, int r, vector<int> &input, string W)
{
    // for every input string, if the query is close to it then return false
    for(int i = 0; i< input.size(); i++)
    {
        if(hdist(q, W.substr(input[i], q.size())) < r)
            return false;
    }
    
    return true;
}



// maps the input int vector through the projection proj
// outputs a vector of ints of string W denoting the positions where
// we have the distinct projections of the input vector's L-mers
vector<int> mapInput(vector<int> &input, vector<int> &proj, string &W)
{
    vector<int> mapped;
    // cout << "Mapping input..."<<endl;

    // use a SET for keeping track of which m-mers have already been mapped
    set<string> mMers;

    for (int i = 0; i < input.size(); i++)
    {
        // build the string we are currently checking by appending every
        // character with the right offset (given by proj) with respect to
        // the input position we are currently considering
        
        // if(i % 100000 == 0)
        //     cout << i << "\t";

        // cout <<endl<< "Current string position is " << input[i] << endl;

        string curr = "";

        for (int j  = 0; j < proj.size(); j++)
            curr.push_back(W[input[i]+proj[j]]);
            // curr = curr + W[input[i]+proj[j]];

        // cout << "Current substring for projection is " << curr << endl;

        
        // cout << "At this moment, inserted set is: ";
        // for (set<string>::iterator it = mMers.begin(); it !=  mMers.end(); it++)
        // {
        //     cout << "\t" << *it;
        // }
        // cout << endl;

        // check if current m-mer already mapped
        // if not, add the corresponding position to the mapped set
        pair<set<string>::iterator,bool> insertion;

        insertion = mMers.insert(curr);

        // cout << "After insertion, inserted set is: ";
        // for (set<string>::iterator it = mMers.begin(); it !=  mMers.end(); it++)
        // {
        //     cout << "\t" << *it;
        // }
        // cout << endl;
        

        if(insertion.second)
        {
            // cout << "INSERTION TOOK PLACE; pushing back position " << input[i] <<endl;
            mapped.push_back(input[i]);
        }
        
        // bool newmmer = true;
        // int h = 0;
        // while(newmmer && h<mapped.size())
        // {
        //     // build corresponding element in mapped
        //     string mappedEl = "";
        //     for (int j = 0; j < proj.size(); j++)
        //         mappedEl = mappedEl + W[mapped[h] + proj[j]];
            
        //     // cout << "Current mapped element that we are checking for equality is " << mappedEl << endl;
            
        //     if(curr == mappedEl)
        //         newmmer = false;

        //     h++;
        // }

        // if(newmmer)
        //     mapped.push_back(i);    
    } 

    cout << endl;

    mMers.clear();

    return mapped;  
}



void recEnumTophMultiple(int h, string prefix, vector<vector<int>> &stringsLeft, vector<vector<int>> &g, string &W, vector<string> &queries, vector<vector<array<int,4>>> &freq)
{
    // cout << "Inside recursion" << endl;
    if(queries.size() == h)
        return; 

    // if all g have been emptied (all their offsets have been considered), we are in the base case
    bool gempty = true;
    for (int i = 0; i < g.size(); i++)
    {
        if (g[i].size() > 0)
            gempty = false;
    }
    
    if(gempty)
    {   
        // cout << "String considered is " << prefix << endl;

        // if we generated a string of correct length which is different from all other strings (stringsLeft is empty), we are done
        bool stringempty = true;
        for (int i = 0; i < stringsLeft.size(); i++)
        {
            if (stringsLeft[i].size() > 0)
                stringempty = false;
        }

        if(stringempty)
        {
            // we have been building the string back-to-front, so we need to reverse it, and then push it into the queries vector
            reverse(prefix.begin(), prefix.end());
            queries.push_back(prefix);
            // cout << "Valid string found: " << prefix << endl;
        }

        return;
    }
    else
    {
        // positions are considered FROM THE BACK for ease of push/pop
        // NOTE: the size of some g could be zero, as we remove elements from them; we need to check for this

        // take the "next" = biggest position
        int current = 0;
        for (int i = 0; i < g.size(); i++)
        {
            // if the last element of the array is bigger than current, it becomes current
            if (g[i].size() > 0 && g[i].back() > current)
                current = g[i].back();
        }

        // now select all indices which realize current as last position
        vector<int> currentfunctions;
        for (int i = 0; i < g.size(); i++)
        {
            if (g[i].size() > 0 && g[i].back() == current)
                currentfunctions.push_back(i);
        }
        
        // NOTE: g[currentfunctions[i]] for all i are the functions where we have removed the current index

        array<int, 4> frequencies;
        vector<int> freqInd;


        // pop the current position we are considering from the currentfunctions
        for (int i = 0; i < currentfunctions.size(); i++)
            g[currentfunctions[i]].pop_back();
        
        // cout << "Next position is " << current << endl;

        // the frequencies are the sum of the frequencies of the corresponding currentfunctions frequency vectors
        // so, start the vector of frequencies as the first current function frequency vector for the last offset
        // (which corresponds to our current index), and then for every other current function sum the vectors
        for (int i = 0; i < frequencies.size(); i++)
            frequencies[i]= 0;
        
        //frequencies = freq[currentfunctions[0]].back();
        for (int i = 0; i < currentfunctions.size(); i++)
        {
            // for every character frequency, sum the value for the currentfunction's last offset character frequency
            for (int j = 0; j < frequencies.size(); j++)
                frequencies[j] += freq[currentfunctions[i]].back()[j];
        }
        
        // cout << "Considering frequencies vector ";
        // for (int i = 0; i < frequencies.size(); i++)
        // {
        //     cout << "\t" << frequencies[i];
        // }


        // order the indices such that frequencies is ordered (ascendingly) according to them
        freqInd = {0,1,2,3};
        sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );

        // cout << endl << "Sorted indices: " << endl;
        // for (int i = 0; i < freqInd.size(); i++)
        // {
        //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
        // }

        
        // this goes through all character in increasing frequency order (so, we start with the LEAST frequent), eliminating the most strings
        for (int i = 0; i < freqInd.size(); i++)
        {
            // current character is exactly alph[freqInd[i]]
            char c = alph[freqInd[i]];

            // vector removed is used to keep track of the positions that have been removed for every currentfunction
            // this is needed to reappend the positions before exiting the recursive call
            // this vector will have the same size as currentfunctions
            vector<vector<int>> removed;
            vector<vector<int>> posToRemove;

            // we need to update stringsLeft before recursing: for every currentfunction, we scan its stringsLeft and 
            // remove the indices corresponding to m-mers having a char different from c at the correct (last) position
            // recall: position (offset) is current, the same for all functions in currentfunctions
            for (int fctn = 0; fctn < currentfunctions.size(); fctn++)
            {
                // NOTE: currentfunctions[fctn] are the indices of the functions we are considering
                int fctnindex = currentfunctions[fctn];

                vector<int> currRemoved;  // currRemoved will be filled with the positions to be removed at index fctnindex
                vector<int> currpos;

                // this iterates over the stringsLeft for the function with index currentfunctions[fctn]
                for (int j = 0; j< stringsLeft[fctnindex].size(); j++)
                {
                    // if the character in the string at the current offset is different from c, add them to the removed array
                    // RECALL: stringsLeft are indexed just like the functions g, so we need to "apply" currentfunctions to the index
                    if(W[stringsLeft[fctnindex][j]+current] != c)
                    {
                        currpos.push_back(j); // try to directly push back the position! In this way we do not need to use a find
                        currRemoved.push_back(stringsLeft[fctnindex][j]);
                    }
                }
                
                // push back the finalized vector of positions to be removed for the corresponding current function
                removed.push_back(currRemoved);
                posToRemove.push_back(currpos);
            }
            

            // now, remove every element of removed from the corresponding stringsLeft
            // iterate over the elements of posToRemove, each corresponding to a currentfunction
            for (int fctn = 0; fctn < posToRemove.size(); fctn++)
            {
                // for every element to remove from the stringsLeft of currentfunctions[fctn]
                for (int j = 0; j < posToRemove[fctn].size(); j++)
                {
                    // find the position in stringsLeft with respect to the currentfunction we are considering, where the element
                    // to be removed occurs, and remove it == we know what position we have to remove. 
                    // vector<int>::iterator it = find(stringsLeft[currentfunctions[fctn]].begin(), stringsLeft[currentfunctions[fctn]].end(), removed[fctn][j]);
                    // NOTE: we need to subtract the index j of how many elements we have already removed, 
                    // as the vector stringsleft is getting shorter every loop!
                    stringsLeft[currentfunctions[fctn]].erase(stringsLeft[currentfunctions[fctn]].begin() + (posToRemove[fctn][j]-j));
                }                
            }

            // cout << "About to recurse by appending char " << c << endl;
            
            recEnumTophMultiple(h, prefix + c, stringsLeft, g, W, queries, freq);


            // now re-add stringsLeft indices for next iteration
            for (int fctn = 0; fctn < currentfunctions.size(); fctn++)
                stringsLeft[currentfunctions[fctn]].insert(stringsLeft[currentfunctions[fctn]].end(), removed[fctn].begin(), removed[fctn].end());
            
        } // this is the end of the loop over the alphabet chars 

        // re-add current to all functions g where it was removed before returning
        for (int i = 0; i < currentfunctions.size(); i++)
            g[currentfunctions[i]].push_back(current);
        
        return;
    }

}


vector<string> enumTophMultiple(int h, vector<vector<int>> &mappedPos, vector<vector<int>> &g, string &W)
{
    // cout << "Inside enumSmart" << endl;
    int k = g.size();


    // check if any of the sets are full
    for (int i = 0; i < mappedPos.size(); i++)
    {
        if(mappedPos[i].size() == pow(alph.size(),g[i].size()))
        {
            // cout << i+1 << "TH STRING SET IS FULL!" << endl;
            return {"fullsets"};
        }
    }
    
    // int missingprod=1;
    // output number of strings missing in each set
    for (int i = 0; i < mappedPos.size(); i++)
    {
        int missing = pow(alph.size(),g[i].size()) - mappedPos[i].size();
        // cout << "Strings missing in the " << i+1 << "th set: " << missing <<endl;
        // missingprod*=missing;
        // if (missing < h)
        //     h = missing;   
    }

    // if (missingprod < h)
    //     h = missingprod;   

    // cout << "Recursing with h=" << h<<endl;

    vector<string> queries;
    int n; // n must be the amount of overlap

    // frequencies will be a vector<vector<vector<int>>>: for every function we have a vector<vector<int>>, where for
    // every offset given by the element of the function, we compute the vector frequency of the alphabet.
    // to this end, we take every character, and for every mapped position we offset it and look at the character

    vector<vector<array<int, 4>>> freq;
    vector<array<int,4>> gfreq;
    array<int,4> count;


    // loop over all functions
    for (int i = 0; i < g.size(); i++)
    {
        gfreq.clear();

        // loop over all offsets
        for (int j = 0; j < g[i].size(); j++)
        {
            fill(count.begin(), count.end(), 0);

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[i].size(); pos++)
            {
                // compute the character for the current mappedPos + offset, find its index in the alphabet
                // and increase its count
                char c = W[mappedPos[i][pos] + g[i][j]];
                int charpos = distance(alph.begin(), find(alph.begin(), alph.end(), c));
                // cout << "Charpos is " << charpos << endl;
                count[charpos]++;
            }

            gfreq.push_back(count);
        }
        
        freq.push_back(gfreq);
    }
    
    // cout << "Frequencies of chars for all functions g: " << endl;
    // for (int i = 0; i < freq.size(); i++)
    // {

    //     cout << "\ng_" << i << endl;
    //     for (int j = 0; j < freq[i].size(); j++)
    //     {
    //         cout << "Frequencies at offset " << g[i][j] << ": ";
    //         for (int t = 0; t < freq[i][j].size(); t++)
    //         {
    //             cout << "\tfreq(" << alph[t] << ") = " << freq[i][j][t];
    //         }
    //         cout << endl;    
    //     }
    //     cout << endl;
    // }
    

    recEnumTophMultiple(h, "", mappedPos, g, W, queries, freq);

    return queries;
}

// extq: extension so far
// currpos: position from 0 to L of the current character 
// cposind: position index in cpos array 
// cpos: array of positions in 0,L-1 which need to be changed for extension
// output vector: vector of extended string we will output at the end
// extnum: number of extensions to be built (final output size)
void recExtendStringMult(string extq, int cposind, vector<int> &cpos, vector<string> &output, int extnum)
{
    // if we have already performed the correct number of extensions, return
    if(output.size() == extnum)
        return;

    // if we have finished filling all positions in cpos, we are done and output
    if (cposind >= cpos.size())
    {
        output.push_back(extq);
        return;
    }
    else
    {
        // for every alphabet char, recur by placing it at position cposind, and increasing the index
        for (int i = 0; i < alph.size(); i++)
        {
            char c = alph[i];
            // cout << "Considering char " << c << endl; 
            string recext = extq;
            recext[cpos[cposind]] = c; 
            // cout << "Recursing with " << recext<<endl; 
            recExtendStringMult(recext, cposind+1, cpos, output, extnum);
        }
        
    }
    
}

// given a string, a total length and a set of positions, complete the string randomly extnum times with the
// input string characters at the given positions, generating all different strings
vector<string> extendStringMult(string q, int L, vector<int> &pos, int extnum)
{
    if (q == "")
        return {""};
    
    if (q == "x")
        return {"x"};


    // check how many strings we can actually compute
    long int freespace = pow(alph.size(), L-pos.size()); 
    if(extnum > freespace)
        extnum = freespace;
    
    // cout << "We can find at most " << freespace << " extensions. " <<endl;

    // first, create array of complementary pos
    vector<int> cpos;
    int j=0;
    for (int i = 0; i < pos.size(); i++)
    {
        while (j < pos[i] && j<L)
        {
            cpos.push_back(j);
            j++;
        }
        
        j++;
    }

    // cout << "Complementary array is ";
    // for (int i = 0; i < cpos.size(); i++)
    //     cout << "\t" << cpos[i];
    // cout << endl;  
    

    string extq(L, 'X');
    for (int i = 0; i < pos.size(); i++)
        extq[pos[i]] = q[i];

    // cout << "String before filling is " << extq<<endl;
    

    vector<string> output;
    recExtendStringMult(extq, 0, cpos, output, extnum);

    return output;
}



int main()
{
    int N,L,r,m,k,h,extnum, trialse, trialsr;
    string W;
    vector<int> input;
    clock_t startTime, endTime;

    // set rand seed according to clock
    srand(time(NULL));


    // input values of L,r,m,k,h,extnum, trialse, trialsr taken from input.txt file in same directory
    // input file must contain in order L, r, k, m, h, extnum, trialse, trialsr separated by a space.
    ifstream inputfile;
    inputfile.open("input.txt");
    
    if (inputfile.is_open()) 
        inputfile >> L >> r >> k >> m >> h >> extnum >> trialse >> trialsr;
    else
        throw logic_error("Could not open input file");
    
    inputfile.close();


    ofstream outputfile;
    string filename = "./TestResults/L" + to_string(L)+"r"+to_string(r)+"k"+to_string(k)+".txt"; 
    outputfile.open("./TestResults/L" + to_string(L)+"r"+to_string(r)+"k"+to_string(k)+".txt", ios_base::app);


    outputfile << "================== m=" << m << "\t\th="<<h<< "\t\textnum="<<extnum<< " ==================" << endl;

    // our input string consists in an array of chars, representing the concatenation of the distinct SORTED qgrams 
    // characters different from A,C,G,T have already been dealt with
    ifstream inputstring;
    inputstring.open("qgrams.txt");
    inputstring >> W;
    inputstring.close();

    N = W.size();
    cout << "String of length " << N << " with last char " << W[N-1]<<endl;

    // Note: string W accounts for distinct 30-mers; so if L=30 we are all set: all mod 30 positions up to N-L form the input
    // If otherwise L<30, we need a bit more work to identify valid positions to form our input
    if (L == 30)
    {
        for (int i = 0; i < W.size(); i++)
        {
            if (i<=N-L && i%30 == 0)
                input.push_back(i);
        }
    }
    else // otherwise, need to create input set a bit more cumbersomely. Recall that the qgrams are sorted
    {
        // first position is always good
        input.push_back(0);
        int i=30;
        // then look at every Lgram and add its position only if it is different from the previous
        // since they are sorted, this is the only check we need to ensure they are all different
        while(i<= N-L)
        {
            if (W.substr(i,L) != W.substr(i-30,L))
                input.push_back(i);
            
            i+=30;
        }
        
    }
    
    cout << "qgrams are " << input.size() << " at positions ";
    for (int i = 0; i < input.size(); i++)
        cout << input[i] << ", ";
    cout << endl;
    
    cout << "qgrams are : " << endl;
    for (int i = 0; i < input.size(); i++)
        cout << "\t" << W.substr(input[i],L);

    outputfile << "Original input as a text is of total length N=" << N << " and the number of input L-mers is " << input.size() << endl;
    outputfile << "Number of enumeration trials: " << trialse <<endl;
    outputfile << "Number of random trials: " << trialsr <<endl<<endl;

    int succ = 0;
    int fail = 0;
    int count = 0;

       
    while(count < trialse)
    {
        // initialization of vector of hash functions
        // each element g[i] of g is a hash function, composed of at most m projections    
        vector<vector<int>> g;
        for (int i = 0; i < k; i++)
            g.push_back(randomProjections(L, m));
        
        
        // outputfile << "Hash functions are:" << endl;

        // for (int i = 0; i < g.size(); i++)
        // {
        //     outputfile << "g_" << i << ": ";
        //     for (int j = 0; j < g[i].size(); j++)
        //         outputfile << "\t" << g[i][j];
        //     outputfile << endl;
        // }
        

        // we need to map the input, in the sense that we need the simple array of ints
        // of positions whose offsets yield DISTINCT m-mers.
        vector<vector<int>> mapped;
        for (int i = 0; i < g.size(); i++)
            mapped.push_back(mapInput(input, g[i], W));

        
        // cout << "Mapped input sizes: ";
        // for (int i = 0; i < mapped.size(); i++)
        //     cout << mapped[i].size() << ", ";
        // cout << endl;
        

        
        // --------------------- TOPH ENUMERATION WITH K FUNCTIONS ------------------ //
        succ = 0;
        fail = 0;
        
        // We want to test k-1 vs k functions on the same instance, so we first run it with all k, and then with the first k-1 of the k
        outputfile << "Enumeration with " << k << " functions: " << endl;
        cout << "First, enumerate with " << k << " functions." << endl;

        startTime= clock();
        vector<string> enumerated = enumTophMultiple(h, mapped, g, W);
        endTime = clock();

        if (enumerated[0] == "fullsets")
        {
            outputfile << "\tX Projected string sets for " << k << " functions are full." << endl <<endl;
            cout << "Projected string sets are full. " << endl;
            enumerated = {};
        }
        else
        {
            outputfile << "\tV Enumeration without completion in clock time " << endTime - startTime << "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC << endl;
            outputfile << "\tV Found " << enumerated.size() << " strings." << endl;
            cout << "Found " << enumerated.size() << " strings." << endl;
        }
        

        // cout << "Strings found are:";
        // for (int i = 0; i < enumerated.size(); i++)
        //     cout << "\t" << enumerated[i];
        // cout << endl;

        

        // Now we need to complete them randomly; to do so, we need the union of positions for g
        // we thus concatenate them, sort them, and remove duplicate indices
        vector<int> positions = g[0];
        for (int gpos = 1; gpos < g.size(); gpos++)
        {
            for (int i = 0; i < g[gpos].size(); i++)
            {
                vector<int>::iterator it = find(positions.begin(), positions.end(), g[gpos][i]);
                if(it != positions.end())
                    positions.erase(it);
            }
            positions.insert(positions.end(), g[gpos].begin(), g[gpos].end());
        }


        // NEED TO SORT POSITIONS!!!
        sort(positions.begin(), positions.end());

        cout << "Sorted positions are: ";
        for (int i = 0; i < positions.size(); i++)
            cout << "\t" << positions[i];
        cout << endl;

        
        // for each template enumerated, extend it extnum times and check whether we obtain a valid output
        // if so, increase succ, otherwise increase fail
        for (int i = 0; i < enumerated.size(); i++)
        {
            vector<string> extq = extendStringMult(enumerated[i], L , positions, extnum);

            for (int j = 0; j < extq.size(); j++)
            {
                bool found = check(extq[j], r, input, W); // COMPARE WITH CHECK USING INPUTSET
                
                if (found)
                    succ++;
                else    
                    fail++;
                
            }
            
        }

        endTime = clock();

        
        // cout << "Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
        cout << "Successes are " << succ << ", and failures are " << fail << endl<<endl;

        // only print if there was actually a trial
        if (enumerated.size()>0)
        {
            outputfile << "\tV Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
            outputfile << "\tV Successes (of extension) are " << succ << ", and failures are " << fail << endl<<endl;
        }
        

        //  ------------------------- NOW, REPEAT WITH FIRST K-1 FUNCTIONS ----------------------------
        // to do so, simply pop from g and mapped, and clear all other vectors
        cout << "Enumeration with " << k-1 << " hash functions: " << endl;
        outputfile << "Enumeration with " << k-1 << " hash functions: " << endl;

    
        enumerated.clear();
        g.pop_back();
        mapped.pop_back();
        positions.clear();
        succ=0;
        fail=0;

        startTime= clock();
        enumerated = enumTophMultiple(h, mapped, g, W);
        endTime = clock();

        if (enumerated[0] == "fullsets")
        {
            outputfile << "\tX Projected string sets for " << k << " functions are full." << endl << endl;
            cout << "Projected string sets are full. " << endl;
            enumerated = {};
        }
        else
        {
            outputfile << "\tV Enumeration without completion in clock time " << endTime - startTime << "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC << endl;
            outputfile << "\tV Found " << enumerated.size() << " strings." << endl;
            cout << "Found " << enumerated.size() << " strings." << endl;
        }
        

        positions = g[0];
        for (int gpos = 1; gpos < g.size(); gpos++)
        {
            for (int i = 0; i < g[gpos].size(); i++)
            {
                vector<int>::iterator it = find(positions.begin(), positions.end(), g[gpos][i]);
                if(it != positions.end())
                    positions.erase(it);
            }
            positions.insert(positions.end(), g[gpos].begin(), g[gpos].end());
        }


        // NEED TO SORT POSITIONS!!!
        sort(positions.begin(), positions.end());

        cout << "Sorted positions are: ";
        for (int i = 0; i < positions.size(); i++)
            cout << "\t" << positions[i];
        cout << endl;

       

        // for each template enumerated, extend it extnum times and check whether we obtain a valid output
        // if so, increase succ, otherwise increase fail
        for (int i = 0; i < enumerated.size(); i++)
        {
            vector<string> extq = extendStringMult(enumerated[i], L , positions, extnum);

            for (int j = 0; j < extq.size(); j++)
            {
                bool found = check(extq[j], r, input, W); // COMPARE WITH CHECK USING INPUTSET
                
                if (found)
                    succ++;
                else    
                    fail++;
                
            }
            
        }

        endTime = clock();

        
        // cout << "Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
        cout << "Successes are " << succ << ", and failures are " << fail << endl<<endl;
    
        // only print if there was actually a trial
        if (enumerated.size()>0)
        {
            outputfile << "\tV Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
            outputfile << "\tV Successes (of extension) are " << succ << ", and failures are " << fail << endl<<endl;
        }

        count++;
    }


    // --------------------------- EXECUTE RANDOM TRIALS ----------------------
    count = 0;
    succ = 0;
    startTime = clock();

    while (count < trialsr)
    {
        // generate a random string of length L, and check against the input. Count successes/fails
        string extq = randomString(L);
        bool found = check(extq, r, input, W);

        if (found)
            succ++;
        else
            fail++;

        count++;
    }

    endTime = clock();
    
    cout << "Random successes are " << succ << endl;
    outputfile << "Random trials: " << endl << "\tNumber of random successes over " << trialsr << " extractions is " << succ << endl;
    outputfile << "\tElapsed clock time: " << endTime-startTime << "; in seconds " << (endTime-startTime)/CLOCKS_PER_SEC << endl<<endl;

    
    outputfile << endl << endl;
    outputfile.close();
    return 0;
}


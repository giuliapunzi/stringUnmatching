#include <iostream> 
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm> 
#include <cmath>
#include <time.h>
#include <stdexcept>

using namespace std;


// alphabet constant vector
const vector<char> alph = {'A', 'C', 'G', 'T'};


// function to print a vector
void printVector(vector<int> &toPrint)
{
    for (int i = 0; i < toPrint.size(); i++)
    {
        cout << "\t" << toPrint[i];
    }
    cout << endl;
}


// function to print a vector of vectors
// note: ELEMENTS MUST BE INTS
void printVectorVector(vector<vector<int>> &toPrint)
{
    for (int i = 0; i < toPrint.size(); i++)
    {
        cout << i << "th element: ";
        for (int j = 0; j < toPrint[i].size(); j++)
            cout << "\t" << toPrint[i][j];
        cout << endl;
    }
    cout << endl;
}



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


// alternatively, checkset looks for a set as input
bool checkSet(string q, int r, set<string> &inputset)
{
    // for every input string, if the query is close to it then return false
    for (set<string>::iterator it = inputset.begin(); it != inputset.end(); it++)
    {
        if(hdist(q, *it) < r)
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
    cout << "Mapping input..."<<endl;

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


void recBruteEnum(string prefix, int k, vector<int> &given, vector<int> &proj, vector<string> &diff, string &W)
{
    if(k==0)
    {   
        // cout << "String considered is " << prefix << endl;
        // if we generated a string of correct length which is not in the given set, we are done
        
        // we need to compute the given string (by adding the projection offsets) that we are considering
        bool found = false;
        int i = 0;
        while(!found && i < given.size())
        {
            string curr = "";
            for (int j = 0; j < proj.size(); j++)
                curr = curr + W[given[i] + proj[j]];

            if(curr == prefix)
                found = true;

            i++;
        }
        
        
        if(!found)
            diff.push_back(prefix);
        
        return;
    }
    else
    {
        for (int i = 0; i < alph.size(); i++)
        {
            char c = alph[i];
            recBruteEnum(prefix + c, k-1, given, proj, diff, W);
        }   
    }
}


// given a set of strings of the same length, find the ones that do not belong
// with a brute force (alphabetic checking) approach
vector<string> enumBrute(vector<int> &given, vector<int> &proj, string &W)
{
    int n = proj.size(); // length of strings in projected space
    string q;
    vector<string> output = {};

    int total = pow(alph.size(),n);

    cout << "There are " << given.size() << " strings in the set, all of length " << n << endl;
    cout << "Power is " << total  << endl;

    if(given.size() == total)
    {   
        cout << "STRING SET IS FULL!" << endl;
        return {};
    }

    // we iterate over all strings in alphabetical order
    recBruteEnum("", n, given, proj, output, W);
    
    return output;
}


// recursively enumerate strings at distance AT LEAST r from the target set
void recSmartEnumDist(int h, string prefix, int r, vector<vector<int>> &mappedPos, vector<vector<int>> &stringsLeft, vector<pair<int,int>> &ogPos, vector<vector<int>> &g, string &W, vector<string> &queries, vector<vector<int>> &freq0, vector<vector<int>> &freq1)
{
    // cout << "Inside recSmartEnumDist" << endl;

    if(queries.size() == h)
        return; 

    if(g[0].size() == 0 && g[1].size()==0)
    {   
        cout << "String being considered is " << prefix << endl;
        cout << "Sorted and labeled positions are: ";
        for (int i = 0; i < ogPos.size(); i++)
            cout << "\t(" << ogPos[i].first << ", " << ogPos[i].second << ")";
        cout << endl;

        // if we generated a string of correct length which is different from all other strings (stringsLeft is empty)
        // AND THE STRING GENERATED IS AT DISTANCE AT LEAST r FROM THE WHOLE MAPPED INPUT
        if(stringsLeft[0].size() == 0 && stringsLeft[1].size() == 0)
        {
            bool valid = true;
            int pos = 0;

            reverse(prefix.begin(), prefix.end());

            // cout << "Going over first set, of size " << mappedPos[0].size() << endl;

            // go over first set
            while(pos<mappedPos[0].size() && valid)
            {
                // build the string starting at pos with offsets given by each g[0]
                string current = "";
                for (int i = 0; i < ogPos.size(); i++)
                {   
                    cout << "ogPos[i] is " << ogPos[i].first << ", " << ogPos[i].second<<endl;
                    if(ogPos[i].second == 0 || ogPos[i].second == 2)
                    {
                        cout << "Considering position of string " << mappedPos[0][pos] << " and offset " << ogPos[i].first << endl;
                        cout << "In the string there is char " << W[mappedPos[0][pos]+ogPos[i].first] << endl;
                        current.push_back(W[mappedPos[0][pos]+ogPos[i].first]);
                    }
                }
                    

                cout << "Current string (pos " << pos << " with offsets from first function) is " << current<<endl;
                
                if(hdist(current, prefix)<r)
                {
                    cout << "It is too close!" << endl;
                    valid = false;
                }
                
                pos++;
            }

            pos = 0;
            // go over second set
            while(pos<mappedPos[1].size() && valid)
            {
                // build the string starting at pos with offsets given by each g[0]
                string current = "";
                for (int i = 0; i < ogPos.size(); i++)
                {
                    if (ogPos[i].second == 1 || ogPos[i].second == 2)
                    {
                        current.push_back(W[mappedPos[1][pos]+ogPos[i].first]);
                    }
                }
                    
                
                if(hdist(current, prefix)<r)
                    valid = false;
                
                pos++;
            }

            if(valid)
            {
                queries.push_back(prefix);
                cout << "Valid string found: " << prefix << endl;
            }
        }

        return;
    }
    else
    {
        // take the next position; first deal with case when they are equal
        // positions are considered from the back for ease of push/pop
        int current;
        int size0 = g[0].size();
        int size1 = g[1].size();

        vector<int> frequencies;
        vector<int> freqInd;
        
        // last position is overlapping
        if(size0 > 0 && size1 > 0 && g[0][size0-1] == g[1][size1-1])
        {

            current = g[0][size0-1];
            g[0].pop_back();
            g[1].pop_back();

            // cout << "Next position is " << current << endl;


            // in this case, the frequencies are the sum of the frequencies of the two vectors
            frequencies = freq0[freq0.size()-1];
            for (int i = 0; i < frequencies.size(); i++)
                frequencies[i]+=freq1[freq1.size()-1][i];

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

            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[freqInd[i]]
                char c = alph[freqInd[i]];
                vector<int> removed0;
                vector<int> removed1;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // recall: position (offset) is current, the same for both g0 and g1
                // let us first do first set, then second
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed0.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0] TO BE DONE LATER!!!
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed0.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed0[j]);
                    stringsLeft[0].erase(it);
                }
                

                // the same for second vector 
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed1
                        removed1.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[1]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed1 from stringsLeft
                for (int j = 0; j < removed1.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed1[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;
                
                
                recSmartEnumDist(h, prefix + c, r, mappedPos, stringsLeft, ogPos, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed0.begin(), removed0.end() );
                stringsLeft[1].insert( stringsLeft[1].end(), removed1.begin(), removed1.end() );

            }

            // re-add current to g before returning
            g[0].push_back(current);
            g[1].push_back(current);
            return;
        }
        

        // last position is in first vector
        if (size1 == 0 || (size1 > 0 && size0 > 0 && g[0][size0-1] > g[1][size1-1]))
        {
            current = g[0][size0-1];
            g[0].pop_back();

            // cout << "Next position is " << current << endl;
            

            // in this case, the frequencies are just freq0
            frequencies = freq0[freq0.size()-1];
            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );

            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }
            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[0]
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {   
                    // cout << "Now considering position " << stringsLeft[0][j] << "; we will sum it to offset " << current << endl;
                    // cout << "In the string, we are looking at " << W[stringsLeft[0][j]+current] << endl;
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed[j]);
                    stringsLeft[0].erase(it);
                }


                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;


                recSmartEnumDist(h, prefix + c, r, mappedPos, stringsLeft, ogPos, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[0].push_back(current);
            return;
        }
        
        
        if(size0 == 0 || (size0>0 && size1>0 && g[0][size0-1] < g[1][size1-1])) // last position is in second vector
        {
            current = g[1][size1-1];
            g[1].pop_back();

            // cout << "Next position is " << current << endl;
            
            // in this case, the frequencies are just freq1
            frequencies = freq1[freq1.size()-1];

            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );
            
            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }

            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[1]
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;

                recSmartEnumDist(h, prefix + c, r, mappedPos, stringsLeft, ogPos, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[1].insert( stringsLeft[1].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[1].push_back(current);
            return;
        }

        throw logic_error("Next position does not fall under any case!");
           
    }
}


vector<string> enumSmartDist(int h, int r, vector<vector<int>> &mappedPos, vector<vector<int>> &g, string &W)
{
    // cout <<endl<<  "Inside enumSmartDist, looking for first " << h << " strings at distance at least " << r << endl;
    vector<vector<int>> stringsLeft = mappedPos;  
    vector<vector<int>> ogPos = g;
    vector<string> queries;
    int n; // n must be the amount of overlap

    vector<vector<int>> freq0;
    vector<vector<int>> freq1;
    vector<int> current;

    // first loop for g[0]
    // for every offset
    for (int i = 0; i < g[0].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[0].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[0][pos] + g[0][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq0.push_back(current);
    }

    // cout << "Frequencies of chars for g[0]: " << endl;
    // for (int i = 0; i < freq0.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[0][i] << ": " << endl;
    //     for (int j = 0; j < freq0[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq0[i][j];
    //     }
    //     cout << endl;
        
    // }
    

    
    // second loop for g[1]
    // for every offset
    for (int i = 0; i < g[1].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[1].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[1][pos] + g[1][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq1.push_back(current);
    }

    // cout << endl << "Frequencies of chars for g[1]: " << endl;
    // for (int i = 0; i < freq1.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[1][i] << ": " << endl;
    //     for (int j = 0; j < freq1[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq1[i][j];
    //     }
    //     cout << endl;
        
    // }


    // NEED THE MERGE TO CONSIDER CORRECT POSITIONS INSIDE THE STRING
    // we thus concatenate them, sort them, and remove duplicate indices, and pass them on

    // start with a vector whose second component indicates whether the index belongs to g0, g1 or both (2)
    vector<pair<int,int>> positions;
    for (int i = 0; i < ogPos[0].size(); i++)
    {
        vector<int>::iterator it = find(ogPos[1].begin(), ogPos[1].end(), ogPos[0][i]);
        // if element does not belong to ogPos[1], append it with second index 0, otherwise with second index 2 and remove it from ogPos
        if(it == ogPos[1].end())
            positions.push_back(make_pair(ogPos[0][i], 0));
        else
        {
            positions.push_back(make_pair(ogPos[0][i], 2));
            ogPos[1].erase(it);
        }
    }
        

    for (int i = 0; i < ogPos[1].size(); i++)
        positions.push_back(make_pair(ogPos[1][i], 1));
    

    // NEED TO SORT POSITIONS!!! (they are sorted by first index)
    sort(positions.begin(), positions.end());

    cout << "Now, sorted and labeled positions are: ";
    for (int i = 0; i < positions.size(); i++)
        cout << "\t(" << positions[i].first << ", " << positions[i].second << ")";
    cout << endl;


    cout << "Before recursing, mappedPos size is " << mappedPos[0].size();

    recSmartEnumDist(h, "", r, mappedPos, stringsLeft, positions, g, W, queries, freq0, freq1);

    return queries;
}




void recSmartEnum(string prefix, vector<vector<int>> &stringsLeft, vector<vector<int>> &g, string &W, vector<string> &queries, vector<vector<int>> &freq0, vector<vector<int>> &freq1)
{
    if(g[0].size() == 0 && g[1].size()==0)
    {   
        // cout << "String considered is " << prefix << endl;
        // if we generated a string of correct length which is different from all other strings (stringsLeft is empty), we are done
        if(stringsLeft[0].size() == 0 && stringsLeft[1].size() == 0)
        {
            reverse(prefix.begin(), prefix.end());
            // we need to compute the given string (by adding the projection offsets) that we are considering
            queries.push_back(prefix);
            cout << "Valid string found: " << prefix << endl;
        }

        return;
    }
    else
    {
        // take the next position; first deal with case when they are equal
        // positions are considered from the back for ease of push/pop
        int current;
        int size0 = g[0].size();
        int size1 = g[1].size();

        vector<int> frequencies;
        vector<int> freqInd;
        
        // last position is overlapping
        if(size0 > 0 && size1 > 0 && g[0][size0-1] == g[1][size1-1])
        {

            current = g[0][size0-1];
            g[0].pop_back();
            g[1].pop_back();

            // cout << "Next position is " << current << endl;


            // in this case, the frequencies are the sum of the frequencies of the two vectors
            frequencies = freq0[freq0.size()-1];
            for (int i = 0; i < frequencies.size(); i++)
                frequencies[i]+=freq1[freq1.size()-1][i];

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

            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[freqInd[i]]
                char c = alph[freqInd[i]];
                vector<int> removed0;
                vector<int> removed1;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // recall: position (offset) is current, the same for both g0 and g1
                // let us first do first set, then second
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed0.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0] TO BE DONE LATER!!!
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed0.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed0[j]);
                    stringsLeft[0].erase(it);
                }
                

                // the same for second vector 
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed1
                        removed1.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[1]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed1 from stringsLeft
                for (int j = 0; j < removed1.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed1[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;
                
                
                recSmartEnum(prefix + c, stringsLeft, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed0.begin(), removed0.end() );
                stringsLeft[1].insert( stringsLeft[1].end(), removed1.begin(), removed1.end() );

            }

            // re-add current to g before returning
            g[0].push_back(current);
            g[1].push_back(current);
            return;
        }
        

        // last position is in first vector
        if (size1 == 0 || (size1 > 0 && size0 > 0 && g[0][size0-1] > g[1][size1-1]))
        {
            current = g[0][size0-1];
            g[0].pop_back();

            // cout << "Next position is " << current << endl;
            

            // in this case, the frequencies are just freq0
            frequencies = freq0[freq0.size()-1];
            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );

            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }
            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[0]
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {   
                    // cout << "Now considering position " << stringsLeft[0][j] << "; we will sum it to offset " << current << endl;
                    // cout << "In the string, we are looking at " << W[stringsLeft[0][j]+current] << endl;
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed[j]);
                    stringsLeft[0].erase(it);
                }


                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;


                recSmartEnum(prefix + c, stringsLeft, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[0].push_back(current);
            return;
        }
        
        
        if(size0 == 0 || (size0>0 && size1>0 && g[0][size0-1] < g[1][size1-1])) // last position is in second vector
        {
            current = g[1][size1-1];
            g[1].pop_back();

            // cout << "Next position is " << current << endl;
            
            // in this case, the frequencies are just freq1
            frequencies = freq1[freq1.size()-1];

            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );
            
            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }

            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[1]
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;

                recSmartEnum(prefix + c, stringsLeft, g, W, queries, freq0, freq1);

                // now re-add stringsLeft indices for next iteration
                stringsLeft[1].insert( stringsLeft[1].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[1].push_back(current);
            return;
        }

        throw logic_error("Next position does not fall under any case!");
           
    }
}


vector<string> enumSmart(vector<vector<int>> &mappedPos, vector<vector<int>> &g, string &W)
{
    // cout << "Inside enumSmart" << endl;

    vector<string> queries;
    int n; // n must be the amount of overlap

    vector<vector<int>> freq0;
    vector<vector<int>> freq1;
    vector<int> current;

    // first loop for g[0]
    // for every offset
    for (int i = 0; i < g[0].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[0].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[0][pos] + g[0][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq0.push_back(current);
    }

    // cout << "Frequencies of chars for g[0]: " << endl;
    // for (int i = 0; i < freq0.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[0][i] << ": " << endl;
    //     for (int j = 0; j < freq0[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq0[i][j];
    //     }
    //     cout << endl;
        
    // }
    

    
    // second loop for g[1]
    // for every offset
    for (int i = 0; i < g[1].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[1].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[1][pos] + g[1][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq1.push_back(current);
    }

    // cout << endl << "Frequencies of chars for g[1]: " << endl;
    // for (int i = 0; i < freq1.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[1][i] << ": " << endl;
    //     for (int j = 0; j < freq1[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq1[i][j];
    //     }
    //     cout << endl;
        
    // }


    recSmartEnum("", mappedPos, g, W, queries, freq0, freq1);

    return queries;
}



void recSmartEnumToph(int h, string prefix, vector<vector<int>> &stringsLeft, vector<vector<int>> &g, string &W, vector<string> &queries, vector<vector<int>> &freq0, vector<vector<int>> &freq1)
{
    // cout << "Inside recursion" << endl;
    if(queries.size() == h)
        return; 

    if(g[0].size() == 0 && g[1].size()==0)
    {   
        // cout << "String considered is " << prefix << endl;
        // if we generated a string of correct length which is different from all other strings (stringsLeft is empty), we are done
        if(stringsLeft[0].size() == 0 && stringsLeft[1].size() == 0)
        {
            reverse(prefix.begin(), prefix.end());
            // we need to compute the given string (by adding the projection offsets) that we are considering
            queries.push_back(prefix);
            // cout << "Valid string found: " << prefix << endl;
        }

        return;
    }
    else
    {
        // take the next position; first deal with case when they are equal
        // positions are considered FROM THE BACK for ease of push/pop
        int current;
        int size0 = g[0].size();
        int size1 = g[1].size();

        vector<int> frequencies;
        vector<int> freqInd;
        
        // last position is overlapping
        if(size0 > 0 && size1 > 0 && g[0][size0-1] == g[1][size1-1])
        {

            current = g[0][size0-1];
            g[0].pop_back();
            g[1].pop_back();

            // cout << "Next position is " << current << endl;


            // in this case, the frequencies are the sum of the frequencies of the two vectors
            frequencies = freq0[freq0.size()-1];
            for (int i = 0; i < frequencies.size(); i++)
                frequencies[i]+=freq1[freq1.size()-1][i];

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

            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[freqInd[i]]
                char c = alph[freqInd[i]];
                vector<int> removed0;
                vector<int> removed1;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // recall: position (offset) is current, the same for both g0 and g1
                // let us first do first set, then second
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed0.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0] TO BE DONE LATER!!!
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed0.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed0[j]);
                    stringsLeft[0].erase(it);
                }
                

                // the same for second vector 
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed1
                        removed1.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[1]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed1 from stringsLeft
                for (int j = 0; j < removed1.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed1[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;
                
                recSmartEnumToph(h, prefix + c, stringsLeft, g, W, queries, freq0, freq1);

                // if(queries.size() == h)
                //     return; 
                

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed0.begin(), removed0.end() );
                stringsLeft[1].insert( stringsLeft[1].end(), removed1.begin(), removed1.end() );

            }

            // re-add current to g before returning
            g[0].push_back(current);
            g[1].push_back(current);
            return;
        }
        

        // last position is in first vector
        if (size1 == 0 || (size1 > 0 && size0 > 0 && g[0][size0-1] > g[1][size1-1]))
        {
            current = g[0][size0-1];
            g[0].pop_back();

            // cout << "Next position is " << current << endl;
            

            // in this case, the frequencies are just freq0
            frequencies = freq0[freq0.size()-1];
            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );

            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }
            
            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[0]
                for (int j  = 0; j < stringsLeft[0].size(); j++)
                {   
                    // cout << "Now considering position " << stringsLeft[0][j] << "; we will sum it to offset " << current << endl;
                    // cout << "In the string, we are looking at " << W[stringsLeft[0][j]+current] << endl;
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[0][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[0][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[0].erase(stringsLeft[0].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[0].begin(), stringsLeft[0].end(), removed[j]);
                    stringsLeft[0].erase(it);
                }


                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;

                recSmartEnumToph(h, prefix + c, stringsLeft, g, W, queries, freq0, freq1);
                
                // if(queries.size() == h)
                //     return; 

                // now re-add stringsLeft indices for next iteration
                stringsLeft[0].insert( stringsLeft[0].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[0].push_back(current);
            return;
        }
        
        
        if(size0 == 0 || (size0>0 && size1>0 && g[0][size0-1] < g[1][size1-1])) // last position is in second vector
        {
            current = g[1][size1-1];
            g[1].pop_back();

            // cout << "Next position is " << current << endl;
            
            // in this case, the frequencies are just freq1
            frequencies = freq1[freq1.size()-1];

            // cout << "Considering frequencies vector ";
            // for (int i = 0; i < frequencies.size(); i++)
            // {
            //     cout << "\t" << frequencies[i];
            // }
            
            // order the indices such that frequencies is ordered (ascendingly) according to them
            freqInd = {0,1,2,3};
            sort(freqInd.begin(), freqInd.end(), [&](int i1, int i2) { return frequencies[i1] < frequencies[i2]; } );
            
            // cout << endl << "sorted indices: " << endl;
            // for (int i = 0; i < freqInd.size(); i++)
            // {
            //     cout << "Index " << freqInd[i] << " with value " << frequencies[freqInd[i]] << endl;
            // }

            // this goes through all character in increasing frequency order
            for (int i = 0; i < freqInd.size(); i++)
            {
                // current character is exactly alph[i]
                char c = alph[freqInd[i]];
                vector<int> removed;

                // now, we need to update stringsLeft before moving on
                // we need to scan all stringsLeft and remove the indices corresponding to m-mers having char c at the correct (last) position
                // we only go through stringsLeft[1]
                for (int j  = 0; j < stringsLeft[1].size(); j++)
                {
                    // if the character in the string is different from c, remove them
                    // we also need to keep them to reinsert for successive iteration
                    if(W[stringsLeft[1][j]+current] != c)
                    {   
                        // add the position to vector removed0
                        removed.push_back(stringsLeft[1][j]);
                        // erase the jth element from vector stringsLeft[0]
                        // stringsLeft[1].erase(stringsLeft[1].begin() + j);
                    }
                }

                //  now, remove every element of removed0 from stringsLeft
                for (int j = 0; j < removed.size(); j++)
                {
                    vector<int>::iterator it = find(stringsLeft[1].begin(), stringsLeft[1].end(), removed[j]);
                    stringsLeft[1].erase(it);
                }

                // cout << "About to recurse by appending char " << c << endl;
                // cout << "Current strings left are:" << endl;
                // cout << "StringsLeft[0] = ";
                // for (int j = 0; j < stringsLeft[0].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[0][j];
                // }
                // cout << endl;

                // cout << "StringsLeft[1] = ";
                // for (int j = 0; j < stringsLeft[1].size(); j++)
                // {
                //     cout << "\t" << stringsLeft[1][j];
                // }
                // cout << endl;

                recSmartEnumToph(h, prefix + c, stringsLeft, g, W, queries, freq0, freq1);
                
                // if(queries.size() == h)
                //     return; 

                // now re-add stringsLeft indices for next iteration
                stringsLeft[1].insert( stringsLeft[1].end(), removed.begin(), removed.end() );
            }
        
            // re-add current to g before returning
            g[1].push_back(current);
            return;
        }

        throw logic_error("Next position does not fall under any case!");
           
    }
}


vector<string> enumSmartToph(int h, vector<vector<int>> &mappedPos, vector<vector<int>> &g, string &W)
{
    // cout << "Inside enumSmart" << endl;
    int n0 = g[0].size(); // length of strings in projected space
    int n1 = g[1].size();

    if(mappedPos[0].size() == pow(alph.size(),g[0].size()) || mappedPos[1].size() == pow(alph.size(),g[1].size()))
    {   
        cout << "STRING SET IS FULL!" << endl;
        return {};
    }

    int missing0 = pow(alph.size(),g[0].size()) - mappedPos[0].size();
    int missing1 = pow(alph.size(),g[1].size()) - mappedPos[1].size();

    cout << "Strings missing in first set: " << missing0 <<endl;
    cout << "Strings missing in second set: " << missing1 <<endl;

    if(missing0<h)
    {
        cout << "missing0 < h" << endl;
        h=missing0;
    }
        
    
    if(missing1<h)
    {
        cout << "missing1 < h" << endl;
        h=missing1;
    }

    cout << "Recursing with h=" << h<<endl;

    vector<string> queries;
    int n; // n must be the amount of overlap

    vector<vector<int>> freq0;
    vector<vector<int>> freq1;
    vector<int> current;

    // first loop for g[0]
    // for every offset
    for (int i = 0; i < g[0].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[0].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[0][pos] + g[0][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq0.push_back(current);
    }

    // cout << "Frequencies of chars for g[0]: " << endl;
    // for (int i = 0; i < freq0.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[0][i] << ": " << endl;
    //     for (int j = 0; j < freq0[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq0[i][j];
    //     }
    //     cout << endl;
        
    // }
    

    
    // second loop for g[1]
    // for every offset
    for (int i = 0; i < g[1].size(); i++)
    {
        current.clear();

        // for every alphabet char, we are performing the count
        for (int charind = 0; charind < alph.size(); charind++)
        {
            int count = 0;

            // for every valid position for the current hash function
            for (int pos = 0; pos < mappedPos[1].size(); pos++)
            {
                // find character that occurs at position + offset is our char, increase its count
                if(W[mappedPos[1][pos] + g[1][i]] == alph[charind])
                    count++;
            }

            current.push_back(count);            
        }

        freq1.push_back(current);
    }

    // cout << endl << "Frequencies of chars for g[1]: " << endl;
    // for (int i = 0; i < freq1.size(); i++)
    // {
    //     cout << "Frequencies at offset " << g[1][i] << ": " << endl;
    //     for (int j = 0; j < freq1[i].size(); j++)
    //     {
    //         cout << "\tfreq(" << alph[j] << ") = " << freq1[i][j];
    //     }
    //     cout << endl;
        
    // }


    recSmartEnumToph(h, "", mappedPos, g, W, queries, freq0, freq1);

    return queries;
}



// given a string, a total length and a set of positions, complete the string randomly with the
// input string characters at the given positions
string extendString(string q, int L, vector<int> &pos)
{
    if (q == "")
        return "";
    
    if (q == "x")
        return "x";
    
    string extq = "";
    int posindex = 0;
    int i = 0;

    // we need to complete the query according to the positions!
    while(i < L)
    {
        // while we are not at a position occupied by g, fill randomly
        while (i!= pos[posindex] && i < L)
        {
            extq.push_back(alph[rand() % alph.size()]);
            i++;
        }

        if(i<L)    
        {
            extq.push_back(q[posindex]);
            posindex++;  
            i++;
        }
        
    }

    return extq;
}


int main()
{
    int N,L,r,m,k,h,genum;
    string W;
    vector<int> input;
    clock_t startTime, endTime;

    // set rand seed according to clock
    srand(time(NULL));

    cout << "How many genomes do you want to use? (1-188039) ";
    cin >> genum;
    cout << endl;
    // genum = 1000;

    if (genum < 1 || genum > 188039)
        throw logic_error("Invalid number of genomes. ");
    
    
    cout << "Insert length of input strings (L-mers): ";
    cin >> L;
    cout << endl;

    // L=30;

    cout << "Insert required distance from input: ";
    cin >> r;
    cout << endl;

    // r=10;

    
    char rore;
    cout << "Do you want random generation or enumeration, or both? (r/e/b) ";
    cin  >> rore;
    cout << endl;


    if(rore != 'e' && rore != 'r' && rore!= 'b')
        throw logic_error("Invalid choice for generation!");

    int trialse, trialsr;

    if(rore == 'e' || rore == 'b')
    {
        cout << "Insert size of target space of LSH: ";
        cin >> m;
        cout << endl;
        

        cout << "Number of hash functions for LSH will be two." << endl;
        // cin >> k;
        // cout << endl;
        k = 2; 

        cout << "Insert cutoff h for enumeration (-1 for full): ";
        cin >> h;
        cout << endl;
        
        // if h=-1, full enumeration

        cout << "How many trials do you want to perform for enumeration? ";
        cin >> trialse;
        cout << endl;

        if(rore == 'b')
        {
            cout << "How many random trials do you want to perform? ";
            cin >> trialsr;
            cout << endl;
        }
    }
    else
    {
        cout << "How many trials do you want to perform? ";
        cin >> trialsr;
        cout << endl;
    }

    ofstream outputfile;
    if(rore == 'e')
    {
        string outputname = "./ExpResults/"+to_string(genum)+"GenomesL" + to_string(L)+"r"+to_string(r)+".txt";
        outputfile.open(outputname, ios_base::app);
    }
    if(rore == 'r')   
    {
        string outputname = "./ExpResults/Rand"+to_string(genum)+"GenomesL" + to_string(L)+"r"+to_string(r)+".txt";
        outputfile.open(outputname, ios_base::app);
    }
    if(rore == 'b')   
    {
        string outputname = "./ExpResults/Both"+to_string(genum)+"GenomesL" + to_string(L)+"r"+to_string(r)+".txt";
        cout << "Output in file " << outputname << endl;
        outputfile.open(outputname, ios_base::app);
    }

    outputfile << endl << endl << endl << "================== m=" << m << "\t\th="<<h<< " ==================" << endl;
    outputfile << "Original input is made of " << genum << " genomes."; //, specifically no. ";


    ifstream myfile;

    // WE NEED TO CHOOSE THE GENUM GENOMES AT RANDOM IN THE DATASET FOLDER 
    int genomecount=0;
    while (genomecount < genum)
    {
        int curr = rand()%188039 +1;
        // outputfile << curr << ", \t";
        string filename = "../../PhD/Codice/C/Farthest/Dataset/genome" + to_string(curr) + ".txt";
        myfile.open(filename); // myfile.open("./Dataset/genome1.txt");

        // cout << "Considering genome no. " << curr << endl;

        if (myfile.is_open()) 
        {
            string line;
            getline(myfile, line);
            // cout << "Discarding line: " << line << endl;
            // cout << "Next char will be " << (char) (myfile.peek()) << endl;

            while(!myfile.eof())
            {
                char c = myfile.get();
                if (myfile.eof()) break;

                if(c == 'U')
                {
                    cout << "We have an U, replacing with T" << endl;
                    c = 'T';
                }

                if(c== 'R')
                {
                    // cout << "We have an R, replacing with A or G" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'A';
                    else
                        c = 'G';
                }

                if(c== 'Y')
                {
                    // cout << "We have a Y, replacing with C or T" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'C';
                    else
                        c = 'T';
                }

                if(c== 'K')
                {
                    // cout << "We have a K, replacing with G or T" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'G';
                    else
                        c = 'T';
                }

                if(c== 'M')
                {
                    // cout << "We have an M, replacing with A or C" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'A';
                    else
                        c = 'C';
                }

                if(c== 'S')
                {
                    // cout << "We have an S, replacing with C or G" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'C';
                    else
                        c = 'G';
                }

                if(c== 'W')
                {
                    // cout << "We have a W, replacing with A or T" << endl;
                    int choice = rand()%2;
                    if (choice == 0)
                        c = 'A';
                    else
                        c = 'T';
                }

                if(c== 'B')
                {
                    // cout << "We have a B, replacing with C, G or T" << endl;
                    int choice = rand()%3;
                    if (choice == 0)
                        c = 'C';
                    else if (choice == 1)
                    {
                        c = 'G';
                    }
                    else
                        c = 'T';
                }

                if(c== 'D')
                {
                    // cout << "We have a D, replacing with A, G or T" << endl;
                    int choice = rand()%3;
                    if (choice == 0)
                        c = 'A';
                    else if (choice == 1)
                    {
                        c = 'G';
                    }
                    else
                        c = 'T';
                }

                if(c== 'H')
                {
                    // cout << "We have an H, replacing with A, C or T" << endl;
                    int choice = rand()%3;
                    if (choice == 0)
                        c = 'A';
                    else if (choice == 1)
                    {
                        c = 'C';
                    }
                    else
                        c = 'T';
                }

                if(c== 'V')
                {
                    // cout << "We have a V, replacing with A, C or G" << endl;
                    int choice = rand()%3;
                    if (choice == 0)
                        c = 'A';
                    else if (choice == 1)
                    {
                        c = 'C';
                    }
                    else
                        c = 'G';
                }

                if(c== 'N')
                {
                    // cout << "We have an N, replacing with A, C, G or T" << endl;
                    int choice = rand()%4;
                    if (choice == 0)
                        c = 'A';
                    else if (choice == 1)
                    {
                        c = 'C';
                    }
                    else if (choice == 2)
                        c = 'G';
                    else
                        c = 'T';
                }

                W.push_back(c);
            }
        }   

        myfile.close();

        genomecount++; 

        // different genomes will be separated by $$
        if(genomecount < genum) W.push_back('$');

    }
    outputfile << endl;
    
    
    // myfile.open("smol2.txt");

    
    N= W.length();
    cout << "String length is " << N << endl;
    // cout << "String is " << W << endl;
    
    cout << "First 100 chars of W are " << W.substr(0,100) << endl;

    // try going through a set
    set<string> inputset;
    pair<set<string>::iterator,bool> insertion;  
    bool newgenome = false; 
    for (int i = 0; i <= N-L; i++)
    {
        // cout << "Last char is "  W[i+L] << endl;
        // if our L-mer ends with a $, skip ahead
        if(W[i+L-1] == '$')
        {
            // if(i<N-L) cout << "The substring of length 2L+1 is " << W.substr(i,2*L) << endl;

            i+=L;
            newgenome = true;
        }
           
        if(newgenome)
        {
            // cout << "About to consider " << W.substr(i,L) << endl;
            newgenome = false;
        }

        insertion = inputset.insert(W.substr(i,L));
        // true if element inserted
        if(insertion.second)
            input.push_back(i);   
        
    }
    
    cout << endl;

    if (input.size() != inputset.size())
        throw logic_error("Set elements are different number than positions!");
    
    

    cout << "Dealing with " << input.size() << " L-mers."<< endl;

    // // cout << "There are " << inputset.size() << " elements in the set. " << endl;

//    cout << "Printing input: ";
//    for (set<string>::iterator it = inputset.begin(); it != inputset.end(); it++)
//    {
//        cout << "\t" << *it;
//    }
//    cout << endl;


    // writing to output
    outputfile << "Original input as a text is of total length N=" << N << " and the number of input L-mers is " << input.size() << endl<<endl;
    if(rore == 'e' || rore == 'b')
        outputfile << "Number of enumeration trials: " << trialse <<endl;

    if(rore == 'r' || rore == 'b')
        outputfile << "Number of random trials: " << trialsr <<endl;

    outputfile << endl;

    int succ = 0;
    int fail = 0;
    int count = 0;
    
    if(rore == 'e' || rore == 'b')
    {
            
        while(count < trialse)
        {
            // initialization of vector of hash functions
            // each element g[i] of g is a hash function, composed of at most m projections    
            vector<vector<int>> g;
            for (int i = 0; i < k; i++)
                g.push_back(randomProjections(L, m));
            
            
            cout << "Vector of hash functions: " << endl;
            printVectorVector(g);

            outputfile << "Hash functions are:" << endl;
            outputfile << "g_0: ";
            for (int j = 0; j < g[0].size(); j++)
                outputfile << "\t" << g[0][j];
            outputfile << endl;

            outputfile << "g_1: ";
            for (int j = 0; j < g[1].size(); j++)
                outputfile << "\t" << g[1][j];
            outputfile << endl << endl;


            // we need to map the input, in the sense that we need the simple array of ints
            // of positions whose offsets yield DISTINCT m-mers.
            vector<vector<int>> mapped;
            for (int i = 0; i < g.size(); i++)
                mapped.push_back(mapInput(input, g[i], W));

            cout << "Mapped input sizes: " << mapped[0].size() <<", "<< mapped[1].size() << endl;
            
            // // cout << "Mapped input:" << endl;
            // // cout << "According to first function: ";
            // // for (int i = 0; i < mapped[0].size(); i++)
            // // {
            // //     string curr = "";

            // //     for (int j = 0; j < g[0].size(); j++)
            // //     {
            // //         curr = curr+ W[mapped[0][i] + g[0][j]];
            // //     }

            // //     cout << "\t" << curr << " starting from pos " << mapped[0][i]; 
            // // }
            // // cout << endl;
            
            // cout << "According to second function: ";
            // for (int i = 0; i < mapped[1].size(); i++)
            // {
            //     string curr = "";

            //     for (int j = 0; j < g[1].size(); j++)
            //     {
            //         curr = curr+ W[mapped[1][i] + g[1][j]];
            //     }

            //     cout << "\t" << curr<< " starting from pos " << mapped[1][i]; 
            // }
            // cout << endl;

            // cout << "Vector of mapped input positions: " << endl;
            // printVectorVector(mapped);

            
            // --------------------- NOW TESTING SMART ENUMERATION ------------------ //
            succ = 0;
            fail = 0;

            // cout << "Testing smart enumeration" << endl;
            outputfile <<"Starting enumeration" << endl;

            startTime= clock();

            vector<string> enumerated;

            if(h==-1)
                enumerated = enumSmart(mapped, g, W);
            else
                enumerated = enumSmartToph(h, mapped, g, W);


            endTime = clock();

            outputfile << "Enumeration without completion in clock time " << endTime - startTime << "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC << endl;
            outputfile << "Found " << enumerated.size() << " strings." << endl;

            cout << "Found " << enumerated.size() << " strings." << endl;

            // cout << "Strings found are:";
            // for (int i = 0; i < enumerated.size(); i++)
            //     cout << "\t" << enumerated[i];
            // cout << endl;


            // cout << "Vector g is "<< endl;
            // printVectorVector(g);
            

            // Now we need to complete them randomly; to do so, we need the union of positions for g
            // we thus concatenate them, sort them, and remove duplicate indices
            vector<int> positions = g[0];
            for (int i = 0; i < g[1].size(); i++)
            {
                vector<int>::iterator it = find(positions.begin(), positions.end(), g[1][i]);
                if(it != positions.end())
                    positions.erase(it);
            }
            positions.insert(positions.end(), g[1].begin(), g[1].end());

            // cout << "Union of positions is: ";
            // for (int i = 0; i < positions.size(); i++)
            //     cout << "\t" << positions[i];
            // cout << endl;

            // NEED TO SORT POSITIONS!!!
            sort(positions.begin(), positions.end());

            cout << "Sorted positions are: ";
            for (int i = 0; i < positions.size(); i++)
                cout << "\t" << positions[i];
            cout << endl;

            
            // TRANSFORMED INTO JUST ONE LOOP
            for (int i = 0; i < enumerated.size(); i++)
            {
                string extq = extendString(enumerated[i], L, positions);

                // time_t compareBeg,compareEnd;

                // compareBeg = clock();
                bool found = check(extq, r, input, W); // COMPARE WITH CHECKSET USING INPUTSET
                // compareEnd = clock();

                // cout << "Check with vector takes " << compareEnd-compareBeg << " clock its." << endl;

                // compareBeg = clock();
                // bool foundset = checkSet(extq, r, inputset); // COMPARE WITH CHECKSET USING INPUTSET
                // compareEnd = clock();

                // cout << "Check with set takes " << compareEnd-compareBeg << " clock its." << endl;

                // cout << "Considering string " << extq;

                if (found)
                {
                    // cout << " \t SUCCESS!!" << endl;
                    // cout << "String found is " << extq << endl;
                    // outputfile << "Found string " << extq << endl;
                    succ++;
                }
                else
                {
                    // cout << " \t FAILURE" << endl;
                    fail++;
                }
            }


            endTime = clock();

            
            cout << "Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
            cout << "Successes are " << succ << ", and failures are " << fail << endl<<endl;

            outputfile << "Total elapsed clock time for trial " << count+1 << " is " << endTime - startTime<< "; in seconds: " << ((float) endTime -startTime)/CLOCKS_PER_SEC  << endl;
            outputfile << "Successes (of extension) are " << succ << ", and failures are " << fail << endl<<endl;
        
            count++;
        }

    }
    
    if(rore == 'r' || rore == 'b')
    {
        count = 0;
        succ = 0;
        outputfile << endl << endl << "Now starting random trials"<< endl;
        startTime = clock();
        while (count < trialsr)
        {
            string extq = randomString(L);

            bool found = check(extq, r, input, W);

            if (found)
            {
                //cout << "SUCCESS!!" << endl;
                // cout << "String found is " << extq << " in time " << end-begin << endl;
                // outputfile << "Found string " << extq << " in time " << end-begin << endl;
                succ++;
            }
            else
            {
                //cout << "FAILURE" << endl;
                fail++;
            }

            count++;
        }
        endTime = clock();
        
        cout << "Random successes are " << succ << endl;
        outputfile << "Number of random successes over " << trialsr << " extractions is " << succ << endl;
        outputfile << "Elapsed clock time: " << endTime-startTime << "; in seconds " << (endTime-startTime)/CLOCKS_PER_SEC << endl<<endl;
    }

    outputfile << endl << endl;
    outputfile.close();

    return 0;
}



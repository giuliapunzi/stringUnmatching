#include <iostream> 
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm> 
#include <cmath>
#include <time.h>
#include <windows.h>

using namespace std;


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



// structure for map of the input
typedef vector<set<string>> target;


// function which given the input and the hash functions, maps the input
target mapInput(set<string> &input, vector<vector<int>> &g)
{
    target mapped;

    for (int i = 0; i < g.size(); i++)
    {   
        set<string> current;

        // g[i] is a vector of ints expressing function g_i
        // we iterate over the whole input set
        for (set<string>::iterator inputit = input.begin(); inputit != input.end(); inputit++)
        {
            string s = "";
            string inputel = *inputit;
            // for each element of the set, we iterate over the current projection 
            // and build the projected string by appending one by one the projected chars
            for (vector<int>::iterator projit = g[i].begin(); projit != g[i].end(); projit++)
                s.push_back(inputel[*projit]);
        
            // cout << "String to be added to current: " << s << endl;
            // add the string corresponding to the current input string to the current set
            current.insert(s);
        }

        // the ith set will be formed by strings of the input projected according to the ith function
        mapped.push_back(current);
    }
    
    return mapped;
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



// bruteForce takes a string query, and int distance r and the input set and checks whether 
// the query is at distance at least r from the whole input.
bool bruteForce(string q, int r, set<string> &input)
{
    // for every input string, if the query is close to it then return false
    for (set<string>::iterator it = input.begin(); it != input.end(); it++)
    {
        // cout << "Looking for distance between q=" << q << " and input element =" << *it << endl;
        // cout << hdist(q,*it) << endl;
        if(hdist(q, *it) < r)
            return false;
    }
    
    return true;
}


const vector<char> alph = {'A', 'C', 'G', 'T'};



string recBrute(string prefix, int k, set<string> &given)
{
    if(k==0)
    {   
        // cout << "String considered is " << prefix << endl;
        // if we generated a string of correct length which is not in the given set, we are done
        if(find(given.begin(), given.end(), prefix) == given.end())
            return prefix;
        else
            return "";
    }
    else
    {
        for (int i = 0; i < alph.size(); i++)
        {
            char c = alph[i];
            string q = recBrute(prefix + c, k-1, given);
            if (q != "")
                return q;
             
        }
        
    }

    return "";

}

// given a set of strings of the same length, find one that does not belong
// with a brute force (alphabetic checking) approach
string findNew(set<string> &given)
{
    int n = given.begin()->length(); // length of strings in projected space
    string q;

    int total = pow(alph.size(),n);

    cout << "There are " << given.size() << " strings in the set, all of length " << n << endl;
    cout << "Power is " << total  << endl;

    if(given.size() == total)
    {   
        cout << "STRING SET IS FULL!" << endl;
        return "x";
    }

    // we iterate over all strings in alphabetical order
    return recBrute("", n, given);
    
}


void recBruteEnum(string prefix, int k, set<string> &given, vector<string> &diff)
{
    if(k==0)
    {   
        // cout << "String considered is " << prefix << endl;
        // if we generated a string of correct length which is not in the given set, we are done
        if(find(given.begin(), given.end(), prefix) == given.end())
            diff.push_back(prefix);
        
        return;
    }
    else
    {
        for (int i = 0; i < alph.size(); i++)
        {
            char c = alph[i];
            recBruteEnum(prefix + c, k-1, given, diff);
        }   
    }
}


// given a set of strings of the same length, find the ones that do not belong
// with a brute force (alphabetic checking) approach
vector<string> enumNew(set<string> &given)
{
    int n = given.begin()->length(); // length of strings in projected space
    string q;
    vector<string> diff = {};

    int total = pow(alph.size(),n);

    cout << "There are " << given.size() << " strings in the set, all of length " << n << endl;
    cout << "Power is " << total  << endl;

    if(given.size() == total)
    {   
        cout << "STRING SET IS FULL!" << endl;
        return {};
    }

    // we iterate over all strings in alphabetical order
    recBruteEnum("", n, given, diff);
    
    return diff;
}


// given a set of strings of the same length, find one that does not belong
// with a random generation approach
// we need the alphabet (size) to do so
string findNewOld(set<string> &given)
{
    int count= 0;
    int n = given.begin()->length(); // length of strings in projected space
    string q;

    cout << "There are " << given.size() << " strings in the set, all of length " << n << endl;
    cout << "Power is " << pow(alph.size(),n) << endl;

    if(given.size() == pow(alph.size(),n))
    {   
        cout << "STRING SET IS FULL!" << endl;
        return "x";
    }

    do
    {
        count++;
        q = "";
        for (int i = 0; i < n; i++)
        {
            int charind = rand() % alph.size();
            q.push_back(alph[charind]);
        }
    } while (find(given.begin(), given.end(), q) != given.end() && count  < 1000000);
    
    if(count == 1000000)
    {
        cout << "count expired" << endl;
        return "";
    }
        

    cout << "Distinct string " << q << " found in " << count << " tests." << endl;
    
    return q;
}


// given a string, a total length and a set of positions, complete the string randomly with the
// input string characters at the given positions
string extendString(string q, int L, vector<int> pos)
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



string findQueryMult(int L, int m, int k, set<string> &input)
{
    // initialization of vector of hash functions
    // each element g[i] of g is a hash function, composed of at most m projections    
    vector<vector<int>> g;
    for (int i = 0; i < k; i++)
        g.push_back(randomProjections(L, m));


    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
        cout << endl;
    }
    cout << endl;


    
    // identify indices which are overlapping and remove them from other components
    // SHOULD WE DO THIS IN A MORE CONTROLLED MANNER? LIKE CHOOSING THE COMPONENT WITH MORE
    // INDICES AND DELETING FROM THAT ONE
    // go through each hash function
    for (int i = 0; i < g.size(); i++)
    {   
        // for every index taken by the current hash function
        for(int j=0; j<g[i].size(); j++ )
        {
            // look for g[i][j] in the other vectors (hash functions) following the ith, and remove them if they are there
            for(int h=i+1; h< g.size(); h++)
            {
                for (int f = 0; f < g[h].size(); f++)   
                {   
                    // if in fact index f of function h is equal to g[i][j], erase it FROM WHICH FUNCTION?
                    if(g[h][f]==g[i][j])
                        g[h].erase (g[h].begin()+f);
                }
                
            }
        }
    }


    // remove possible empty functions
    for (int i = 0; i < g.size(); i++)
    {
        if (g[i].size() == 0)
        {
            g.erase(g.begin() + i);
        }
        
    }
    

    cout << endl << "After cleaning multiple indices: " << endl;
    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
       
        cout << endl;
    }
    cout << endl;

        

    target mapped = mapInput(input, g);

    // for (int i = 0; i < mapped.size(); i++)
    // {
    //     cout << i << "th projected set is: {" << endl;
    //     for (set<string>::iterator s = mapped[i].begin(); s != mapped[i].end(); s++) 
    //     {                                                        
    //         cout << *s << ', '; 
    //     }
    //     cout << "}" << endl << endl;
    // } 


    // cout << "Input has been mapped. Now proceeding to find strings. " << endl; 
    // ----------------------------------OK UP TO HERE----------------------------------

    vector<string> q;
    
    vector<int> pos;
    string merged;


    // for every piece of mapped input, find the strings that do not belong
    // if any of them are empty, return the empty string. Otherwise, create a 
    // string vector containing all strings found this way
    for (int i = 0; i < g.size(); i++)
    {   
        string s = findNew(mapped[i]);
        if(s == "")
            return "";
        
        if(s == "x")
            return "x";

        q.push_back(s);
    }
    
    cout << "New strings have been found for each target space. Specifically: ";
    for (int i = 0; i < q.size(); i++)
    {
        cout << "\t" << q[i];
    }
    cout << endl;
    

    // merge the obtained strings into one string according to the positions of the (ordered) arrays
    // while there are at least two strings, merge the first two
    while (q.size() > 1)
    {
        // pop from the two stacks simultaneously so that the vectors are correct
        string s1 = q.back();
        q.pop_back();
        vector<int> pos1 = g.back();
        g.pop_back();

        string s2 = q.back();
        q.pop_back();
        vector<int> pos2 = g.back();
        g.pop_back();

        pos.clear();
        merged.clear();

        // cout << "At this iteration, s1=" << s1 << ", s2=" << s2 << endl;

        // we proceed until we empty both position arrays
        while (pos1.size()>0 && pos2.size()>0)
        {   
            // ADD CONDITION WHEN ONE OF THEM IS EMPTY TO JUST GO FULL ON THE OTHER ONE
            // cout << "Inside while loop" << endl;
            if(pos1.back() >= pos2.back())
            {
                // cout << "last element of pos1 is bigger" << endl;
                pos.push_back(pos1.back());
                pos1.pop_back();

                merged.push_back(s1.back());
                s1.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            else
            {
                // cout << "last element of pos2 is bigger" << endl;
                pos.push_back(pos2.back());
                pos2.pop_back();

                merged.push_back(s2.back());
                s2.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        } 

        while(pos1.size()>0)
        {
            // cout << "only elements of pos1 are left" << endl;
            pos.push_back(pos1.back());
            pos1.pop_back();

            merged.push_back(s1.back());
            s1.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        }
        
        while(pos2.size()>0)
        {
            // cout << "only elements of pos2 are left" << endl;
            pos.push_back(pos2.back());
            pos2.pop_back();

            merged.push_back(s2.back());
            s2.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;

        }
        
        // we need to reverse the array of positions and the string
        reverse(merged.begin(), merged.end());
        sort(pos.begin(), pos.end());

        // once we are done, we insert pos, merged back into the corresponding vectors
        g.push_back(pos);
        q.push_back(merged);

        cout << endl << "Arrays are currently " << endl;
        cout << "Hash functions: " << endl;
        for (int i = 0; i < g.size(); i++)
        {
            // cout << endl << i << "th hash function is: " << endl;
            cout << "g_" << i << ": ";
            for (int j = 0; j < g[i].size(); j++)
                cout << "\t" << g[i][j];
            cout << endl;
        }
        cout << endl;

        cout << "Strings: ";
        for (int i = 0; i < q.size(); i++)
        {
            cout << "\t" << q[i]  << endl;
        }
        cout << endl;   
    }   


    merged = q[0];
    pos = g[0];

    
    string extended = extendString(merged, L, pos);
    cout << "Extended string is " << extended << endl;
 
    return extended; //extendString(merged, L, pos);
}


// given a vector of two strings, and the corresponding position arrays, merge the strings 
pair<string, vector<int>> mergeStrings(vector<string> q, vector<vector<int>> g)
{
    vector<int> pos;
    string merged;

    // merge the obtained strings into one string according to the positions of the (ordered) arrays
    // while there are at least two strings, merge the first two
    while (q.size() > 1)
    {
        // pop from the two stacks simultaneously so that the vectors are correct
        string s1 = q.back();
        q.pop_back();
        vector<int> pos1 = g.back();
        g.pop_back();

        string s2 = q.back();
        q.pop_back();
        vector<int> pos2 = g.back();
        g.pop_back();

        pos.clear();
        merged.clear();

        // cout << "At this iteration, s1=" << s1 << ", s2=" << s2 << endl;

        // we proceed until we empty both position arrays
        while (pos1.size()>0 && pos2.size()>0)
        {   
            // ADD CONDITION WHEN ONE OF THEM IS EMPTY TO JUST GO FULL ON THE OTHER ONE
            // cout << "Inside while loop" << endl;
            if(pos1.back() >= pos2.back())
            {
                // cout << "last element of pos1 is bigger" << endl;
                pos.push_back(pos1.back());
                pos1.pop_back();

                merged.push_back(s1.back());
                s1.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            else
            {
                // cout << "last element of pos2 is bigger" << endl;
                pos.push_back(pos2.back());
                pos2.pop_back();

                merged.push_back(s2.back());
                s2.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        } 

        while(pos1.size()>0)
        {
            // cout << "only elements of pos1 are left" << endl;
            pos.push_back(pos1.back());
            pos1.pop_back();

            merged.push_back(s1.back());
            s1.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        }
        
        while(pos2.size()>0)
        {
            // cout << "only elements of pos2 are left" << endl;
            pos.push_back(pos2.back());
            pos2.pop_back();

            merged.push_back(s2.back());
            s2.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;

        }
        
        // we need to reverse the array of positions and the string
        reverse(merged.begin(), merged.end());
        sort(pos.begin(), pos.end());

        // once we are done, we insert pos, merged back into the corresponding vectors
        g.push_back(pos);
        q.push_back(merged);

        cout << endl << "Arrays are currently " << endl;
        cout << "Hash functions: " << endl;
        for (int i = 0; i < g.size(); i++)
        {
            // cout << endl << i << "th hash function is: " << endl;
            cout << "g_" << i << ": ";
            for (int j = 0; j < g[i].size(); j++)
                cout << "\t" << g[i][j];
            cout << endl;
        }
        cout << endl;

        cout << "Strings: ";
        for (int i = 0; i < q.size(); i++)
        {
            cout << "\t" << q[i]  << endl;
        }
        cout << endl;   
    }   

    merged = q[0];
    pos = g[0];

    return make_pair(merged, pos);
}




// given a vector of two strings, and the corresponding position arrays and absolute overlap between them, merge the strings 
// NOTE: it is assumed that the strings are coherent on the overlap
pair<string, vector<int>> mergeTwoOverlapping(vector<string> q, vector<vector<int>> g)
{
    vector<int> pos;
    string merged;

    // merge the obtained strings into one string according to the positions of the (ordered) arrays
    // while there are at least two strings, merge the first two
    while (q.size() > 1)
    {
        // pop from the two stacks simultaneously so that the vectors are correct
        string s1 = q.back();
        q.pop_back();
        vector<int> pos1 = g.back();
        g.pop_back();

        string s2 = q.back();
        q.pop_back();
        vector<int> pos2 = g.back();
        g.pop_back();

        pos.clear();
        merged.clear();

        // cout << "At this iteration, s1=" << s1 << ", s2=" << s2 << endl;

        // we proceed until we empty both position arrays
        while (pos1.size()>0 && pos2.size()>0)
        {  
            // if the position is overlapping, move onto the next for one of them
            // note that this cannot happen twice in a row, as in the position vectors all elements are distinct
            if(pos1.back() == pos2.back())
                pos2.pop_back();
            

            // cout << "Inside while loop" << endl;
            if(pos1.back() > pos2.back())
            {
                // cout << "last element of pos1 is bigger" << endl;
                pos.push_back(pos1.back());
                pos1.pop_back();

                merged.push_back(s1.back());
                s1.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            else
            {
                // cout << "last element of pos2 is bigger" << endl;
                pos.push_back(pos2.back());
                pos2.pop_back();

                merged.push_back(s2.back());
                s2.pop_back();
                // cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
            }
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        } 

        while(pos1.size()>0)
        {
            // cout << "only elements of pos1 are left" << endl;
            pos.push_back(pos1.back());
            pos1.pop_back();

            merged.push_back(s1.back());
            s1.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;
        }
        
        while(pos2.size()>0)
        {
            // cout << "only elements of pos2 are left" << endl;
            pos.push_back(pos2.back());
            pos2.pop_back();

            merged.push_back(s2.back());
            s2.pop_back();
            cout << "Now, s1=" << s1 << ", s2=" << s2 << endl;

        }
        
        // we need to reverse the array of positions and the string
        reverse(merged.begin(), merged.end());
        sort(pos.begin(), pos.end());

        // once we are done, we insert pos, merged back into the corresponding vectors
        g.push_back(pos);
        q.push_back(merged);

        cout << endl << "Arrays are currently " << endl;
        cout << "Hash functions: " << endl;
        for (int i = 0; i < g.size(); i++)
        {
            // cout << endl << i << "th hash function is: " << endl;
            cout << "g_" << i << ": ";
            for (int j = 0; j < g[i].size(); j++)
                cout << "\t" << g[i][j];
            cout << endl;
        }
        cout << endl;

        cout << "Strings: ";
        for (int i = 0; i < q.size(); i++)
        {
            cout << "\t" << q[i]  << endl;
        }
        cout << endl;   
    }   

    merged = q[0];
    pos = g[0];

    return make_pair(merged, pos);
}




// enumerate the possible solutions for k=2
vector<string> enumQueryTwo(int L, int m, int k, set<string> &input)
{
    // initialization of vector of hash functions
    // each element g[i] of g is a hash function, composed of at most m projections    
    vector<vector<int>> g;
    for (int i = 0; i < k; i++)
        g.push_back(randomProjections(L, m));


    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
        cout << endl;
    }
    cout << endl;

    
    target mapped = mapInput(input, g);

    // enumerate the strings not belonging to the sets
    // if no solution is found for a set, return empty 
    vector<vector<string>> q;
    for(int i = 0; i < g.size(); i++)
    {
        vector<string> s = enumNew(mapped[i]);
        if(s.size() == 0)
            return {};

        q.push_back(s);           
    }


    // now we need to resolve the possible overlaps. 
    // 1) find the indices where g_1, g_2 overlap 
    //      the overlaps are represented as pairs of ints, expressing the relative
    //      positions inside the functions which correspond to an overlap
    //      that is, (i,j) is in overlap iff g[0][i]= g[1][j]

    vector<pair<int,int>> relativeOverlap;
    vector<int> absOverlap;

    for(int i = 0; i< g[0].size(); i++)
    {
        int curr = g[0][i]; 
        for(int j = 0; j< g[1].size(); j++)
        {
            if(g[1][j] == curr)
            {
                relativeOverlap.push_back(make_pair(i,j));
                absOverlap.push_back(curr);
                cout << "The two functions overlap at index " << curr << " (relative indices are " << i << " and " << j << ")"<< endl;
            }         
        }
    }

    cout << "New strings have been found for each target space. Specifically: ";
    for (int i = 0; i < q.size(); i++)
    {
        cout << i << "th target space: ";
        for (int j = 0; j < q[i].size(); j++)
            cout << "\t" << q[i][j];
        
        cout << endl;
    }
    cout << endl;
    
    // we need to select strings from different spaces to merge them with each other
    
    // 2) go through the overlaps and only build combined strings which share the same 
    //      characters at the overlapping positions

    // we do this by manually checking whether every string has a coherent overlap,
    // and merging the other parts of the strings (i.e. removing the overlaps from one of the two)

    // note: q[0] = queries enumerated for g_0, q[1] = queries enumerated for g_1

    // validQueries will be the same size as q[0]; for string i of q[0], we have validQueries[i]
    // containing a vector of strings which are compatible with string i on the overlapping positions
    vector<vector<string>> validQueries;
    vector<string> currentValid;

    // for every string in the first set
    for(int i = 0; i < q[0].size(); i++)
    {
        currentValid.clear();

        string first = q[0][i];
        // for every string in the second set
        for(int j = 0; j< q[1].size(); j++)
        {
            string second = q[1][j];

            bool coherent = true;

            // for every overlap position, if they don't coincide coherent becomes false
            int o = 0;
            while(coherent && o < relativeOverlap.size())
            {
                pair<int,int> indpair = relativeOverlap[o];

                if(first[indpair.first] != second[indpair.second])
                    coherent = false;
                o++;
            }

            if(coherent)
                currentValid.push_back(second);
        }

        validQueries.push_back(currentValid);
    }
    

    // we will have a vector of merged queries. Each element of the vector will be 
    // created by calling mergeStrings on a pair in validQueries (expressed in toMerge).
    vector<pair<string,vector<int>>> mergedQueries;
    vector<string> toMerge;

    // for every string in q[0]
    for(int i=0; i<q[0].size(); i++)
    {   
        toMerge.clear();
        toMerge.push_back(q[0][i]);

        // for every coherent string q[0] has been matched with
        for(int j = 0; j < validQueries[i].size(); j++)
        {
            cout << "Trying to merge " << q[0][i] << " and " << validQueries[i][j] << endl;
            // toMerge must be q[0][i], validQueries[i][j]
            toMerge.push_back(validQueries[i][j]);
            mergedQueries.push_back(mergeTwoOverlapping(toMerge,g));
        }        
    }
     

    vector<string> extendedQueries;

    // now we go through every merged pair of queries and extend it
    for(int i = 0; i < mergedQueries.size(); i++)  
    {
        string extended = extendString(mergedQueries[i].first, L, mergedQueries[i].second);
        cout << "Extended string is " << extended << endl;
        extendedQueries.push_back(extended);
    }

 
    return extendedQueries; //extendString(merged, L, pos);
}




// NOT WORKING YET
vector<string> enumQueryMult(int L, int m, int k, set<string> &input)
{
    // initialization of vector of hash functions
    // each element g[i] of g is a hash function, composed of at most m projections    
    vector<vector<int>> g;
    for (int i = 0; i < k; i++)
        g.push_back(randomProjections(L, m));


    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
        cout << endl;
    }
    cout << endl;


    
    // identify indices which are overlapping and remove them from other components
    // SHOULD WE DO THIS IN A MORE CONTROLLED MANNER? LIKE CHOOSING THE COMPONENT WITH MORE
    // INDICES AND DELETING FROM THAT ONE
    // go through each hash function
    for (int i = 0; i < g.size(); i++)
    {   
        // for every index taken by the current hash function
        for(int j=0; j<g[i].size(); j++ )
        {
            // look for g[i][j] in the other vectors (hash functions) following the ith, and remove them if they are there
            for(int h=i+1; h< g.size(); h++)
            {
                for (int f = 0; f < g[h].size(); f++)   
                {   
                    // if in fact index f of function h is equal to g[i][j], erase it FROM WHICH FUNCTION?
                    if(g[h][f]==g[i][j])
                        g[h].erase (g[h].begin()+f);
                }
                
            }
        }
    }


    // remove possible empty functions
    for (int i = 0; i < g.size(); i++)
    {
        if (g[i].size() == 0)
        {
            g.erase(g.begin() + i);
        }
        
    }
    

    cout << endl << "After cleaning multiple indices: " << endl;
    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
       
        cout << endl;
    }
    cout << endl;

        

    target mapped = mapInput(input, g);


    // queries are in a vector of vectors of strings
    vector<vector<string>> q;

    // for every piece of mapped input, find the strings that do not belong
    // if any of them are empty, return the empty string. Otherwise, create a 
    // string vector containing all strings found this way
    for (int i = 0; i < g.size(); i++)
    {   
        vector<string> s = enumNew(mapped[i]);

        // if no words were found , return the empty vector
        if(s.size()== 0)
            return {};
        
        // if any of them had filled the set, return a single vector with an "x"
        // if(s[0]== "x")
        //     return {{"x"}};

        q.push_back(s);
    }
    
    cout << "New strings have been found for each target space. Specifically: ";
    for (int i = 0; i < q.size(); i++)
    {
        cout << i << "th target space: ";
        for (int j = 0; j < q[i].size(); j++)
            cout << "\t" << q[i][j];
        
        cout << endl;
    }
    cout << endl;
    
    // we need to select strings from different spaces to merge them with each other
    
    

    pair<string, vector<int>> query = mergeStrings(q[0], g);

    vector<string> extQueries;

    string extended = extendString(query.first, L, query.second);
    cout << "Extended string is " << extended << endl;

    extQueries.push_back(extended);
 
    return extQueries; //extendString(merged, L, pos);
}




string findQuery(int L, int m, int k, set<string> &input)
{
    // initialization of vector of hash functions
    // each element g[i] of g is a hash function, composed of at most m projections    
    vector<vector<int>> g;
    for (int i = 0; i < k; i++)
        g.push_back(randomProjections(L, m));


    for (int i = 0; i < g.size(); i++)
    {
        // cout << endl << i << "th hash function is: " << endl;
        cout << "g_" << i << ": ";
        for (int j = 0; j < g[i].size(); j++)
            cout << "\t" << g[i][j];
        cout << endl;
    }
    cout << endl;
        

    target mapped = mapInput(input, g);

    // for (int i = 0; i < mapped.size(); i++)
    // {
    //     cout << i << "th projected set is: {" << endl;
    //     for (set<string>::iterator s = mapped[i].begin(); s != mapped[i].end(); s++) 
    //     {                                                        
    //         cout << *s << ' '; 
    //     }
    //     cout << "}" << endl << endl;
    // }   

    string q = findNew(mapped[0]);

    return extendString(q, L, g[0]);
}



string randomString(int L)
{
    string s = "";
    for (int i = 0; i < L; i++)
        s.push_back(alph[rand() % alph.size()]);
    
    return s;
}


int main()
{
    int N,L,r,m,k;
    string W;
    set<string> input;
    time_t begin, end;

    // set rand seed according to clock
    srand(time(NULL));

    // cout << hdist("ACACGT", "ACCCGT") << endl;

    cout << "Insert length of input strings: ";
    cin >> L;
    cout << endl;

    cout << "Insert required distance from input: ";
    cin >> r;
    cout << endl;

    // for now, set r= L/2
    // r = (int) L/2;


    cout << "Insert size of target space of LSH: ";
    cin >> m;
    cout << endl;
    

    cout << "Insert number of hash functions for LSH: ";
    cin >> k;
    cout << endl;

    // for now, just one hash function
    //k = 1; 


    int trials;
    cout << "How many trials do you want to perform? ";
    cin >> trials;


    ifstream myfile;
    myfile.open("shorterY.txt");

    ofstream outputfile;
    outputfile.open("bruteShorterOutput.txt", ios_base::app);



    if (myfile.is_open() && outputfile.is_open())
    {
        while(!myfile.eof())
        {
            char c = myfile.get();
            if (myfile.eof()) break;
            W.push_back(c);
        }
    }        

    
    N= W.length();
    cout << "String length is " << N << endl;

    for (int i = 0; i <= N-L; i++)
    {
        input.insert(W.substr(i,L));
    }
    
    // set<string>::iterator it;
    // cout << "Input contains:";
    // for (it=input.begin(); it!=input.end(); ++it)
    //     cout << ' ' << *it;
    // cout << endl;

    cout << "Dealing with " << input.size() << " strings "<< endl;






    // writing to output
    outputfile << "================== \tL=" << L << "\tr=" << r << "\tm=" << m << "\tk="<<k<< " ==================" << endl;
    //outputfile << "Parameters: \t L=" << L << "\tr=" << r << "\tm= " << m << "\tk="<<k<< endl;
    outputfile << "Original input text has length N=" << N << " and the number of input strings is " << input.size() << endl<<endl;
    outputfile << "Number of trials: " << trials<<endl<<endl;

    int succ = 0;
    int fail = 0;
    int count = 0;

//hello giubini pastalinguini

    while(count < trials)
    {   
        cout << endl <<  "Trial no. " << count+1 << endl;
        
        begin = time(nullptr);
        string extq = findQueryMult(L,m,k,input);
        end = time(nullptr);

        if(extq != "" && extq != "x")
        {
            // cout << "Found query q = " << q << endl;
            // cout << "Our hash function is g: ";
            // for (int j = 0; j < g[0].size(); j++)
            // cout << "\t" << g[0][j];
            // cout << endl;

            // cout << "Extended string is " << extq << endl;
            // cout << "Now checking with brute force." << endl;

            bool found = bruteForce(extq, r, input);

            if (found)
            {
                cout << "SUCCESS!!" << endl;
                cout << "String found is " << extq << " in time " << end-begin << endl;
                outputfile << "Found string " << extq << " in time " << end-begin << endl;
                succ++;
            }
            else
            {
                cout << "FAILURE in time " << end-begin << endl;
                fail++;
            }
        }
        else
        {
            cout << "No different string has been found."<<endl;
            fail++;
            if(extq == "x")
            {
                cout << "String set was full." << endl;
                outputfile << "No string found because at least one projected string set was full" << endl;
            }
        }

        count++;
    }

        // initialization of vector of hash functions
        // each element g[i] of g is a hash function, composed of at most m projections
        // vector<vector<int>> g;
        // for (int i = 0; i < k; i++)
        //     g.push_back(randomProjections(L, m));


        // for (int i = 0; i < g.size(); i++)
        // {
        //     cout << endl << i << "th hash function is: " << endl;
        //     cout << "g_" << i << ": ";
        //     for (int j = 0; j < g[i].size(); j++)
        //     cout << "\t" << g[i][j];
        // }
        // cout << endl;
            

        // target mapped = mapInput(input, g);

        // for (int i = 0; i < mapped.size(); i++)
        // {
        //     cout << i << "th projected set is: {" << endl;
        //     for (set<string>::iterator s = mapped[i].begin(); s != mapped[i].end(); s++) 
        //     {                                                        
        //         cout << *s << ' '; 
        //     }
        //     cout << "}" << endl << endl;
        // }   

        // string q = findNew(mapped[0]);

    cout << "Trials are " << trials << endl;
    cout << "Successes are " << succ<< endl;

    outputfile << "Number of successes with LSH: " << succ << endl<<endl;


    count = 0;
    succ = 0;
    while (count < trials)
    {
        begin = time(nullptr);
        string extq = randomString(L);
        end = time(nullptr);

        bool found = bruteForce(extq, r, input);

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
    
    cout << "Random successes are " << succ<< endl;
    outputfile << "Number of random successes: " << succ << endl;
    outputfile << endl<<endl<<endl;


    Beep(600,1000);

    myfile.close();
    outputfile.close();

    return 0;
}


// test length < 13 
// controlled test on shorter strings 

int oldmain()
{
    set<string> input;
    vector<vector<int>> g;
    string W;
    int N;

    vector<int> pos = {0,1,2,3,4,5,6,7,8,9,10};
    int L = 20;

    srand(time(NULL));

    ifstream myfile;
    myfile.open("test.txt");

    ofstream outputfile;
    outputfile.open("testoutput.txt", ios_base::app);

    
    if (myfile.is_open() && outputfile.is_open())
    {
        while(!myfile.eof())
        {
            char c = myfile.get();
            if (myfile.eof()) break;
            W.push_back(c);
        }
    }        

    
    // N= W.length();
    // cout << "String length is " << N << endl;

    // for (int i = 0; i <= N-L; i++)
    //     input.insert(W.substr(i,L));
    
    L =3;
    input = {"AAC", "AAA", "AAG", "AAT", "ACA", "ACG", "ACC", "ACT", "AGA"};

    set<string>::iterator it;
    cout << "Input contains:";
    for (it=input.begin(); it!=input.end(); ++it)
        cout << ' ' << *it;
    cout << endl;

    cout << "Dealing with " << input.size() << " strings "<< endl;
    cout << "Room for different words is " << pow(4,L) << endl;



    string q = findNew(input);
    cout << "Found string " << q << endl;
    if(find(input.begin(), input.end(), q) == input.end())
        cout << "String does not belong to set" << endl;
    else
        cout << "ERROR!!!" << endl;

    

    // set<string> given = {"AA", "AC", "CC", "GA", "TC", "GT", "TT"};
    // string q = findNew(given);

    // if (find(given.begin(), given.end(), q) == given.end())
    // {
    //     cout << "String q not found in set" << endl;
    // }
    // else
    //     cout << "String q belongs to set" << endl;

    
    myfile.close();
    outputfile.close();
    return 0;
}
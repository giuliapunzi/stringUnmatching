#include <iostream> 
#include <fstream>
#include <string>
#include <set>

using namespace std;


int main()
{
    ifstream myfile;
    myfile.open("all_seqs.fa.tar");
    int count = 0;

    if (myfile.is_open())
    {
        while(!myfile.eof())
        {
            char c = myfile.get();
            if(c == '>') count++;

            if (myfile.eof()) break;
        }
    }

    // cout << "Number of genomes is " << count << endl;

    myfile.close();
    return 0;
}

int preprocessmain()
{
    set<char> alph = {};

    ifstream myfile;
    myfile.open("all_seqs.fa.tar");

    ofstream outputfile;
    string filename = "genome";
    string extension = ".txt";
    string gcount = "1";
    string currfile = filename+gcount+extension;

    int count = 0;
    int countlim = 2;

    // Build genome according to input file
    if (myfile.is_open())
    {
        while(!myfile.eof())
        {
            outputfile.open(currfile, ios_base::app);
            // get the first line and write it 
            string line; 
            getline(myfile, line);
            outputfile<<line<< endl;
            // cout << "Line is: " << line << endl;

            while (count <countlim)
            {
                char c = myfile.peek();
                // c = toupper(c);              

                if(c== '>')
                    count++;
                else
                {
                    c = myfile.get();
                    c = toupper(c);

                    if (myfile.eof()) break;

                    // if not in alphabet and not newline, add it
                    if(c != '\n' )
                    {
                        outputfile << c;
                        if(alph.find(c) == alph.end())
                        {
                            alph.insert(c);
                            // cout << "Added char " << c << " to alphabet, found in file " << count+1 << endl;
                        }
                    }
                }
                
            }
            
            // once we get here, we have reached another '>'
            // we increase the counter, and change outputfile
            countlim++;
            outputfile.close();
            gcount = to_string(count);
            currfile = filename+gcount+extension;
        }

    }  

    // cout<< "Alphabet is: { ";
    // for (set<char>::iterator it = alph.begin(); it != alph.end(); it++)
    // {
    //     cout << *it << ", ";
    // }
    // cout << " }"<<endl;

    // cout << "Total number of files is " << count-1 << endl;

    myfile.close();

    return 0;
}
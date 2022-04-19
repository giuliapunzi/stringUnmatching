# stringUnmatching

## USAGE 

The method aims to generate 32-grams which are suitably distant from every substring in the input. (the method works for generic Q-grams of any length, but the current implementation works only with Q=32).

Input: “dataset.txt” which contains sequences of bases 
Characters other than ACGT are ignored as for the approach it is crucial that the alphabet does not contain rare characters. 
At the moment the alphabet is hard-wired to 4: 00=A, 01=C, 10=G, 11=T

compile with ```g++ -mbmi2 -std=c++2a```
(g++ 9 or later is required for the c++2a flag)

The folder Templ-gen/ contains the generation method and a sequential testing procedure for the quality of the 32-grams.
The folder GPU-matching/ contains the efficient quality testing procedure (and has its separate readme).

## 1- Generating the templates (candidates)
Run the function: 
Temp-gen/template-generation.cpp::template_generation(char* input_filename, char* output_log_filename, char* output_templates_filename)

input_filename : text file with the sequence of bases (FASTA)
output_log_filename : will contain useful logs
output_templates_filename : binary file containing the templates generated


To test, you can compile with:
```g++ -mbmi2 -std=c++2a template-generation.cpp -o tempgen```
and use the dummy main with hard-coded parameters ( ./tempgen ).

Output: the binary file output_templates_filename will contain the templates, i.e., Qgrams which should be good candidates for having high distance from any Qgram in the dataset.

The binary file in output is a sequence of uint64, representing 32-grams: each 2 bits of the uint64 correspond to a base (00=A, 01=C, 10=G, 11=T).

Some parameters are hard-coded in Cpp/template-generation.cpp but should be updated according to the dataset size as they can heavily impact the result (we include some examples below of what we found to work experimentally): 
Q = length of Qgrams; must always be set to 32 in this implementation
N_hash_fctns = number of projections
target_size = size of image space of projections; that is, how many indices the projections contain (equal for all functions)
UNIVERSE_SIZE = size of the universe of the projected Qgrams; MUST BE SET TO 4^target_size
SEED = seed for random functions is fixed for trial reproducibility; may be changed as needed

Examples that we experimentally found reasonable:
Gastro (2.8B 32-grams)
N_hash_fctns = 6
target_size = 13
Templates obtained in output: 157,222
Blood (200M 32-grams)
N_hash_fctns = 6
target_size = 11
Templates obtained in output: 600,015

**NOTE**: the parameters N_hash_fctns and especially target_size strongly depend on the size and distribution of the dataset, as they influence the number of templates: a very small target_size results in zero templates, while a larger one can result in too many, not significant, templates.

Some info of what is done by this step:
A set of random projections are generated by the function build_functions().
Each projection is a set of indices within each Qgram, e.g., the 1st 3rd and 4th position. Projecting the Qgram ACGACG on that projection would give AGA.
The templates generated are Qgrams which are “safe” according to each projection. “safe” meaning that taken any projection of any Qgram in the input dataset, the template differs from it by at least one character.
(E.g., in the example above, a template projected on 1st 3rd and 4th position must not give AGA if the Qgram above was in the dataset)
In order not to lose information, every Qgram position is guaranteed to be covered by at least one function.



## 2- Checking the templates

**NOTE**: this is a sequential implementation for toy cases. 
To do this step efficiently, use instead the GPU-based implementation in the “GPU-matching” folder, which also has both hamming and edit distance. 

The goal here is to actually test the templates against the input, and only retain those with a good distance (for a distance threshold k, we want the template to have distance k or more from EVERY substring of the input genome). This implementation considers hamming distance.

Run the function: Templ-gen/check-templates-from-file.cpp::map_and_check(int minimum_dist, const char* input_filename, const char* Qgrams_filename, const char* templates_filename, const char* output_log_filename, const char* good_templates_filename)

minimum_distance: distance threshold
input_filename : input text file (FASTA format)
Qgrams_filename : binary file that will be filled with the qgrams of the input -to be created-
templates_filename : binary file containing the templates (output of previous step)
output_log_filename : log file -to be created-
good_templates_filename : binary file containing the templates that pass the distance check -to be created-

To test, you can compile with:
```g++ -mbmi2 -std=c++2a check-templates-from-file.cpp -o check```
and use the dummy main with hard-coded parameters ( ./check ).

### Exmaple usage 

In the folder “Templ-gen/”, one can run the following functions:

template-generation::template_generation(“Blood-prefix.fsa”, “gen-log.txt”, “blood-templates.bin”)

check-templates-from-file::map_and_check(9, “Blood-prefix.fsa”, “blood-qgrams.bin”, “blood-templates.bin”, “check_log”, “far_templates.bin”)

The file “far_templates.bin” will contain all 32-grams whose hamming distance is at least 9 from every 32-grams in the input file “Blood-prefix.fsa”.

For a direct demo of the approach using the provided sample file, it is possible to run, in the Templ-gen/ folder:

```g++ -mbmi2 -std=c++2a template-generation.cpp -o tempgen```
```g++ -mbmi2 -std=c++2a check-templates-from-file.cpp -o check```
```./tempgen```
```./check```

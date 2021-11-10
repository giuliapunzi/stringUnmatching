String Un-Matching
==================

Installation
------------

### Requirements 

To compile (and run) this software you need

 - A NVIDIA GPU with compute capability at least 6.0
 - CUDA/CUDA Toolkit >= 10.1
 - GCC/G++ >= 7.5
 - CMake >= 3.18

Optionally, also `python3`, `tqdm`, `numpy` and `pandas` to compute running and result analytics.

### Conda Environment

You may install the requirement by creating a virual environment using [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/). This is also useful to install the software on a machine *without root privileges*. Check if `conda` is already installed with `which conda`, or install it (e.g., in your home directory) running
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
and following the instructions.

Once `conda` is installed, you can install all the requirements by running
```
conda env create -f conda/environment.yml
```
in the project directory. This will create an environment named `strum`, which can be activated with the command
```
conda activate strum
```
*(**Note:** if you choose to use `conda`, remember to always run this command after opening any new shell, before executing any of the project-related commands and executables)*.

### Compile the project

Run the following commands to compile all the targets
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -B build
cmake --build build -j 8
```

Project Executables
-------------------

### Main targets

Once compiled, in the `./build` folder you can find the following executables:

 1. `convert`, which is an utility program to convert FASTA strings to/from compressed bytes. By defaults, converts data from standard input and outputs the result in the standard output. For example, you can convert the FASTA file `sequence.fa` in binary using the command
 ```
 cat sequence.fa | convert > sequence.bin
 ```
 More info are availabe by running `convert -h`.

 2. `hamming`, which computes the minimum Hamming distance of multiple templates (of exactly 32 nucleotides) in a given FASTA sequence, which must be provided as the first argument (or after the `-s`/`--sequence` option). By default, it assumes that the sequence is already binarized (using `convert`), unless the `-f`/`--fasta` option is given. The templates are read from standard input or from the trailing CLI argument in FASTA format, one for each line, unless the `-b`/`--binary` option is given, and the computed Hamming distance is printed in the standard output. For example, the following command computes the distances in the binary `template.bin` file and stores the result in `output.txt`
 ```
 cat templates.bin | hamming sequence.fa -f -b > output.txt
 ```
 More info are availabe by running `hamming -h`.

### Tracking execution progress

For larger files, it may be usefull to keep track of the progress of the conversion and/or computation. For this, you may use [`tqdm`](https://tqdm.github.io/) as follows
```
export N=100000
dd if=/dev/urandom bs=8 count=$N | hamming sequence.bin -b | tqdm --total=$N > output.txt
```
This example generates the distance of 100000 random templates, while visualizing the computation with a progress bar.

### Test targets

There are a couple of tests that can be ran to check that everything works (you can find them in the `./build/test/` directory). You can run them all with the `./build/test/all_tests` target.

The `Matcher` class
-------------------

### GPU memory allocation (`d_bytes` property)

The core of the project lies in the `Matcher` class. Given a FASTA sequence, it stores in GPU memory the whole sequence in compressed byte format (4 nucleotides per byte) four times, one per nucleotide. Specifically, the sequence `GATTACAGATTACA`, will be mapped in memory as
```
 byte   0    1    2    3
shift
    0   GATT ACAG ATTA CA**
    1   ATTA CAGA TTAC A***
    2   TTAC AGAT TACA ****
    3   TACA GATT ACA* ****
```
to allow fast comparison with the given templates. Two warnings:

 1. As you may notice, not all bytes are meaningful: depending on the sequence length and on the shift, some nucleotide "chunk" can be misaligned and should be ignored. The (last) misaligned bytes are indicated in the code by the `excess` variable, which specify the number of (rightmost) nucleotides, which can be a value between 0 (perfectly aligned) and 3 (the byte contains only a signigicant nucleotide). More bytes may become misaligned as we increase the shift. 

 2. The shifted sequence is actually stored "flattened", so in the previous example we have
 ```
  byte 0    1    2    3    4    5    6    7    8    ...
 shift 0    1    2    3    0    1    2    3    0    ...
       GATT ACAG ATTA CA** ATTA CAGA TTAC A*** TTAC ...
 ```
 
Summing up, a byte of index `idx` can be considered useful if it satisfies the inequality `4*idx + excess + shift < 4*num_bytes`.

### Interface

For the moment, the only available function provided by the class is `min_hamming_distance`, which is used by the `hamming` executable. Other function/metrics could be implemented as long as they exploit the aforementioned memory mapping.

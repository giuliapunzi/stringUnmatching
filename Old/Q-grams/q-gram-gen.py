
# global var Q
Q = 30


# print Q-grams that only contains A C G T
def printQgrams( s ):
    s = s.upper()
    key = ""
    for i in range(len(s)):
        if s[i] == 'A' or s[i] == 'C' or s[i] == 'G' or s[i] == 'T':
            key += s[i]
            if len(key) == Q:
                print(key)
                key = key[-(Q-1):]
        else:
            key = ""
            


if __name__ == "__main__":
    with open('all_seqs.fa', 'r') as f:
        seq = ""
        for line in f.readlines():
            # Loop over lines in file
            if line.startswith(">"):
                # if we get '>' it is time for a new sequence
                seq = ""
            else:
                # accumulate current line
                seq += line.strip()
                if len(seq) >= Q:
                    printQgrams(seq)       # output the newly added Q-grams
                    seq = seq[-(Q-1):]     # take the last Q-1 elements for the next iteration


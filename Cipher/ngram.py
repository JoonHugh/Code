DEBUG = 0

import csv

def compute_frequency(path_to_input, n):
    """
    compute the frequency of each n-gram
    and return it as a list of tuples.
    Parameters:
    ----------------
    path_to_input: string containing the path to the input file,
                    i.e., 'ciphertext.txt'
    n: an integer corresponding to the number of characters to consider
    """
    #-------------------------------------------#
    #        Your code goes here.               #
    #-------------------------------------------#
    with open(path_to_input) as file:
        content = file.read()

    count = {}

    for index in range(len(content) - (n - 1)):
            if index != len(content):
                ngram = content[index:index+n]
                if ngram in count:
                    count[ngram] += 1
                else:
                    count[ngram] = 1
                if (DEBUG): print(content[index:index+n])


    if (DEBUG): print("char count:", count)

    freq_list = sorted(count.items(), key=lambda x: (-x[1], x[0])) # sorting in descending order for frequencies of chars

    # writing to a CSV file, had to import csv
    output_filename = f"frequency_{n}.txt"
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for ngram, freq in freq_list:
            writer.writerow([ngram, freq])

    return freq_list

freq_list = compute_frequency("ciphertext.txt", 1)

print("freq_list:", freq_list)
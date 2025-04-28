DEBUG = 0

def count_doubles(path_to_input):
    """
    This function returns the frequencies of double characters as a
    list of tuples.
    Parameters:
    ----------------
    path_to_input: string containing the path to the input file,
                    i.e., 'ciphertext.txt'
    """
    #-------------------------------------------#
    #        Your code goes here.               #
    #-------------------------------------------#
    
    with open(path_to_input) as file:
        content = file.read()

    freq_list = []
    count = {}

    for index in range(len(content) - 1):
        if content[index] == content[index+1]:
            double = content[index:index+2]
            if double in count:
                count[double] += 1
            else:
                count[double] = 1
            if (DEBUG): print(content[index:index+2])


    if (DEBUG): print("char count:", count)

    freq_list = sorted(count.items(), key=lambda x: (-x[1], x[0])) # sorting in descending order for frequencies of chars
    
    return freq_list


doubles_freq = count_doubles("ciphertext.txt")

print("doubles frequency:", doubles_freq)
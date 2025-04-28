DEBUG = 1

def get_distances_between_trigrams(ciphertext):
    # Scan trigrams in the given `ciphertext` and find the distances
    # between repeated trigrams.
    # Returns:
    #  - distances: a dictionary in which keys are trigrams and values
    #               are lists of distances (i.e., number of letters)
    #               between repeated trigrams
    # e.g., distances={ 'sys': [265], 'vwf': [230] ...}
        distances = {}
        trigram_positions = {}
        # -----------------------#
        #  Your code goes here   #
        # -----------------------#
        for index in range(len(ciphertext) - 2):
            trigram = ciphertext[index:index+3]
            if trigram in trigram_positions:
                distance = index - trigram_positions[trigram]
                if trigram in distances:
                    distances[trigram].append(distance)
                else:
                    distances[trigram] = [distance]
                trigram_positions[trigram] = index  # update to new position
            else:
                trigram_positions[trigram] = index

        if (DEBUG):
            print("distances", distances)    
            print("dict", trigram_positions)
            print(len(ciphertext))


        return distances

def get_all_factors(repeated_distances, max_key_len=20):
    """
    Returns all the factors of observed distances.
    For each distance value in `repeated_distances`, find its factors
    and then compute their frequencies.

    Parameters:
    --------------------
    repeated_distance: a dictionary containing repeated trigrams and the
                       distances between the repeats
    max_key_len: an integer, the largest key length to consider (i.e., the
                 largest factor to consider)

    Return:
    --------------------
    factors_histo: a list of tuples consisting of (factor, freqeuncy),
                   sorted in decreasing order of frequency

    Example:
    --------------------
    Assume `repeated_distance` = {'sys': [265], 'vwf': [230]}.
    Then factors are [
        5, 53                       # factors of 265
        2, 10, 23, 46, 115, 230     # factors of 230
    ]
    and your output should be [(5, 1), (2, 1), (10, 1)]
    """

    factor_freq = []

    # --------------------------#
    #   Your code goes here     #
    # --------------------------#

    # you will need internally call `find factors` function to 
    # find the factors of an observed distance.

    list_freq = []

    # calls find_factor function and appends result to a list.
    for distance_list in repeated_distances.values():
         for distance in distance_list:
            list_freq.append(find_factors(distance, 20))
    if (DEBUG): print("list frequency:", list_freq)

    res = [x for sublist in list_freq for x in sublist] # flatten the nested list. 
    # For every sublist within the list, and for every elem in sublist.
    if (DEBUG): print("resulting flattened list:", res)

    count = {} # new dictionary
    # increments the frequencies of factors for each int
    for factor in res:
        if factor in count:
            count[factor] += 1
        else:
            count[factor] = 1
    if (DEBUG): print("counted factors:", count)

    factor_freq = sorted(count.items(), key=lambda x: (-x[1], x[0])) # sorting in descending order for frequencies of factors
    if (DEBUG): print("factor_frequency", factor_freq)

    return factor_freq

def find_factors(num, max_key_len):
    """
    finds the factors of `num`, excluding 1.
    You don't have to consider the factors greater than `max_key_len`

    Return
    ------------------
    factors: a list containing factors of `num` smaller than or
             equal to `max_key_len`
    """
    factors = []

    # -----------------------#
    #  Your code goes here   #
    # -----------------------#
    if (DEBUG):
        print("num in factor function:", num)
        print("max_key_len in factor function", max_key_len)
        print("range:", range(1, max_key_len))

    # simply iterating through all possible factors in integer (excluding 1, and 20+)
    for i in range(2, max_key_len + 1):
         if num % i == 0:
              factors.append(i)

    return factors

def kasiski_test(ciphertext):
    """
    A driver program
    """
    repeated_distances = get_distances_between_trigrams(ciphertext)
    key_len_to_try = get_all_factors(repeated_distances, max_key_len=20)
    for i, candidate in enumerate(key_len_to_try):
       print(f"Candidate [{i}]: {candidate[0]}")


ciphertext = "nenusyegjlegnpwzealpffgzcohojvvsjkwoddirsaoomyaoevzwvoztwjvwfsxldyuselxmngoksvzfyifwcaxouevcnxgqpvrwjtbumuofvdcllusmhzpletrusepwejrtgkshafpovafwmaqocojhbxzpccjhvizlhmpwfqxazhcdnsrgkhrfvmlhvuzngksokrvefoxdgkkinhclxqcietprlaehfqwrzqxtulgfdwjzrsrqqdudkhuuflbspvvyzgrqsdaqzsyeelvalsprlwrusmxzvwfuglzuvsdxkunowzzsorwcblboervqtebuamupvbfusrizzoihgenwspscigkhnwweebjbecjlhtpvvnvyjrfphsejkhrlhtafndphbsskkixhktbusmzhyhdefvosapvifrrwvqdcdhnoenweziv"

distances = get_distances_between_trigrams(ciphertext)
print(distances)

kasiski_test(ciphertext)
#get_all_factors(distances, max_key_len=20)
#factors = find_factors(200, 20)
#print("factors:", factors)
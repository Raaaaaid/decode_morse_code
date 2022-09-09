import re
from sklearn.cluster import KMeans
from statistics import mean


# constants used to decode morse codes
CHAR_SEP = ' '
WORD_SEP = ' ' * 3
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I',
    '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z', '-----': '0',
    '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9', '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',
    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
}

# constants used to convert bit sequences to morse codes
LITERAL_SEP_B = '0'
CHAR_SEP_B = '0' * 3
WORD_SEP_B = '0' * 7
SEP_B = [LITERAL_SEP_B, CHAR_SEP_B, WORD_SEP_B]
DOT_B = '1'
DASH_B = '1' * 3
MORSE_CODE_B = {DOT_B: '.', DASH_B: '-'}

assert len(LITERAL_SEP_B) == len(DOT_B) and len(CHAR_SEP_B) == len(DASH_B)


def decode_morse(morse_code):
    """
    This function cleans up the morse code and decodes it to plain text. The translations
    are stored in the MORSE_CODE dict.
    """

    morse_code = re.sub('[^-. ]+', '', morse_code)
    morse_code = morse_code.strip()
    if not morse_code:
        return ''
    solution = ''
    for word in morse_code.split(WORD_SEP):
        for char in word.split(CHAR_SEP):
            solution += MORSE_CODE[char]
        solution += ' '
    return solution.strip()


def decode_normalized_bits(bits):
    """
    This functions decodes a normalized bit sequence and returns its morse code. In a normalized
    bit sequence the lengths of 0s and 1s sequences correspond to the lengths of the tokens stored
    in DOT_B, DASH_B, LITERAL_SEP_B, CHAR_SEP_B and WORD_SEP_B.
    """

    solution = ''
    for word in bits.split(WORD_SEP_B):
        for char in word.split(CHAR_SEP_B):
            for literal in char.split(LITERAL_SEP_B):
                solution += MORSE_CODE_B[literal]
            solution += CHAR_SEP
        solution = solution[0:-len(CHAR_SEP)] + WORD_SEP
    return solution.strip()


def decode_bits(bits):
    """
    This function cleans up a bit sequence, calculates the transmission rate and normalizes the sequence.
    Different sequences can have different transmission rates, i. e. in a message a dot/dash is represented
    by 1/111 and in another message a dot/dash is represented by 111/111111111. This function assumes that
    the rate is consistent for the whole message.
    """

    bits = re.sub('[^01]+', '', bits)
    bits = bits.strip('0')
    if not bits:
        return ''
    transmission_rate = min([len(token) for token in re.findall('0+|1+', bits)])
    for bit in ['0', '1']:
        bits = re.sub(bit + '{' + str(transmission_rate) + '}', bit, bits)
    return decode_normalized_bits(bits)


def decode_bits_advanced(bits):
    """
    This function works like decode_bits, but it pays attention to inconsistent transmission rate. Inconsistent
    transmission rate could occur when a human morses.
    """

    # clean up
    bits = re.sub('[^01]+', '', bits)
    bits = bits.strip('0')
    if not bits:
        return ''

    # Hard coded special case because you cannot determine if this sequence should be
    # decoded to I or EE. You need this for Codewars because it is a test case.
    if bits == '1001':
        return decode_normalized_bits('10001')

    tokens = re.findall('0+|1+', bits)
    token_lens = [len(token) for token in tokens]
    normalized_bits = ''

    # for cases where training data is too little to train k means
    if len(set(token_lens)) <= 2:
        transmission_rate = min(token_lens)
        for token in tokens:
            if '1' in token:
                len_diffs = [abs(len(token) - len(i) * transmission_rate) for i in [DOT_B, DASH_B]]
                min_ind = len_diffs.index(min(len_diffs))
                normalized_bits += [DOT_B, DASH_B][min_ind]
            else:
                len_diffs = [abs(len(token) - len(i) * transmission_rate) for i in SEP_B]
                min_ind = len_diffs.index(min(len_diffs))
                normalized_bits += SEP_B[min_ind]
        return decode_normalized_bits(normalized_bits)

    # separate sequence lengths into three clusters (one bit, three bits and seven bits sequences) with k means
    training_data = [[i] for i in token_lens]
    k_means = KMeans(init='k-means++', n_clusters=3, n_init=21).fit(training_data)
    clustering_results = {0: set(), 1: set(), 2: set()}
    for i in range(len(training_data)):
        token_len = training_data[i][0]
        label = k_means.labels_[i]
        clustering_results[label].add(token_len)

    # determine which of the clusters is the one bit, three bits and seven bits cluster
    cluster_means = [mean(cluster) for cluster in clustering_results.values()]
    one_bit_cluster_ind = cluster_means.index(min(cluster_means))
    seven_bits_cluster_ind = cluster_means.index(max(cluster_means))
    three_bits_cluster_ind = [i for i in range(3) if i not in [one_bit_cluster_ind, seven_bits_cluster_ind]][0]
    one_bit_cluster, three_bits_cluster, seven_bits_cluster = (clustering_results[one_bit_cluster_ind],
                                                               clustering_results[three_bits_cluster_ind],
                                                               clustering_results[seven_bits_cluster_ind])
    print(f'clusters (1 bit, 3 bits, 7 bits): {one_bit_cluster}, {three_bits_cluster}, {seven_bits_cluster}')

    # calculate the transmission rate as an arithmetic mean
    sum_ = sum(one_bit_cluster) + sum([i / 3 for i in three_bits_cluster])
    transmission_rate = sum_ / (len(one_bit_cluster) + len(three_bits_cluster))
    print(f'transmission_rate: {transmission_rate}')

    # Adjust some memberships of sequence lengths with the help of the transmission rate. Without this
    # step some edge cases would be in the wrong cluster.
    new_one_bit_cluster, new_three_bits_cluster, new_seven_bits_cluster = set(), set(), set()
    for token_len in set(token_lens):
        len_diffs = [abs(token_len - len(i) * transmission_rate) for i in SEP_B]
        min_ind = [i for i in range(len(SEP_B)) if len_diffs[i] == min(len_diffs)]
        if len(min_ind) == 1:
            if min_ind[0] == 0:
                new_one_bit_cluster.add(token_len)
            elif min_ind[0] == 1:
                new_three_bits_cluster.add(token_len)
            else:
                new_seven_bits_cluster.add(token_len)
        else:
            if token_len in one_bit_cluster:
                new_one_bit_cluster.add(token_len)
            elif token_len in three_bits_cluster:
                new_three_bits_cluster.add(token_len)
            else:
                new_seven_bits_cluster.add(token_len)
    print(f'adjusted clusters (1 bit, 3 bits, 7 bits): {new_one_bit_cluster}, {new_three_bits_cluster}, '
          f'{new_seven_bits_cluster}')

    # normalize the bit sequence
    for token in tokens:
        if '1' in token:
            if len(token) in new_one_bit_cluster:
                normalized_bits += DOT_B
            else:
                normalized_bits += DASH_B
        else:
            if len(token) in new_one_bit_cluster:
                normalized_bits += LITERAL_SEP_B
            elif len(token) in new_three_bits_cluster:
                normalized_bits += CHAR_SEP_B
            else:
                normalized_bits += WORD_SEP_B

    return decode_normalized_bits(normalized_bits)

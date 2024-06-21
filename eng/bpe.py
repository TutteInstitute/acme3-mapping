import numba
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm


@numba.njit(nogil=True)
def count_pairs(char_list):
    """Generate pair counts for all pairs of codes in char_list.

    Parameters
    ----------
    char_list: list of arrays of int64
        A list of encoded arrays for which pairs will be counted

    Returns
    -------
    pair_counts: dict of pairs to int
        A dictionary mapping pairs of codes to the count of the total
        number of occurrences of the pair in encoded arrays.
    """
    result = {}
    for array in char_list:
        for i in range(array.shape[0] - 1):
            pair = (array[i], array[i + 1])
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result


@numba.njit(inline="always", nogil=True)
def pair_length(pair, pair_lengths, max_char_code):
    left_length = 1 if pair[0] <= max_char_code else pair_lengths[pair[0]]
    right_length = 1 if pair[1] <= max_char_code else pair_lengths[pair[1]]
    return left_length + right_length


@numba.njit(
    nogil=True,
    locals=dict(
        i=numba.uint32,
        skip_char=numba.boolean,
        len_char_list=numba.uint32,
        last_char_added=numba.int64,
    )
)
def contract_and_count_pairs(char_list, pair_to_contract, pair_counts, new_code=-1):
    """Generate a new encoding by replacing ``pair_to_contract`` with ``new_code``
    and simultaneously updating the ``pair_counts`` dictionary to reflect the new
    pair counts resulting from the merges. This allows counting and merging to be done
    in a single pass.

    Parameters
    ----------
    char_list: array of int64
        The current encoding to be improved by contracting ``pair_to_contract`` to ``new_code``

    pair_to_contract: pair of int64
        The pair of codes to be contracted to a single new code

    pair_counts: dict of pairs to int
        The current counts of pairs that are being kept track of. This dict will
        be updated to reflect the new counts resulting from the contractions.

    new_code: int
        The new code value to replace ``pair_to_contract`` with.

    Returns
    -------
    new_char_list: array of int64
        The new array of codes with ``pair_to_contract`` merged to ``new_code``
        wherever it occurs.

    correction: updates to apply to pair_counts
    """
    skip_char = False
    len_char_list = len(char_list)
    correction = [((-1, -1), 0)]
    correction.pop(0)
    if len_char_list == 1:
        return char_list, correction
    last_char_added = -1
    new_char_list = np.zeros(len_char_list, dtype=np.int64)
    new_char_index = 0

    for i in range(len_char_list - 1):
        if skip_char:
            skip_char = False
            continue

        if char_list[i] == pair_to_contract[0] and char_list[i + 1] == pair_to_contract[1]:
            if i > 0:
                prior_pair = (last_char_added, char_list[i])
                # if prior_pair in pair_counts:
                #     pair_counts[prior_pair] -= 1
                correction.append((prior_pair, -1))
                new_prior_pair = (last_char_added, new_code)
                # if new_prior_pair in pair_counts:
                #     pair_counts[new_prior_pair] += 1
                # else:
                #     pair_counts[new_prior_pair] = 1
                correction.append((new_prior_pair, 1))

            if i < len_char_list - 2:
                next_pair = (char_list[i + 1], char_list[i + 2])
                # if next_pair in pair_counts:
                #     pair_counts[next_pair] -= 1
                correction.append((next_pair, -1))
                new_next_pair = (new_code, char_list[i + 2])
                # if new_next_pair in pair_counts:
                #     pair_counts[new_next_pair] += 1
                # else:
                #     pair_counts[new_next_pair] = 1
                correction.append((new_next_pair, 1))

            new_char_list[new_char_index] = new_code
            last_char_added = new_code
            skip_char = True
        else:
            new_char_list[new_char_index] = char_list[i]
            last_char_added = char_list[i]

        new_char_index += 1

    if not skip_char:
        new_char_list[new_char_index] = char_list[i + 1]
        new_char_index += 1

    # return new_char_list[:new_char_index], pair_counts
    return new_char_list[:new_char_index], correction


@numba.njit(nogil=True)
def pruning_max_freq_pair(count_dict, code_lengths, max_char_code, min_count=1):
    """Find the maximum frequency pair given a dictionary of counts of pair occurrences.
    Ties are broken first on the lengths (as string token representation) of the pairs,
    and then lexicographically on the pair's code values.

    During the search for the max, we will also find pairs to be removed from the
    dictionary based on the ``min_count`` value. This allows the dictionary to remain
    smaller than it otherwise would.

    Parameters
    ----------
    count_dict: dict of pairs to ints
        The counts of the number of occurrences of pairs

    code_lengths: dict of codes to ints
        The lengths of different codes as string token representations

    max_char_code: int
        The maximum code value of any single character in the learned code

    min_count: int
        The minimum number of occurrences of a pair for it to remain
        in the dictionary. Pairs with fewer than this number of occurences
        will be pruned out.

    Returns
    -------
    best_pair: pair of int64s
        The pair of codes that are the most frequent

    max_count: int
        the number of occurrences of the best pair.
    """
    result = (-1, -1)
    max_count = 0
    best_length = 0
    keys_to_kill = set([(-1, -1)])
    for pair, count in count_dict.items():
        if count > max_count:
            result = pair
            max_count = count
            best_length = pair_length(pair, code_lengths, max_char_code)
        elif count == max_count:
            length = pair_length(pair, code_lengths, max_char_code)
            if length > best_length or (length == best_length and pair > result):
                result = pair
                max_count = count
                best_length = length
        elif count <= min_count:
            keys_to_kill.add(pair)

    if len(keys_to_kill) > 0:
        for key in keys_to_kill:
            if key[0] >= 0:
                count_dict.pop(key)

    if max_count == 1:
        return (-1, -1), 0

    return result, max_count


@numba.njit(inline="always", nogil=True)
def to_unicode(code, tokens, max_char_code):
    if code <= max_char_code:
        return chr(code)
    else:
        return tokens[code - max_char_code - 1]


@numba.njit(inline="always", nogil=True)
def to_code_num(code, code_list, max_char_code):
    if code <= max_char_code:
        return [code]
    else:
        return code_list[code - max_char_code - 1]


@numba.njit(inline="always", nogil=True)
def pair_to_string(pair, tokens, max_char_code):
    return to_unicode(pair[0], tokens, max_char_code) + to_unicode(pair[1], tokens, max_char_code)


@numba.njit(inline="always", nogil=True)
def pair_to_list(pair, code_list, max_char_code):
    return to_code_num(pair[0], code_list, max_char_code) + to_code_num(pair[1], code_list, max_char_code)


@numba.njit(nogil=True, parallel=True)
def train(char_list, vocab_size=10000, min_count=1, max_char_code=255):
    """Train a byte pair encoding on a given list of strings.

    Parameters
    ----------
    char_list: list of strings
        The strings to learn a byte pair encoding scheme from

    vocab_size: int
        The maximum number of new codes representing byte sequences to learn.

    min_count: int
        The minimum number of occurrences a pair must have to be considered for merging.

    max_char_code: int64
        The maximum value of character codes in data handled. For ascii strings
        this is simply 127, but for unicode strings it may be significantly larger. Code
        values associated with new learned tokens begin at ``max_char_code + 1``.

    Returns
    -------
    tokens: list of strings
        The string representations of the new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code + 1``.

    code_list: list of pairs of int64s
        The pairs merged to create new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code + 1``.

    compressed_chars: list of arrays of int64s
        The encoded versions of the input strings
    """
    # Initialize compressed chars
    compressed_chars = [np.empty(len(chars), dtype=np.int64) for chars in char_list]
    # for i, chars in enumerate(char_list):
    for i in numba.prange(len(char_list)):
        for j, c in enumerate(char_list[i]):
            c_val = ord(c)
            if c_val > max_char_code:
                c_val = 0
            compressed_chars[i][j] = c_val

    # Initialize coding, counts, and lengths
    new_code = max_char_code + 1
    pair_counts = count_pairs(compressed_chars)
    current_min_count = np.max(np.array(list(pair_counts.values()))) // 2
    code_lengths = {-1: 1}

    # Initialize code words and lengths so numba gets the types right
    pair_to_replace, count = pruning_max_freq_pair(
        pair_counts, code_lengths, max_char_code, min_count=current_min_count
    )
    tokens = [chr(pair_to_replace[0]) + chr(pair_to_replace[1])]
    code_list = [pair_to_replace]
    code_lengths[new_code] = pair_length(pair_to_replace, code_lengths, max_char_code)

    corrections = [[((-1, -1), 0)]] * len(compressed_chars)
    while len(tokens) < vocab_size:
        for i in numba.prange(len(compressed_chars)):
            compressed_chars[i], corrections[i] = contract_and_count_pairs(
                compressed_chars[i], pair_to_replace, pair_counts, new_code
            )
        for correction in corrections:
            for pair, n in correction:
                if n < 0:
                    if pair in pair_counts:
                        pair_counts[pair] += n
                elif n > 0:
                    pair_counts.setdefault(pair, 0)
                    pair_counts[pair] += n

        pair_counts.pop(pair_to_replace)
        new_code += 1
        pair_to_replace, count = pruning_max_freq_pair(
            pair_counts, code_lengths, max_char_code, min_count=current_min_count
        )

        if current_min_count > min_count and count <= current_min_count:
            current_min_count = max(current_min_count // 2, min_count)
            pair_counts = count_pairs(compressed_chars)
            pair_to_replace, count = pruning_max_freq_pair(
                pair_counts, code_lengths, max_char_code,
                min_count=current_min_count
            )

        if pair_to_replace[0] >= 0 and pair_counts[pair_to_replace] > 1:
            tokens.append(pair_to_string(pair_to_replace, tokens, max_char_code))
            code_list.append(pair_to_replace)
            code_lengths[new_code] = pair_length(pair_to_replace, code_lengths, max_char_code)
        else:
            break

    return tokens, code_list, compressed_chars


@numba.njit()
def contract_pair(char_list, pair_to_contract, new_code=-1):
    """Generate a new array on codes by contracting ``pair_to_contract`` to
    the code ``new_code``.

    Parameters
    ----------
    char_list: array of int64
        The array to apply pair contraction to

    pair_to_contract: pair of int64
        The code pair to be contracted to a new code value

    new_code: int64
        The new code value to use in place of ``pair_to_contract``

    Returns
    -------
    new_char_list: array of int64
        The new array of codes with ``pair_to_contract`` merged to ``new_code``
        wherever it occurs.
    """
    skip_char = False
    len_char_list = len(char_list)
    if len_char_list == 1:
        return char_list
    new_char_list = np.zeros(len_char_list, dtype=np.int64)
    new_char_index = 0

    for i in range(len_char_list - 1):
        if skip_char:
            skip_char = False
            continue

        if char_list[i] == pair_to_contract[0] and char_list[i + 1] == pair_to_contract[1]:
            new_char_list[new_char_index] = new_code
            skip_char = True
        else:
            new_char_list[new_char_index] = char_list[i]

        new_char_index += 1

    if not skip_char:
        new_char_list[new_char_index] = char_list[i + 1]
        new_char_index += 1

    return new_char_list[:new_char_index]


@numba.njit(nogil=True)
def bpe_encode(chars, code_list, max_char_code):
    """Encode a string given a BPE code_list

    Parameters
    ----------
    chars: unicode_type
        THe string to be encoded

    code_list: list of code pairs (int64, int64)
        The learned encoding dictionary

    max_char_code: int64
        The maximum allowed code char for the given learned encoding

    Returns:
    --------
    compressed_array: ndarray of int64
        The array of the encoded representation
    """
    compressed_chars = np.empty(len(chars), dtype=np.int64)
    for i, c in enumerate(chars):
        code = ord(c)
        if code > max_char_code:
            raise ValueError(f"Character at index {i}, code {code}, is out of bounds)")
        compressed_chars[i] = code if code <= max_char_code else 0

    new_code = max_char_code + 1
    for code_pair in code_list:
        compressed_chars = contract_pair(compressed_chars, code_pair, new_code=new_code)
        new_code += 1

    assert np.all(compressed_chars <= max_char_code + len(code_list))
    return compressed_chars


@numba.njit(nogil=True)
def bpe_decode(code_array, tokens, max_char_code):
    """Decode a BPE code array into a string

    Parameters
    ----------
    code_array: array of int64
        The code array to decode

    tokens: list of unicode_type
        The string representations of learned codes

    max_char_code: int64
        The maximum allowed code char for the given learned encoding

    Returns
    -------

    """
    result = [
        chr(c) if c <= max_char_code else tokens[c - max_char_code - 1]
        for c in code_array
    ]
    return "".join(result)


@numba.njit(nogil=True)
def _coo_block(row_index, encoding):
    tokens = np.unique(encoding)
    block = np.zeros((tokens.shape[0], 3), dtype=np.int64)
    block[:, 0] = int(row_index)
    block[:, 1] = tokens
    token2brow = dict(zip(tokens, np.arange(tokens.shape[0])))
    for token in encoding:
        block[token2brow[token], 2] += 1
    return block


@numba.njit(
    nogil=True,
    parallel=True,
)
def _gather_coo_data(strings, code_list, max_char_code):
    blocks = [np.empty((1, 3), dtype=np.int64)] * len(strings)
    for i in numba.prange(len(strings)):
        blocks[i] = _coo_block(i, bpe_encode(strings[i], code_list, max_char_code))
    return blocks


def vectorize(strings, code_list, max_char_code=255):
    coo_data = np.vstack(_gather_coo_data(strings, code_list, max_char_code))
    return sp.coo_matrix((coo_data[:, 2], (coo_data[:, 0], coo_data[:, 1])))

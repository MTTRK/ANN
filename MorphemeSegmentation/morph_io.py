import sys
import numpy as np
import string


SEPARATOR = '\t'
START_SIGN = '<s>'
STOP_SIGN = '</s>'
START = 'START'
STOP = 'STOP'
BEGIN = 'B'
MIDDLE = 'M'
END = 'E'
SINGLE = 'S'
ALPHABET = [START_SIGN, STOP_SIGN, "'", '-'] + list(string.ascii_lowercase)
OUTPUT_ALPHABET = list({BEGIN, MIDDLE, SINGLE, END})


def read_file(file_path: str):
    """
    :param file_path: input file path
    """

    with open(file_path, 'r') as file:
        for line in file:
            yield line


def read_stdin():
    for line in sys.stdin:
        yield line


def get_input_map():
    input_identity = np.identity(len(ALPHABET))
    return {key: input_identity[ALPHABET.index(key)] for key in ALPHABET}


def get_output_map():
    output_identity = np.identity(len(OUTPUT_ALPHABET))
    return {key: output_identity[OUTPUT_ALPHABET.index(key)] for key in OUTPUT_ALPHABET}


def print_mapping(title, mapping):
    """
    :param title: str
    :param mapping: dictionary
    """

    print('\n' + title)
    l = list(mapping)
    l.sort()
    for key in l:
        print(str(key) + (8 - len(key)) * ' ' + '--> ' + str(mapping[key]))


def prepare_symbols_for_windowing(symbols, n, pad, drop):
    """
    :param symbols: ['<s>', 'a', ...]
    :param n: int
    :param pad: '</s>'
    :param drop: '<s>'
    """

    w_input_symbols = []

    for s in symbols:
        # we will need n-1 'pad' symbols (for the window)
        if s == pad:
            for i in range(0, n - 1):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)
    # we can remove the 'drop' symbols (won't be using them)
    return [x for x in w_input_symbols if x != drop]


def use_left_window(n, input_symbols):
    """
    :param n: int
    :param input_symbols: ['<s>', 'a', 'b', ...]
    """

    w_input_symbols = prepare_symbols_for_windowing(input_symbols, n, START_SIGN, STOP_SIGN)
    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i + n - 1] != START_SIGN:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


def use_right_window(n, input_symbols):
    """
    :param n: int
    :param input_symbols: ['<s>', 'a', 'b', ...]
    """

    w_input_symbols = prepare_symbols_for_windowing(input_symbols, n, STOP_SIGN, START_SIGN)
    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i] != STOP_SIGN:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


def use_center_window(n, input_symbols):
    """
    :param n: int
    :param input_symbols: ['<s>', 'a', 'b', ...]
    """

    hw_size = int(n / 2)
    w_size = hw_size * 2 + 1
    w_input_symbols = []

    for s in input_symbols:
        if s in [STOP_SIGN, START_SIGN]:
            for i in range(0, hw_size):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)

    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - hw_size):
        if w_input_symbols[i + hw_size] not in [STOP_SIGN, START_SIGN]:
            block = []
            for j in range(0, w_size):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


def map_to_list(block, input_map):
    """
    :param block: ['a', 'b', 'c'] (window size=3)
    :param input_map: {'a':np.array([0 0 0 1 ...]), ...}
    """

    onehot_list = [input_map[x] for x in block]
    return [x for sublist in onehot_list for x in sublist]


def predictions_to_symbols(prediction_matrix, output_map):
    """
    :param prediction_matrix: [[1 0 ...], [0 0 1 ...], ...]
    :param output_map: {'B': [1 0 ...], ...}
    """

    symbols = []
    for block in prediction_matrix:
        max_index = block.tolist().index(max(block))
        filtered_symbol = [key for (key, value) in output_map.items() if value.tolist().index(1.0) == max_index]
        symbols.append(filtered_symbol[0])

    return symbols


def transform_input(input_symbols, window_type, window_size):
    """
    :param input_symbols: ['<s>', 'a', ...]
    :param window_type: windowing function
    :param window_size: int
    """

    # applying the windowing
    # the function returns a list of lists (the nested lists are window-sized)
    w_input_symbols = window_type(window_size, input_symbols)

    # one-hot vectors for the input
    input_map = get_input_map()
    inputs = [map_to_list(block, input_map) for block in w_input_symbols]
    return inputs


def process_training_input(input, window_type, window_size):
    """
    :param input: [['<s>', 'START'], ['a', 'B'], ...]
    :param window_type: windowing function
    :param window_size: int
    """

    # entire list of input symbols
    input_symbols = [x[0] for x in input]
    inputs = transform_input(input_symbols, window_type, window_size)

    # we don't need the START/STOP signs (they are not going to be classified anyway)
    output_symbols = list(x[1] for x in input if x[1] not in [START, STOP])

    # one-hot vectors for the output
    output_map = get_output_map()
    outputs = [output_map[x] for x in output_symbols]

    return np.asarray(inputs), get_input_map(), np.asarray(outputs), output_map

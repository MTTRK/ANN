'''
Expected format:
...
<s>     START
a       B
n       E
n       B
o       M
t       E
a       B
t       E
i       B
o       M
n       E
s       S
</s>    STOP
...
'''
import sys
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np


# filepath: str
def read_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            yield line


def read_stdin():
    for line in sys.stdin:
        yield line


# title: str
# mapping: dictionary
def print_mapping(title, mapping):
    print('\n' + title)
    l = list(mapping)
    l.sort()
    for key in l:
        print(str(key) + (8 - len(key)) * ' ' + '--> ' + str(mapping[key]))


# n: int
# input_symbols: ['<s>', 'a', 'b', ...]
def use_left_window(n, input_symbols):
    print('\nUsing left window')
    w_input_symbols = []

    for s in input_symbols:
        # we will need n-1 INPUT_START every time for the window
        if s == INPUT_START:
            for i in range(0, n - 1):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)
    # we can drop the INPUT_STOP signs (won't be using them)
    w_input_symbols = [x for x in w_input_symbols if x != INPUT_STOP]

    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i + n - 1] != INPUT_START:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


# n: int
# input_symbols: ['<s>', 'a', 'b', ...]
def use_right_window(n, input_symbols):
    print('\nUsing right window')
    w_input_symbols = []

    for s in input_symbols:
        # we will need n-1 INPUT_STOP every time for the window
        if s == INPUT_STOP:
            for i in range(0, n - 1):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)

    # we can drop the INPUT_START signs (won't be using them)
    w_input_symbols = [x for x in w_input_symbols if x != INPUT_START]

    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i] != INPUT_STOP:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


# n: int
# input_symbols: ['<s>', 'a', 'b', ...]
def use_center_window(n, input_symbols):
    print('\nUsing center window')
    hw_size = int(n / 2)
    w_size = hw_size * 2 + 1
    w_input_symbols = []

    for s in input_symbols:
        if s in [INPUT_STOP, INPUT_START]:
            for i in range(0, hw_size):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)

    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - hw_size):
        if w_input_symbols[i + hw_size] not in [INPUT_STOP, INPUT_START]:
            block = []
            for j in range(0, w_size):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)

    return w_sized_blocks


# block: ['a', 'b', 'c'] (window size=3)
# input_map: {'a':np.array([0 0 0 1 ...]), ...}
def map_to_nparray(block, input_map):
    onehot_list = [input_map[x] for x in block]
    return np.array([x for sublist in onehot_list for x in sublist])


# input: [['<s>', 'START'], ['a', 'B'], ...]
def process_input(input):
    # entire list of input symbols
    input_symbols = [x[0] for x in input]
    # we don't need the  START/STOP signs (those are not going to be classified)
    output_symbols = list({x[1] for x in input if x[1] not in OUTPUT_EXFILTER})

    # applying the windowing
    # the function returns a list of lists (the nested lists are window-sized)
    w_input_symbols = WINDOW_TYPE(WINDOW_SIZE, input_symbols)
    print('\nAfter windowing: ' + str(w_input_symbols[:5]) + '...')

    # input symbols (set)
    alphabet = list({x for sublist in w_input_symbols for x in sublist})
    alphabet.sort()
    alphabet_size = len(alphabet)

    # one-hot vectors for the output
    output_symbols.sort()
    output_identity = np.identity(len(output_symbols))
    output_map = {key: output_identity[output_symbols.index(key)] for key in output_symbols}

    # one-hot vectors for the input
    input_identity = np.identity(alphabet_size)
    input_map = {key: input_identity[alphabet.index(key)] for key in alphabet}

    inputs = [map_to_nparray(block, input_map) for block in w_input_symbols]
    outputs = [output_map[x] for x in output_symbols]

    # print extra information
    print_mapping('INPUT mapping', input_map)
    print_mapping('OUTPUT mapping', output_map)

    return inputs, input_map, outputs, output_map


def main():
    input = []
    if len(sys.argv[1:]) == 1:
        for line in read_file(sys.argv[1:][0]):
            input.append(line.strip('\n').split('\t'))
    else:
        for line in read_stdin():
            input.append(line)

    # process_input will apply a windowing function to the input just before
    # transforming it (list of symbols) into appropriately sized
    # vectors (the outputs will just simply be translated into one-hot vectors)
    input_matrix, input_mapping, output_matrix, output_mapping = process_input(input)

    # create and train NN
    print('\nConstructing NN')


## HYPER PARAMTERS
WINDOW_SIZE = 3
WINDOW_TYPE = use_center_window
OUTPUT_EXFILTER = ['START', 'STOP']
INPUT_START = '<s>'
INPUT_STOP = '</s>'


if __name__ == "__main__":
    main()

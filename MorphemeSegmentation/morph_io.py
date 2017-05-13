import numpy as np
from context import Context
from context import SegmentationContext
from context import GoldStdWord


def read_goldstd(path: str, ctx: Context):
    """
    Reads the file (in GoldStd format) and returns each word in
    an appropriate context (GoldStdWord instance)
    :param path: path to file
    :param ctx: contains runtime parameters
    :return: [GoldStdWord(...), ...]
    """
    goldstd_words = []
    for line in read_file(path):
        pieces = line.strip('\n').split(ctx.SEPARATOR)
        word = pieces[0]
        if len(pieces) == 1:
            segmentation = word
            mapping = map_segmentation(segmentation, ctx)
            goldstd_words.append(GoldStdWord(word, segmentation, mapping))
        else:
            for segm in pieces[1].split(','):
                segmentation = segm.strip(' ')
                mapping = map_segmentation(segmentation, ctx)
                goldstd_words.append(GoldStdWord(word, segmentation, mapping))

    return goldstd_words


def read_file(path: str):
    """
    :param path: input file path
    :return: generator
    """
    with open(path, 'r') as file:
        for line in file:
            yield line.strip(' \n')


def output_predictions(predictions: list, ctx: Context):
    """
    Outputs predictions in goldstd format to the stdout
    :param predictions: list of Prediction-s
    :param ctx: Context containing runtime parameters
    """
    for pred in predictions:
        segments = ''
        for i in range(0, len(pred.prediction)):
            if pred.prediction[i] in [ctx.BEGIN, ctx.SINGLE]:
                segments += ' '
            segments += pred.word[i]
        print(pred.word + ctx.SEPARATOR + segments.strip(' '))


def transform_output(symbols: list, ctx: Context):
    """
    Transforms the list of symbols into a one-hot matrix
    :param symbols: ['B', 'M', ...]
    :param ctx: Context containing runtime parameters
    :return: matrix
    """
    output_map = ctx.get_output_map()
    onehots = [output_map[symbol] for symbol in symbols]
    return np.asarray(onehots)


def transform_input(symbols: list, ctx: SegmentationContext):
    """
    Transforms list of symbols into a one-hot matrix
    after applying the appropriate windowing function
    :param symbols: ['<s>', 'a', ...]
    :param ctx: SegmentationContext containing the runtime parameters
    :return: matrix
    """
    windowing_func = get_window_function(ctx.windowtype)
    w_symbols = windowing_func(ctx.windowsize, symbols, ctx)

    input_map = ctx.get_input_map()
    onehots = [map_to_elements(block, input_map) for block in w_symbols]
    return np.asarray(onehots)


def predictions_to_symbols(prediction_matrix, ctx: Context):
    """
    Using the one-hot vectors (matrix) this method
    returns the symbols those vectors represent
    :param prediction_matrix: one-hot matrix
    :param ctx: Context containing runtime parameters
    :return: list of symbols, ex.: [B, M, M, B, M]
    """
    symbols = []
    output_map = ctx.get_output_map()

    for block in prediction_matrix:
        max_index = block.tolist().index(max(block))
        filtered_symbol = [key for (key, value) in output_map.items() if value.tolist().index(1.0) == max_index]
        symbols.append(filtered_symbol[0])

    return symbols


def map_to_elements(block: list, e_map: dict):
    """
    Maps the elements in block to a one-hot vector and
    returns the merged list of these
    :param block: ex.: ['a', 'b', 'c']
    :param e_map: element map, ex.: {'a':np.array([0 0 0 1 ...]), ...}
    :return: merged list of one-hot vectors
    """
    onehot_list = [e_map[e] for e in block]
    return [x for onehot in onehot_list for x in onehot]


def map_segmentation(segmentation: str, ctx: Context):
    """
    Maps the segmentation of a word into an internal
    representation, ex.: 'abound ed' --> ['B', 'M', ...]
    :param segmentation: ex.: 'abound ed'
    :param ctx: contains runtime parameters
    :return: mapped segmentation (list)
    """
    mapping = []
    for segment in segmentation.split(' '):
        if len(segment) == 1:
            mapping.append(ctx.SINGLE)
        else:
            mapping.append(ctx.BEGIN)
            for index in range(1, len(segment) - 1):
                mapping.append(ctx.MIDDLE)
            mapping.append(ctx.END)
    return mapping


def get_window_function(_id: int = 0):
    """
    :param _id: (0: left, 1: center, 2: right)
    :return: the windowing function selected
    """
    if _id == 0:
        return use_left_window
    elif _id == 1:
        return use_center_window
    elif _id == 2:
        return use_right_window
    else:
        raise Exception('No windowing function can be selected: ', str(_id))


def prepare_symbols_for_windowing(symbols: list, n: int, pad: str, drop: str):
    """
    Prepares the list of symbols for windowing,
    i.e. replicates the padding symbol and removes the unused one
    :param symbols: input symbol list, ex.: ['<s>', 'a', ...]
    :param n: size of window
    :param pad: symbol to be replicated, ex.: '</s>'
    :param drop: unused symbol, ex.: '<s>'
    :return:
    """
    w_input_symbols = []
    for s in symbols:
        # we will need n-1 'pad' symbols (for the window)
        if s == pad:
            for i in range(0, n - 1):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)

    return [x for x in w_input_symbols if x != drop]


def use_left_window(n: int, input_symbols: list, ctx: Context):
    """
    Applies a left window
    :param n: size of window
    :param input_symbols: ['<s>', 'a', 'b', ...]
    :param ctx: Context containing runtime parameters
    :return: modified input-symbol list
    """
    w_input_symbols = prepare_symbols_for_windowing(input_symbols, n, ctx.START, ctx.STOP)
    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i + n - 1] != ctx.START:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)
    return w_sized_blocks


def use_right_window(n: int, input_symbols: list, ctx: Context):
    """
    Applies a right window
    :param n: size of window
    :param input_symbols: ['<s>', 'a', 'b', ...]
    :param ctx: Context containing runtime parameters
    :return: modified input-symbol list
    """
    w_input_symbols = prepare_symbols_for_windowing(input_symbols, n, ctx.STOP, ctx.START)
    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - n + 1):
        if w_input_symbols[i] != ctx.STOP:
            block = []
            for j in range(0, n):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)
    return w_sized_blocks


def use_center_window(n: int, input_symbols: list, ctx: Context):
    """
    Applies a center window
    :param n: size of window
    :param input_symbols: ['<s>', 'a', 'b', ...]
    :param ctx: Context containing runtime parameters
    :return: modified input-symbol list
    """
    hw_size = int(n / 2)
    w_size = hw_size * 2 + 1
    w_input_symbols = []

    for s in input_symbols:
        if s in [ctx.STOP, ctx.START]:
            for i in range(0, hw_size):
                w_input_symbols.append(s)
        else:
            w_input_symbols.append(s)

    w_sized_blocks = []
    # creating the window sized 'blocks'
    for i in range(0, len(w_input_symbols) - hw_size):
        if w_input_symbols[i + hw_size] not in [ctx.STOP, ctx.START]:
            block = []
            for j in range(0, w_size):
                block.append(w_input_symbols[i + j])
            w_sized_blocks.append(block)
    return w_sized_blocks

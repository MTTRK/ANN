import string
import numpy as np


class GoldStdWord:
    """
    Wrapper class for a goldstd context
    i.e. word - segmentation - segment_mapping
    """
    def __init__(self, _word: str = '', _segmentation: str = '', _seg_mapping: list = []):
        """
        :param _word: 'abounded'
        :param _segmentation: 'abound ed'
        :param _seg_mapping: ['B', 'M', 'M', 'M', 'M', 'M', 'B', 'M']
        """
        self.word = _word
        self.segmentation = _segmentation
        self.seg_mapping = _seg_mapping


class Prediction:
    """
    Wrapper class for the prediction context
    """
    def __init__(self, word, pred):
        """
        :param word: 'abounded'
        :param pred: ['B', 'M', ...]
        """
        self.word = word
        self.prediction = pred


class Context:
    """
    Runtime context
    """
    def __init__(self):
        self.SEPARATOR = '\t'
        self.BEGIN = 'B'
        self.MIDDLE = 'M'
        self.END = self.MIDDLE
        self.SINGLE = self.BEGIN
        self.START = '<s>'
        self.STOP = '</s>'
        self.ALPHABET = \
            [self.START, self.STOP] + \
            ["'", '-'] + \
            list(string.ascii_lowercase) + \
            ['å', 'ä', 'ö'] + \
            ['í', 'ö', 'ü', 'ó', 'ő', 'ú', 'ű', 'é', 'á']
        self.input_map = None
        self.output_map = None

    def set_bmes_context(self):
        """
        Sets the context for BMES classification
        """
        self.END = 'E'
        self.SINGLE = 'S'

    def get_input_map(self):
        """
        Returns a numpy matrix representing the one-hot vectors
        for the input alphabet
        :return: matrix
        """
        if not self.input_map:
            identity = np.identity(len(self.ALPHABET))
            self.input_map = {key: identity[self.ALPHABET.index(key)] for key in self.ALPHABET}
        return self.input_map

    def get_output_map(self):
        """
        Returns a numpy matrix representing the one-hot vectors
        for the output alphabet
        :return: matrix
        """
        if not self.output_map:
            alphabet = list({self.BEGIN, self.SINGLE, self.END, self.MIDDLE})
            identity = np.identity(len(alphabet))
            self.output_map = {key: identity[alphabet.index(key)] for key in alphabet}
        return self.output_map


class EvaluationContext(Context):
    """
    Evaluation specific Context
    """
    def __init__(self):
        super().__init__()
        self.expected = []
        self.actual = []
        self.POS = [self.BEGIN, self.SINGLE]
        self.NEG = [self.MIDDLE, self.END]


class SegmentationContext(Context):
    """
    Segmentation specific Context
    """
    def __init__(self):
        super().__init__()
        self.windowtype = 0
        self.windowsize = 1


class MainContext(SegmentationContext):
    """
    Train & Build specific Context
    """
    def __init__(self, _windowsize: int, _windowtype: int, _hiddenlayer: int, _epochs: int,
                  _activate: str, _optimize: str, _loss: str, _init: str, _earlystop: int,
                  _training: str, _test: str, _devel: str = None,
                 ):
        """
        :param _windowsize: size of window
        :param _windowtype: type of window (0,1,2 --> left,center,right)
        :param _hiddenlayer: number of hidden layers in the network
        :param _epochs: number of epochs to run
        :param _activate: name of the activation function
        :param _optimize: name of the optimization method
        :param _loss: name of the loss function
        :param _init: name of the initialization method
        :param _earlystop: value for early-stopping
        :param _training: training data for segmentation learning
        :param _test: file containing the words to be segmented
        :param _devel: file to be used for benchmark evaluation
        """
        super().__init__()

        self.devel = _devel
        self.training = _training
        self.test = _test
        self.windowsize = _windowsize
        self.windowtype = _windowtype
        self.hiddenlayer = _hiddenlayer
        self.epochs = _epochs
        self.activate = _activate
        self.optimize = _optimize
        self.loss = _loss
        self.init = _init
        self.earlystop = _earlystop

    def __str__(self):
        return '[Window size: ' + str(self.windowsize) + '; ' + \
               'Window type: ' + str(self.windowtype) + '; ' + \
               'Hidden layers: ' + str(self.hiddenlayer) + '; ' + \
               'Epoch size: ' + str(self.epochs) + '; ' + \
               'Activation: ' + self.activate + '; ' + \
               'Optimizer: ' + self.optimize + '; ' + \
               'Loss: ' + self.loss + '; ' + \
               'Initialization: ' + self.init + '; ' + \
               'Early stopping patience: ' + str(self.earlystop) + ']'

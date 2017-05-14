import argparse


class MainOption:
    """
    Options for the morpheme segmentation module
    """
    def __init__(self):
        """
        :param arguments: to be parsed by 'argparse'
        """
        parser = argparse.ArgumentParser(description='== Morpheme Segmentation ==')

        # Required
        parser.add_argument('-t', '--training',
                            type=str,
                            help='Path to the training input',
                            required=True)

        parser.add_argument('-w', '--words',
                            type=str,
                            help='Path to the word-list file (to be segmented)',
                            required=True)

        # Optional
        parser.add_argument('-b', '--benchmark',
                            action='store_true',
                            help='Flag to mark this a benchmark run')

        parser.add_argument('-bmes',
                            action='store_true',
                            help='Flag for the type of classification to use (by default: BM)')

        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Verbose logging')

        parser.add_argument('-d', '--devel',
                            type=str,
                            help='Path to development file (contains segmentation to be used for evaluation)')

        parser.add_argument('-ws', '--windowsize',
                            type=int,
                            default=4,
                            help='Size of the window')

        parser.add_argument('-wt', '--windowtype',
                            type=int,
                            default=1,
                            help='Type of the window [0: left, 1: center, 2: right]')

        parser.add_argument('-hl', '--hiddenlayer',
                            type=int,
                            default=1,
                            help='Number of hidden layers')

        parser.add_argument('-ep', '--epochs',
                            type=int,
                            default=400,
                            help='Number of epochs')

        parser.add_argument('-ac', '--activate',
                            type=str,
                            default='relu',
                            help='Activation method')

        parser.add_argument('-op', '--optimize',
                            type=str,
                            default='adam',
                            help='Optimization method')

        parser.add_argument('-lo', '--loss',
                            type=str,
                            default='mean_squared_error',
                            help='Loss function')

        parser.add_argument('-it', '--init',
                            type=str,
                            default='uniform',
                            help='Initialization method')

        parser.add_argument('-es', '--earlystop',
                            type=int,
                            default=50,
                            help='Early-stop patience')

        args = parser.parse_args()

        self.training = args.training
        self.words = args.words
        self.benchmark = args.benchmark
        self.bmes = args.bmes
        self.devel = args.devel
        self.windowsize = args.windowsize
        self.windowtype = args.windowtype
        self.hiddenlayer = args.hiddenlayer
        self.epochs = args.epochs
        self.activate = args.activate
        self.optimize = args.optimize
        self.loss = args.loss
        self.init = args.init
        self.earlystop = args.earlystop

        if args.verbose:
            self.verbose = 1
        else:
            self.verbose = 0


class EvaluateOption:
    """
    Options for the evaluation module
    """
    def __init__(self):
        """
        :param arguments: to be parsed by 'argparse'
        """
        parser = argparse.ArgumentParser(description='== Evaluation ==')

        # Required
        parser.add_argument('-e', '--expected',
                            type=str,
                            help='Path to the file holding the expected segmentations (in goldstd format)',
                            required=True)

        parser.add_argument('-a', '--actual',
                            type=str,
                            help='Path to the file containing the actual segmentations (in goldstd format)',
                            required=True)

        args = parser.parse_args()

        self.expected_file = args.expected
        self.actual_file = args.actual

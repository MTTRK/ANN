"""
Expected format for training:
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

Expected format for predictions:
...
ablatives
abounded
abrogate
...
"""
import sys
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense
import numpy as np
import morph_io as mio
import evaluate as eval


class Prediction:
    """
    Wrapper class for the prediction context
    """

    def __init__(self, word, pred):
        self.word = word
        self.prediction = pred


def predict_file(fileinput: str, model, output_mapping, window_type, window_size: int):
    """
    :param fileinput: path to inputfile that contains words
    :param model: NeuralNetwork
    :param output_mapping: {'B': [1 0 ...], 'E'...}
    :param window_type: function
    :param window_size: size of window
    :return: [Prediction("ablatives", ['B', ...]), Pred....]
    """

    predictions = []
    for word in mio.read_file(fileinput):
        w = word.strip(' \n')
        current = predict_word(model, w, output_mapping, window_type, window_size)
        predictions.append(Prediction(w, current))

    return predictions


def output_predictions(fileoutput: str, predictions: list):
    """
    :param fileoutput: path
    :param predictions: [Prediction(...), Pred...] list of predictions
    """

    with open(fileoutput, 'w') as ofile:
        for p in predictions:
            ofile.write(mio.START + '\n')
            for symbol in p.prediction:
                ofile.write(symbol + '\n')
            ofile.write(mio.STOP + '\n')


def output_segmentation(fileoutput: str, predictions: list):
    """
    :param fileoutput: path
    :param predictions: [Prediction(...), Pred...] list of predictions
    """

    with open(fileoutput, 'w') as ofile:
        for p in predictions:
            ofile.write(p.word + mio.SEPARATOR + mio.segment(p.word, p.prediction) + '\n')


def predict_word(model, X: str, output_mapping, window_type, window_size: int):
    """
    :param model: NeuralNetwork (Keras) model
    :param X: str (ex.: 'abounded')
    :param output_mapping: {'B': [1 0 ...], 'E'...}
    :param window_type: the windowing function itself
    :param window_size: size of the window
    :return ['B', 'E', 'B', ...]
    """

    symbols = [s for s in X]
    symbols.insert(0, mio.START_SIGN)
    symbols.append(mio.STOP_SIGN)

    symbol_matrix = mio.transform_input(symbols, window_type, window_size)
    prediction_matrix = model.predict(symbol_matrix)

    return mio.predictions_to_symbols(prediction_matrix, output_mapping)


def createNetwork(seed: int, layers: list):
    """
    :param seed: int
    :param layers: [Dense(...), ...]
    """

    np.random.seed(seed)
    NN = Sequential()
    for layer in layers:
        NN.add(layer)
    return NN


def createLayers(indim: int, outdim: int, init: str, activation: str, hidden: int):
    """
    :param indim: number of input nodes
    :param indim: number of output nodes
    :param init: name of initialization function (ex.: uniform)
    :param activation: name of activation function (ex.: relu)
    :param hidden: number of hidden layers in the network
    :return: [Layer, Layer, ...] list of Layer implementations
    """

    layers = []
    layers.append(Dense(input_dim=indim, output_dim=indim, init=init, activation=activation))
    layers.extend([Dense(output_dim=indim, init=init, activation=activation) for i in range(0, hidden)])
    layers.append(Dense(output_dim=outdim, init=init, activation=activation))
    return layers


def train_model(model, X, Y, epochs: int, batches: int, lossf: str, opt: str, callbacks=[], verbose=1):
    """
     :param model: NN-Model (ex.: Sequential)
     :param X: input list of arrays
     :param Y: output list of arrays
     :param epochs: number of rounds
     :param batches: number of samples per gradient update
     :param lossf: loss function to be used
     :param opt: optimizer to be used
     :param callbacks: [Callback, Callback, ...] list of Callback implementations
    """

    model.compile(loss=lossf, optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches, callbacks=callbacks, validation_split=0.1, verbose=verbose)


def build_and_train(input: list, verbose=1):
    """
    :param input: [['<s>', 'START'], ['a', 'B'], ...]
    :return model: NeuralNetwork (already trained)
    :return output_mapping: {'B': [1 0 ...], ...}
    """

    # process_training_input will apply a windowing function to the input just before
    # transforming it (list of symbols) into appropriately sized
    # vectors (the outputs will just simply be translated into one-hot vectors)
    input_matrix, input_mapping, output_matrix, output_mapping = \
        mio.process_training_input(input, WINDOW_TYPE, WINDOW_SIZE)

    # create and train NN
    indim = len(input_matrix[0])
    outdim = len(output_matrix[0])
    batches = len(input_matrix)
    activation = ACTIVATION
    optimizer = OPTIMIZER
    epochs = EPOCHS
    loss = LOSS
    init = INIT

    model = createNetwork(
        seed=561,
        layers=createLayers(indim, outdim, init, activation, HIDDEN_LAYER))

    callbacks = [EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE)]

    train_model(model, input_matrix, output_matrix, epochs, batches, loss, optimizer, callbacks, verbose=verbose)
    scores = model.evaluate(input_matrix, output_matrix, verbose=verbose)
    if verbose == 1:
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model, output_mapping


def read_training_input(path: str):
    """
    :param path: path to file
    :return: [['<s>', 'START'], ['a', 'B'], ...]
    """

    input = []
    for line in mio.read_file(path):
        input.append(line.strip('\n').split('\t'))

    return input


def train_predict_output(trainingpath: str, wordspath: str, output):
    """
    :param trainingpath: path to file
    :param wordspath: path to file
    :param output: function to use for outputtin
    """

    input = read_training_input(trainingpath)

    model, output_mapping = build_and_train(input)

    predictions = predict_file(wordspath, model, output_mapping, WINDOW_TYPE, WINDOW_SIZE)

    output(wordspath + '.PRED', predictions)


def benchmark(trainingpath: str, develpath: str, wordspath: str):
    """
    :param trainingpath: path to file containing the trainset words and their segmentations
    :param develpath: path to file containing the develset words and their segmentations
    :param wordspath: path to file containing the develset words
    """

    input = read_training_input(trainingpath)
    develset = read_training_input(develpath)
    expected_symbols = eval.create_word_blocks([x[1] for x in develset])

    print('=== Benchmark ===\n')
    for window_size in [2, 3, 4]:
        for window_type in [mio.use_left_window, mio.use_center_window, mio.use_right_window]:
            for hidden_layer in [1, 2, 3]:
                for epochs in [200, 300, 400]:
                    for activation in ['sigmoid', 'relu', 'tanh', 'softmax']:
                        for optimizer in ['adam', 'rmsprop']:
                            for loss in ['mean_squared_error', 'binary_crossentropy']:
                                for earlystop_patience in [10, 20]:
                                    global WINDOW_SIZE
                                    WINDOW_SIZE = window_size
                                    global WINDOW_TYPE
                                    WINDOW_TYPE = window_type
                                    global HIDDEN_LAYER
                                    HIDDEN_LAYER = hidden_layer
                                    global EPOCHS
                                    EPOCHS = epochs
                                    global ACTIVATION
                                    ACTIVATION = activation
                                    global OPTIMIZER
                                    OPTIMIZER = optimizer
                                    global LOSS
                                    LOSS = loss
                                    global EARLYSTOP_PATIENCE
                                    EARLYSTOP_PATIENCE = earlystop_patience

                                    model, output_mapping = build_and_train(input, verbose=0)

                                    predictions = predict_file(wordspath, model, output_mapping, WINDOW_TYPE, WINDOW_SIZE)
                                    pred_lists = [pred.prediction for pred in predictions]

                                    aggr_metric = eval.get_aggregated_metric(
                                                    eval.generate_metrics(expected_symbols, pred_lists))

                                    print(hyperparameters_tostring() + \
                                          '--> F-Score=' + str(aggr_metric.get_fscore()) + \
                                          ' Precision=' + str(aggr_metric.get_precision()) + \
                                          ' Recall=' + str(aggr_metric.get_recall()),
                                          flush=True)


def hyperparameters_tostring():
    """
    :return: the string representation of the current setup
    """

    return '[Window size: ' + str(WINDOW_SIZE) + '; ' +\
            'Window type: ' + str(WINDOW_TYPE.__name__) + '; ' +\
            'Hidden layers: ' + str(HIDDEN_LAYER) + '; ' +\
            'Epoch size: ' + str(EPOCHS) + '; ' +\
            'Activation: ' + ACTIVATION + '; ' +\
            'Optimizer: ' + OPTIMIZER + '; ' +\
            'Loss: ' + LOSS + '; ' +\
            'Initialization: ' + INIT + '; ' +\
            'Early stopping patience: ' + str(EARLYSTOP_PATIENCE) + ']'


def main():
    if len(sys.argv[1:]) != 2:
        raise Exception('Script needs 2 input-parameters (training samples, words to be predicted)')

    trainingpath = sys.argv[1:][0]
    wordspath = sys.argv[1:][1]

    #train_predict_output(trainingpath, wordspath, output_segmentation)
    benchmark(trainingpath, 'test_input/finn/bmes/goldstd_develset.segmentation', wordspath)


"""
 HYPER PARAMETERS
"""
WINDOW_SIZE = 2
WINDOW_TYPE = mio.use_left_window
HIDDEN_LAYER = 3
EPOCHS = 300
ACTIVATION = 'tanh'
OPTIMIZER = 'rmsprop'
LOSS = 'mean_squared_error'
INIT = 'uniform'
EARLYSTOP_PATIENCE = 15

#mio.END = 'M'
#mio.SINGLE = 'B'


if __name__ == "__main__":
    main()

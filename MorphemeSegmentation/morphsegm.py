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
        current = predict_word(model, word.strip(' \n'), output_mapping, window_type, window_size)
        current.insert(0, mio.START)
        current.append(mio.STOP)
        predictions.append(Prediction(word, current))

    return predictions


def output_predictions(fileoutput: str, predictions: list):
    """
    :param fileoutput: path
    :param predictions: [Prediction(...), Pred...] list of predictions
    """

    with open(fileoutput, 'w') as ofile:
        for p in predictions:
            for symbol in p.prediction:
                ofile.write(symbol + '\n')


def output_word_prediction_pairs(fileoutput: str, predictions: list):
    """
    :param fileoutput: path
    :param predictions: [Prediction(...), Pred...] list of predictions
    """

    with open(fileoutput, 'w') as ofile:
        for p in predictions:
                ofile.write(p.word + '\t' + p.prediction)


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


def train_model(model, X, Y, epochs: int, batches: int, lossf: str, opt: str, callbacks=[]):
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
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches, callbacks=callbacks, validation_split=0.1)


def build_and_train(param: str):
    """
    :param param: file-path
    :return model: NeuralNetwork (already trained)
    :return output_mapping: {'B': [1 0 ...], ...}
    """

    input = []
    for line in mio.read_file(param):
        input.append(line.strip('\n').split('\t'))

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

    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    train_model(model, input_matrix, output_matrix, epochs, batches, loss, optimizer, callbacks)
    scores = model.evaluate(input_matrix, output_matrix)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model, output_mapping


def main():
    if len(sys.argv[1:]) != 2:
        raise Exception('Script needs 2 input-parameters (training samples, words to be predicted)')

    trainingpath = sys.argv[1:][0]
    wordspath = sys.argv[1:][1]

    model, output_mapping = build_and_train(trainingpath)
    predictions = predict_file(wordspath, model, output_mapping, WINDOW_TYPE, WINDOW_SIZE)
    output_predictions(wordspath + '.PRED', predictions)


"""
 HYPER PARAMETERS
"""
WINDOW_SIZE = 3
WINDOW_TYPE = mio.use_left_window
HIDDEN_LAYER = 1
EPOCHS = 200
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
INIT = 'uniform'
mio.END = 'M'
mio.SINGLE = 'B'


if __name__ == "__main__":
    main()

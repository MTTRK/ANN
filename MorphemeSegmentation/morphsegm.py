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
from keras.layers.core import Dense
import numpy as np
import morph_io as mio


def createNetwork(seed, layers):
    """
    :param seed: int
    :param layers: [Dense(...), ...]
    """

    np.random.seed(seed)
    NN = Sequential()
    for layer in layers:
        NN.add(layer)
    return NN


def train_model(model, X, Y, epochs, batches, lossf, opt):
    """
     :param model: NN-Model (ex.: Sequential)
     :param X: input list of arrays
     :param Y: output list of arrays
     :param epochs: number of rounds
     :param batches: number of samples per gradient update
     :param lossf: loss function to be used
     :param opt: optimizer to be used
    """

    model.compile(loss=lossf, optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches)


def predict_word(model, X, output_mapping):
    """
    :param model: NeuralNetwork (Keras) model
    :param X: str (ex.: 'abounded')
    :param output_mapping: {'B': [1 0 ...], 'E'...}
    :return ['B', 'E', 'B', ...]
    """

    symbols = [s for s in X]
    symbols.insert(0, mio.START_SIGN)
    symbols.append(mio.STOP_SIGN)

    symbol_matrix = mio.transform_input(symbols, WINDOW_TYPE, WINDOW_SIZE)
    prediction_matrix = model.predict(symbol_matrix)

    return mio.predictions_to_symbols(prediction_matrix, output_mapping)


def build_and_train(param: str):
    """
    :param param: file-path
    :return model: NeuralNetwork (already trained)
    :return output_mapping: {'B': [1 0 ...], ...}
    """

    input = []
    for line in mio.read_file(param):
        input.append(line.strip('\n').split('\t'))

    # process_input will apply a windowing function to the input just before
    # transforming it (list of symbols) into appropriately sized
    # vectors (the outputs will just simply be translated into one-hot vectors)
    input_matrix, input_mapping, output_matrix, output_mapping = \
        mio.process_training_input(input, WINDOW_TYPE, WINDOW_SIZE)

    # create and train NN
    indim = len(input_matrix[0])
    outdim = len(output_matrix[0])
    activation = ACTIVATION
    optimizer = OPTIMIZER
    epochs = EPOCHS
    batches = len(input_matrix)
    loss = 'binary_crossentropy'
    init = 'uniform'

    model = createNetwork(
        seed=561,
        layers=
            [Dense(input_dim=indim, output_dim=indim, init=init, activation=activation)] +
            [Dense(output_dim=indim, init=init, activation=activation) for i in range(0, HIDDEN_LAYER)] +
            [Dense(output_dim=outdim, init=init, activation=activation)])

    train_model(model, input_matrix, output_matrix, epochs, batches, loss, optimizer)
    scores = model.evaluate(input_matrix, output_matrix)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model, output_mapping


def output_predictions(fileinput: str, model, output_mapping):
    """
    :param fileinput: path to inputfile
    :param model: NeuralNetwork
    :param output_mapping: {'B': [1 0 ...], 'E'...}
    """

    # do predictions for all the words
    predictions = []
    for word in mio.read_file(fileinput):
        current = predict_word(model, word.strip(' \n'), output_mapping)
        current.insert(0, mio.START)
        current.append(mio.STOP)
        predictions.append(current)

    # write out the results
    with open(fileinput + '.PREDICTIONS', 'w') as ofile:
        for prediction in predictions:
            for symbol in prediction:
                ofile.write(symbol + '\n')


def main():
    if len(sys.argv[1:]) != 2:
        raise Exception('Script needs 2 input-parameters (training samples, words to be predicted)')

    trainingpath = sys.argv[1:][0]
    wordspath = sys.argv[1:][1]

    model, output_mapping = build_and_train(trainingpath)
    output_predictions(wordspath, model, output_mapping)


"""
 HYPER PARAMETERS
"""
WINDOW_SIZE = 3
WINDOW_TYPE = mio.use_left_window
HIDDEN_LAYER = 1
EPOCHS = 200
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
#mio.END = 'M'
#mio.SINGLE = 'B'


if __name__ == "__main__":
    main()

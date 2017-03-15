"""
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
"""
import sys
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import morph_io as mio


def read_file(filepath):
    """
     filepath: str
    """

    with open(filepath, 'r') as file:
        for line in file:
            yield line


def read_stdin():
    for line in sys.stdin:
        yield line


def createNetwork(seed, layers):
    """
     seed: int
     layers: [Dense(...), ...]
    """

    np.random.seed(seed)
    NN = Sequential()
    for layer in layers:
        NN.add(layer)
    return NN


def train_model(model, X, Y, epochs, batches, lossf, opt):
    """
     model: NN-Model (ex.: Sequential)
     X: input list of arrays
     Y: output list of arrays
     epochs: number of rounds
     batches: number of samples per gradient update
     lossf: loss function to be used
     opt: optimizer to be used
    """

    model.compile(loss=lossf, optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches)


def predict(model, X, output_mapping):
    """
    :param model: NeuralNetwork (Keras) model
    :param X: str (ex.: 'abounded')
    :param output_mapping: {'B': [1 0 ...], 'E'...}
    """

    symbols = [s for s in X]
    symbols.insert(0, mio.START_SIGN)
    symbols.append(mio.STOP_SIGN)

    symbol_matrix = mio.transform_input(symbols, WINDOW_TYPE, WINDOW_SIZE)

    prediction_matrix = model.predict(symbol_matrix)

    return mio.predictions_to_symbols(prediction_matrix, output_mapping)


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
    input_matrix, input_mapping, output_matrix, output_mapping = \
        mio.process_training_input(input, WINDOW_TYPE, WINDOW_SIZE)

    # create and train NN
    indim = len(input_matrix[0])
    outdim = len(output_matrix[0])
    init = 'uniform'
    activation = 'sigmoid'
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    batches =len(input_matrix)
    epochs = EPOCHS

    model = createNetwork(
            seed=561,
            layers=[
                Dense(input_dim=indim, output_dim=indim, init=init, activation=activation),
                Dense(output_dim=indim, init=init, activation=activation),
                Dense(output_dim=outdim, init=init, activation=activation)])

    train_model(model, input_matrix, output_matrix, epochs, batches, loss, optimizer)
    scores = model.evaluate(input_matrix, output_matrix)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # test it with a word (from the training-set)
    print(predict(model, 'abounded', output_mapping))


"""
 HYPER PARAMETERS
"""
WINDOW_SIZE = 5
WINDOW_TYPE = mio.use_left_window
EPOCHS = 2000

if __name__ == "__main__":
    main()

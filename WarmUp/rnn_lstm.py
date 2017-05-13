"""
[1, 0, ..., 0] = 'a'    --> 1 (because its a vowel)
[0, 1, ..., 0] = 'b'    --> 0 (because its a consonant)
...
[0, 0, ..., 1] = 'z'    --> 0
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def create_model(seed):
    np.random.seed(seed)
    return Sequential()


def train_model(model, X, Y, epochs, batches, lossf, opt):
    model.compile(loss=lossf, optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches)


def add_lstm_layer(model, in_shape, cell_num):
    return model.add(
        LSTM(input_shape=in_shape,
            output_dim=cell_num))


def add_dense_layer(model, in_dim, out_dim, _activation):
    return model.add(
        Dense(input_dim=in_dim,
            output_dim=out_dim,
            activation=_activation)
    )


def predict(model, X):
    return model.predict(X)


def run():
    seed = 561
    alphabet = 26
    cell_num = 13
    epochs = 1000
    batches = 26

    dataX = np.identity(alphabet)
    X = np.reshape(dataX, (len(dataX), alphabet, 1))
    Y = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    model = create_model(seed)
    add_lstm_layer(model, [alphabet, 1], cell_num)
    add_dense_layer(model, cell_num, 1, 'sigmoid')
    train_model(model, X, Y, epochs, batches, 'binary_crossentropy', 'adam')

    scores = model.evaluate(X, Y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    predictions = predict(model, X)
    print([x.round()[0] for x in predictions])


run()

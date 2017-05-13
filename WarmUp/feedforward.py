"""
[1, 0, ..., 0] = 'a'    --> 1 (because its a vowel)
[0, 1, ..., 0] = 'b'    --> 0 (because its a consonant)
...
[0, 0, ..., 1] = 'z'    --> 0
"""
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np


def create_model(seed):
    np.random.seed(seed)
    return Sequential()


def add_dense_layer(model, _in_dim, _size, _init, _activation):
    return model.add(
        Dense(
            input_dim=_in_dim,
            output_dim=_size,
            init=_init,
            activation=_activation))


def predict(_model, X):
    return _model.predict(X)


def train_model(model, X, Y, epochs, batches, lossf, opt):
    model.compile(loss=lossf, optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=epochs, batch_size=batches)


def run():
    alphabet = 26
    epochs = 500
    batches = alphabet
    X = np.identity(alphabet)
    Y = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    model = create_model(seed=561)
    add_dense_layer(model, _in_dim=alphabet, _size=1, _init='uniform', _activation='sigmoid')
    train_model(model, X, Y, epochs, batches, 'binary_crossentropy', 'adam')

    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print([x.round()[0] for x in predict(model, X)])


run()

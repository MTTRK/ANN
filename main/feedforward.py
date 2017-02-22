from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

"""
[1, 0, ..., 0] = 'a'    --> 1 (because its a vowel)
[0, 1, ..., 0] = 'b'    --> 0 (because its a consonant)
...
[0, 0, ..., 1] = 'z'    --> 0
"""

in_dim = 26
hidden_layer_dim = 5
out_dim = 1
epochs = 500
random_seed = 561

X = np.identity(in_dim)
Y = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

model = Sequential()
model.add(
    Dense(
        input_dim=in_dim,
        output_dim=in_dim,
        init='uniform',
        activation='relu'))
model.add(
    Dense(
        output_dim=hidden_layer_dim,
        init='uniform',
        activation='relu'))
model.add(
    Dense(
        output_dim=out_dim,
        init='uniform',
        activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=epochs, batch_size=in_dim)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
rounded = [x.round()[0] for x in predictions]
print(rounded)

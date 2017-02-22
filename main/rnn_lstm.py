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


random_seed = 561
alphabet = 26
cell_num = 16
epochs = 1000
batch_size = 26

np.random.seed(random_seed)
dataX = np.identity(alphabet)
X = np.reshape(dataX, (len(dataX), alphabet, 1))
Y = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

model = Sequential()
model.add(
    LSTM(
        input_shape=[alphabet, 1],
        output_dim=cell_num))
model.add(
    Dense(
        output_dim=1,
        activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size)

scores = model.evaluate(X, Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict(X)
rounded = [x.round()[0] for x in predictions]
print(rounded)

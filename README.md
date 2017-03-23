# College Project about Artificial Neural Networks

## WarmUp
This project contains some exercises that help to embrace the theory of ANN-s, especially RNN-s.
Also the use of 3rd party libraries like Keras will appear in the code.

## MorphemeSegmentation
Morphological segmentation, which aims to break words into meaning-bearing morphemes, is an important task in natural language processing.
Here we are going to concern ourselves with the following languages: english, hungarian, etc.

### prep_data.py
Prepares the training data for the ANN
```
mt:MorphemeSegmentation MT$ head -n 4 test_input/en/goldstd_trainset.words_and_segments
ablatives       ablative s
abounded        abound ed
abrogate        abrogate
abusing ab us ing
...
mt:MorphemeSegmentation MT$ cat test_input/en/goldstd_trainset.words_and_segments | python3.5 prep_data.py > file
mt:MorphemeSegmentation MT$ head -n 50 file
<s>     START
a       B
b       M
l       M
a       M
t       M
i       M
v       M
e       E
s       S
</s>    STOP
<s>     START
a       B
b       M
o       M
u       M
n       M
d       E
e       B
d       E
</s>    STOP
<s>     START
a       B
b       M
r       M
o       M
g       M
a       M
t       M
e       E
</s>    STOP
<s>     START
a       B
b       E
u       B
s       E
i       B
n       M
g       E
</s>    STOP
...
```

### morphsegm.py
* Builds and trains a neural network using the input data in the format
provided by *prep_data.py*. Single words will be classified alone, the START-STOP tokens
mark the beginning and ending of these.
* Once the NN-model is 'ready to go', it will be used to classify word-segments (combined set of the
training and validation lists of words)
```
mt:MorphemeSegmentation MT$ python3.5 morphsegm.py test_input/en/bm/goldstd_trainset.segmentation test_input/en/goldstd_combined.words
Using TensorFlow backend.
Epoch 1/200
8726/8726 [==============================] - 0s - loss: 3.6211 - acc: 0.7500
Epoch 2/200
8726/8726 [==============================] - 0s - loss: 1.5465 - acc: 0.7500
...
Epoch 200/200
8726/8726 [==============================] - 0s - loss: 0.1643 - acc: 0.9430
8384/8726 [===========================>..] - ETA: 0sacc: 94.31%
```
The output is in the following format:
```
mt:en MT$ ls -l
total 328
drwxr-xr-x  6 MT  staff    204 Mar 23 22:28 bm
drwxr-xr-x  6 MT  staff    204 Mar 23 22:24 bmes
-rw-r--r--  1 MT  staff  16004 Mar 23 18:28 goldstd_combined.words
-rw-r--r--  1 MT  staff  47182 Mar 23 22:35 goldstd_combined.words.PREDICTIONS
-rw-r--r--  1 MT  staff  36880 Mar 21 20:44 goldstd_combined.words_and_segments_with_duplicates
-rw-r--r--  1 MT  staff  34177 Mar 23 20:45 goldstd_combined.words_and_segments_without_duplicates
-rw-r--r--  1 MT  staff  21734 Mar  8 23:44 goldstd_trainset.words_and_segments
mt:en MT$ head -n 20 goldstd_combined.words.PREDICTIONS
START
B
M
M
M
M
M
M
M
M
STOP
START
B
M
M
M
M
M
M
M
mt:en MT$
```

### evaluate.py
Now that we have some results and can produce new ones, we need a way be able to assess the quality
of the predictions. F-Score will come to our help:
* ![equation](http://latex.codecogs.com/gif.latex?F%20%3D%202%20%5Ctimes%20%5Cfrac%20%7Bprecision*recall%7D%7Bprecision&plus;recall%7D)
* ![equation](http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%20%7BTP%7D%7BTP&plus;FN%7D)
* ![equation](http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%20%7BTP%7D%7BTP&plus;FP%7D)

TN = True Negative, TP = True Positive, FN = False Negative (should have found these), FP = False Positive (shouldn't have found these)
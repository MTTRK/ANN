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
...
```

### morphsegm.py
* Builds and trains a neural network using the input data in the format
provided by *prep_data.py*. Single words will be classified alone, the START-STOP tokens
mark the beginning and ending of these.
* Once the NN-model is 'ready to go', it will be used to classify word-segments
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
STOP
START
B
M
M
M
M
M
mt:en MT$
```

### evaluate.py
Now that we have some results and can produce new ones, we need a way to be able to assess the quality
of the predictions. In case of binary decisions (B,M) F-Score can help us out.
In case of (B,M,E,S), we will consider (B,S) -> True and (E,M) -> False. Then we can apply F-Score.
```
Average F-Score=0.7572712758572836 Precision=0.9348396501457724 Recall=0.682142857142857
Aggregated F-Score=0.746617915904936 Precision=0.9011473962930273 Recall=0.6373283395755306

Per word:
0th word --> F-Score=0.5 Precision=1.0 Recall=0.3333333333333333
1th word --> F-Score=0.5 Precision=1.0 Recall=0.3333333333333333
2th word --> F-Score=0.5 Precision=1.0 Recall=0.3333333333333333
3th word --> F-Score=0.6666666666666666 Precision=1.0 Recall=0.5
4th word --> F-Score=0.8571428571428571 Precision=1.0 Recall=0.75
5th word --> F-Score=1.0 Precision=1.0 Recall=1.0
6th word --> F-Score=1.0 Precision=1.0 Recall=1.0
7th word --> F-Score=0.8 Precision=1.0 Recall=0.6666666666666666
8th word --> F-Score=0.4 Precision=1.0 Recall=0.25
9th word --> F-Score=0.6666666666666666 Precision=1.0 Recall=0.5
...
```
The above metrics show how well the network could recognize the beginning of words


## Benchmark
Finding the best set of hyperparameters is not an easy task. We are going to try to find
it by brute-forcing our way through the different combinations of these:
* WINDOW_SIZE
* WINDOW_TYPE
* HIDDEN_LAYER
* EPOCHS
* ACTIVATION
* OPTIMIZER
* LOSS
* INIT
* EARLYSTOP_PATIENCE
```
=== Benchmark ===

[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: sigmoid; Optimizer: sgd; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.0 Precision=0.0 Recall=0.0
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: sigmoid; Optimizer: sgd; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.0 Precision=0.0 Recall=0.0
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: sigmoid; Optimizer: sgd; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.0 Precision=0.0 Recall=0.0
...
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: adam; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.0 Precision=0.0 Recall=0.0
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: adam; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.0 Precision=0.0 Recall=0.0
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: adam; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: adam; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: rmsprop; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: rmsprop; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: rmsprop; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 150; Activation: relu; Optimizer: rmsprop; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
...
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 250; Activation: tanh; Optimizer: adam; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 250; Activation: tanh; Optimizer: adam; Loss: binary_crossentropy; Initialization: uniform; Early stopping patience: 10]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
[Window size: 2; Window type: use_left_window; Hidden layers: 1; Epoch size: 250; Activation: tanh; Optimizer: adam; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 5]--> F-Score=0.999985000124999 Precision=0.9999900000999989 Recall=0.9999900000999989
...
```
The benchmarking we conducted involved the **{B, M, E, S}** classes (classes for the characters [begin, middle, end, single]),
**test_input/en/bmes/goldstd_trainset.segmentation** and the **test_input/en/goldstd_develset.words**.

Although the numbers look good sometimes (F-Score=0.99...), this still sadly is a poor performance. The endings barely get recognized. The evaluation considers ['B', 'S'] TRUE and ['E', 'M'] FALSE.


## Inference
After a Train & Prediction run:
```
mt:en MT$ ll
total 360
-rw-r--r--  1 MT  staff  30740 Apr  1 16:25 benchmark.txt
drwxr-xr-x  4 MT  staff    136 Mar 31 22:11 bm
drwxr-xr-x  4 MT  staff    136 Mar 31 22:28 bmes
-rw-r--r--  1 MT  staff  16003 Mar 29 01:00 goldstd_combined.words
-rw-r--r--  1 MT  staff  36880 Mar 21 20:44 goldstd_combined.words_and_segments
-rw-r--r--  1 MT  staff   6568 Mar 29 01:00 goldstd_develset.words
-rw-r--r--  1 MT  staff  13580 Apr  1 16:29 goldstd_develset.words.PRED
-rw-r--r--  1 MT  staff  15145 Mar 29 00:59 goldstd_develset.words_and_segments
-rw-r--r--  1 MT  staff  14053 Mar 29 00:59 goldstd_develset.words_and_segments_without_duplicates
-rw-r--r--  1 MT  staff   9434 Mar 29 01:00 goldstd_trainset.words
-rw-r--r--  1 MT  staff  21734 Mar  8 23:44 goldstd_trainset.words_and_segments
mt:en MT$ paste goldstd_develset.words_and_segments_without_duplicates goldstd_develset.words.PRED | cut -f1,2,4 > goldstd_develset_INFERENCE
mt:en MT$ head goldstd_develset_INFERENCE 
accompanied	ac compani ed	accompanied
accompaniment	ac compani ment	accompaniment
acknowledging	ac knowledg ing	acknowledging
acquisition	acquis ition	acquisition
acquisitions'	acquis ition s '	acquisition s '
acupuncture	acupuncture	acupuncture
acupuncture's	acupuncture 's	acupuncture 's
adjudged	ad judg ed	ad judged
advantageously	advant age ous ly	advantageously
afire	a fire	afire
```
This file can be found here: **test_input/en/**

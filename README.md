# College Project on Artificial Neural Networks

## WarmUp
This project contains some exercises that help to embrace the theory of ANN-s, especially RNN-s.
Also the use of 3rd party libraries like Keras appears in the code.

## MorphemeSegmentation
Morphological segmentation, which aims to break words into meaning-bearing morphemes, is an important task in natural language processing.
Here we are going to concern ourselves with the following languages: english, hungarian, etc.

### morphsegm.py
* Builds and trains a neural network.
* Once the NN-model is 'ready to go', it will be used to classify word-segments
```
$ python3.5 morphsegm.py 
Using TensorFlow backend.
usage: morphsegm.py [-h] -t TRAINING -w WORDS [-b] [-bmes] [-d DEVEL]
                    [-ws WINDOWSIZE] [-wt WINDOWTYPE] [-hl HIDDENLAYER]
                    [-ep EPOCHS] [-ac ACTIVATE] [-op OPTIMIZE] [-lo LOSS]
                    [-it INIT] [-es EARLYSTOP]
morphsegm.py: error: the following arguments are required: -t/--training, -w/--words
```
```
python3.5 morphsegm.py -t test_input/en/goldstd_trainset.words_and_segments -w test_input/en/goldstd_develset.words
Using TensorFlow backend.
accompanied	accompani ed
accompaniment	accompaniment
acknowledging	acknowledg ing
...
```

### evaluate.py
Takes 2 files in goldstd format and determines the F-Score like this:
```
$ python3.5 evaluate.py 
usage: evaluate.py [-h] -e EXPECTED -a ACTUAL
evaluate.py: error: the following arguments are required: -e/--expected, -a/--actual
```
```
$ python3.5 evaluate.py -e expected -a actual
Average F-Score=0.5819930697770022 Precision=0.6282991486638425 Recall=0.5761861618361995
Aggregated F-Score=0.7596568830836983 Precision=0.8499999885135137 Recall=0.6866812152109038

Per word:
1th word --> F-Score=0.6666577778296295 Precision=0.9999900000999989 Recall=0.49999750001249993
2th word --> F-Score=0.0 Precision=0.0 Recall=0.0
...
```
The above metrics show how well the network could recognize the beginning of words.

## Benchmark
Finding the best set of hyperparameters is not an easy task. We are going to try to find
it by brute-forcing our way through the different combinations of these:
* size of window
* type of window
* number of hidden layers
* number of epochs
* activation function
* optimization method
* loss function
* initialization method
* early-stopping patience
```
=== Benchmark ===

[Window size: 2; Window type: 0; Hidden layers: 1; Epoch size: 200; Activation: sigmoid; Optimizer: adam; Loss: mean_squared_error; Initialization: uniform; Early stopping patience: 50]--> F-Score=0.0 Precision=0.0 Recall=0.0
...
```

## Hungarian corpus
I cleaned the data using the following tools:
```
date ; zgrep -vE "(\[|\]|\}|\{|@|#|\\^|\\$|<|>|_|\+|\*|\||\\\|\?|\/|\%|\!|\:|\;|\.|\,|[0-9]|\"|\(|\))" webcorp.100M.segmented.gz | tr 'ÍÖÜÓŐÚŰÉÁ[:upper:]' 'íöüóőúűéá[:lower:]' | sort -u | head -n -254 | tail -n +400 | grep -vE '\-\-' | sort -R | gzip > webcorp.100M.segmented_filtered.gz ; date

zcat webcorp.filtered.2.4M.segmented.gz | sed 's/\(.\)/\1\n/g' | sort -u | tr '\n' ','
<<GIVES BACK EVERY CHARACTER USED>>

zgrep -vE "[<<EVERY NON-ENGLISH/HUNGARIAN CHARACTER>>]" webcorp.filtered.2.4M.segmented.gz | gzip > webcorp.filtered.2.4M.segmented.gz2
```
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
...
```

### evaluate.py
Takes 2 files in goldstd format and determines the F-Score like this:
```
...
```
The above metrics show how well the network could recognize the beginning of words


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
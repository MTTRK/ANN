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
mt:MorphemeSegmentation MT$ head -n 4 test_input/goldstd_trainset.segments.eng
ablatives       ablative s
abounded        abound ed
abrogate        abrogate
abusing ab us ing
...
mt:MorphemeSegmentation MT$ cat test_input/goldstd_trainset.segments.eng | python3.5 prep_data.py > file
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
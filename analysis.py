import sys, os, re, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.neural_network import MLPClassifier


train = pd.read_csv('data/reddit_train.csv', encoding = 'latin-1')
test = pd.read_csv('data/reddit_test.csv', encoding = 'latin-1')

y = train['REMOVED'].values
y_te = test['REMOVED'].values

# tempY= []
# tempYTe = []
# for val in y:
#     if (val):
#         tempY.append(1)
#     else:
#         tempY.append(0)
# y = tempY
# for val in y_te:
#     if (val):
#         tempYTe.append(1)
#     else:
#         tempYTe.append(0)
# y_te=tempYTe

list_sentences_train = train["BODY"]
list_sentences_test = test["BODY"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

clf = MLPClassifier(activation='relu', solver='lbfgs', learning_rate='constant')
print("training data")
clf.fit(X_t,y)

score = clf.score(X_te,y_te)
print(score)
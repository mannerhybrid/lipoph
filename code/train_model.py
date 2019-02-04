import os
import re
import sys
import gensim
import random
import pickle
import nltk
import itertools
from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
import pandas as pd 
import numpy as np 
from Bio import Entrez
from random import shuffle
import tensorboard as board
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from ast import literal_eval
from gensim import corpora
from collections import defaultdict
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as soup
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense,Dot,Input, Embedding, Reshape, LSTM, Concatenate, Conv1D, Reshape
from keras.datasets import boston_housing
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pprint import pprint

from time import time
from keras.utils import Progbar
import os
from datetime import datetime

tf.enable_eager_execution()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from model import LipophRecommender
from preprocessing import *

maindict = corpora.Dictionary.load("..\\data\\maindict.dict")

seed = 7
np.random.seed(42)
p=0.75
metadata = pd.read_csv("..\\data\\absrecord.csv", encoding='latin-1')

metadata['keywords'] = metadata['keywords'].apply(literal_eval)
metadata['keywords'] = metadata['keywords'].apply(get_list)
metadata['keywords'].replace('[]', np.nan, inplace=True)
metadata = metadata.dropna(subset=['keywords'])

x_train_titles = [preprocess(str(text))[0] for text in metadata['title'].values]
x_train_body = [preprocess(str(text))[0] for text in metadata['body'].values]
x_train_keywords = metadata['keywords']

# nphrases = [extractNounPhrases(text) for text in metadata['title'].values]

indexes = list(range(len(x_train_titles)))

vectors_titles = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_titles]
vectors_bodies =  [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_body]
vectors_keywords = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_keywords]

indexes = list(range(len(vectors_titles)))
vectors_titles = [np.expand_dims(v,0) for v in vectors_titles]
vectors_bodies = [np.expand_dims(v,0) for v in vectors_bodies]
vectors_keywords = [np.expand_dims(v,0)  for v in vectors_keywords]

y_train = [np.ones_like(l) for l in vectors_keywords]

main_data = list(zip(vectors_titles,vectors_bodies, vectors_keywords, y_train))
print("Before Processing: ", len(main_data))

main_data = remove_nan(main_data); print(len(main_data))
vectors_keywords, y_train = negative_sampling(vectors_keywords, y_train)
main_data = replicate_lines(main_data); print(len(main_data))

random.shuffle(main_data)
p = 0.7

train_data = main_data[:int(p*len(main_data))]
x_train = [(a,b,c) for a,b,c,_ in train_data]
y_train = [d for _,_,_,d in train_data]

test_data = main_data[int(p*len(main_data)):]
x_test = [(a,b,c) for a,b,c,_ in test_data]
y_test = [d for _,_,_,d in test_data] 

x_trainset = x_train[:10]
y_trainset = y_train[:10]

sample = dict(x=x_trainset, y=y_trainset)

checkpoint_directory = '..\\model'
with open('..\\data\\datasample', 'wb') as sample_file:
    pickle.dump(sample, sample_file)

os.system('cls')

model = ""
model = LipophRecommender(maindict)
sgd_optimizer = tf.train.AdamOptimizer(0.05)
model.compile(sgd_optimizer)
history = model.fit(x_train, y_train, sgd_optimizer, num_epochs=10)

print('============================================================================')

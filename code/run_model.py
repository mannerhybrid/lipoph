import os
import re
import sys
import gensim
import random
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
import pickle
from time import time
from keras.utils import Progbar
import os
from datetime import datetime
import win32com.client as wincl

speak = wincl.Dispatch("SAPI.SpVoice")
tf.enable_eager_execution()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from model import LipophRecommender
from preprocessing import *

checkpoint_directory = '..\\model'
maindict = corpora.Dictionary.load("..\\data\\maindict.dict") 

model = LipophRecommender(maindict)
adam_opt = tf.train.AdamOptimizer(0.05)

with open('..\\data\\datasample', 'rb') as df:
    data = pickle.load(df)

x_train = data['x'][:10]; y_train = data['y'][:10]

model.fit(x_train, y_train, adam_opt, num_epochs=1)

saver = tfe.Saver(model.variables)
saver.restore(tf.train.latest_checkpoint(checkpoint_directory))

os.system('cls')

pmid = input('Enter PMID: \n')
title, abstract = miner(pmid)
pprint(title)
pprint(abstract)

all_keywords = input('Enter keywords: \n')

title, abstract, keywords = map(preprocess, [str(title),str(abstract),str(all_keywords)])

v_titles = [np.array(maindict.doc2idx(word, unknown_word_index=len(maindict))) for word in title][0]
v_bodies =  [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in abstract][0]
v_keywords = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in keywords][0]
# print(v_keywords)

v_titles = np.expand_dims(v_titles,0)
v_bodies = np.expand_dims(v_bodies,0)
v_keywords = np.expand_dims(v_keywords,0) 
# print(v_keywords)

v_titles = [np.expand_dims(v,0) for v in v_titles]
v_bodies = [np.expand_dims(v,0) for v in v_bodies]

def replicate_lines(train_data):
  _data = []
  for record in train_data:
    
    numkw = len(record[-1])
    for i in range(numkw):
        _data.append((record[0], record[1], np.expand_dims(record[2][i],0)))
  return _data

data = list(zip(v_titles, v_bodies, v_keywords))
data = replicate_lines(data)

def obtain_similarity_score(data, batch_size=5000, model=model):
  scores = []
  datagen = batch_generator(data, batch_size)
  num_epochs = 1 + len(data) // batch_size
  for i in range(num_epochs):
    temp = model.predict(next(datagen))
    scores.extend(temp)
  return scores

def softmax(arr):
  return np.divide(np.exp(arr), np.sum(np.exp(arr)))

sim = obtain_similarity_score(data)
similarities = list(map(lambda x:round(x.numpy()[0]*100,2), sim))

if len(keywords) == 1:
  similarities = similarities
else:
  similarities = softmax(similarities)

keywords = [k for kw in keywords for k in kw]
results = list(zip(keywords, similarities))
results = sorted(results, key=lambda x:x[1], reverse=True)
print(*results, sep='\n')
speak.Speak('The most relevant word is {} with a similarity score of {} %.'.format(results[0][0], results[0][1]))

keywords = v_keywords[0]

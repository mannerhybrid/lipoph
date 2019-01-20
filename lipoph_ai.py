# -*- coding: utf-8 -*-
"""Lipoph-AI.ipynb
"""

import os
import re
import sys
import gensim
import nltk
from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
import pandas as pd 
import numpy as np 
from random import shuffle
import tensorboard as board
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from ast import literal_eval
from gensim import corpora
from collections import defaultdict
import matplotlib.pyplot as plt
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

tf.enable_eager_execution()
nltk.download('wordnet')

"""#Preprocessing"""

def remove_stopwords(text):
    stopwords = [word.replace('\n', '') for word in open("stopwords.txt", encoding='latin-1').readlines()]
    text = ' '.join(w for w in text.split() if w not in stopwords and not len(w) == 1)
    return text

def preprocess(text):
    sent_tokenizer = PunktSentenceTokenizer()
    
    sentences = [sentence.lower() for sentence in sent_tokenizer.tokenize(text)]
    sentences = [re.sub(r'\([\W\w\d\D%=.,]+\)', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'\.', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'-', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\d\D]\.[\d\D]+', '', sentence) for sentence in sentences]
    sentences = [re.sub(r' \d\D ', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'[;,:-@#%&\"\'�]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\(\)\[\]\+\*\/]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[��]', ' ', sentence) for sentence in sentences]
    sentences = [remove_stopwords(sent) for sent in sentences]
    sentences = " __END__ ".join(sentences)
    sentences = sentences.split(" __END__ ")
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    texts = [gensim.corpora.textcorpus.remove_stopwords([lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(sent)]) for sent in sentences]
    frequency = defaultdict(int)
    return texts

def tokenize(sentences):
    sentences = sentences.split(" __END__ ")
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    texts = [gensim.corpora.textcorpus.remove_stopwords([lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(sent)]) for sent in sentences]
    frequency = defaultdict(int)
    return texts

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names

    #Return empty list in case of missing/malformed data
    return []

"""#Form Training and Validation Data"""

seed = 7
np.random.seed(42)
p=0.75
metadata = pd.read_csv("absrecord.csv", encoding='latin-1')

metadata['keywords'] = metadata['keywords'].apply(literal_eval)
metadata['keywords'] = metadata['keywords'].apply(get_list)
metadata['keywords'].replace('[]', np.nan, inplace=True)
metadata = metadata.dropna(subset=['keywords'])

x_train_titles = [preprocess(str(text))[0] for text in metadata['title'].values]
x_train_body = [preprocess(str(text))[0] for text in metadata['body'].values]
x_train_keywords = metadata['keywords']
indexes = list(range(len(x_train_titles)))

maindict = corpora.Dictionary.load("maindict.dict")

def negative_sampling(vk, y, maindict=maindict):
  vkns = []; yns =[]
  keys = np.array(maindict.keys())
  for kwords, label in list(zip(vk, y)):
    absent_keys = np.setdiff1d(keys, kwords)
    kwords_noise = np.random.choice(absent_keys, [1,kwords.shape[1]])
    label_noise = np.zeros_like(label)
    kwords_new = np.concatenate((kwords,kwords_noise), axis=1)
    labels_new = np.concatenate((label, label_noise), axis=1)
    vkns.append(np.array(kwords_new)); yns.append(labels_new)
  return vkns, yns

def remove_nan(train_data):
  error=1
  i = 0
  outdata = []
  for record in train_data:
#     print(record[3])
    if not record[2] == []:
      outdata.append(record)
  return outdata

def replicate_lines(train_data):
  _data = []
  for record in train_data:
    numkw = len(record[-1][0])
    for i in range(numkw):
        _data.append((record[0], record[1], np.expand_dims(record[2][0][i],0), np.expand_dims(record[3][0][i],0)))
  return _data

vectors_titles = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_titles]
vectors_bodies =  [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_body]
vectors_keywords = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_keywords]

"""# Continue Preprocessing for Prediction"""

vectors_titles = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_titles]
vectors_bodies =  [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_body]
vectors_keywords = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_keywords]

indexes = list(range(len(vectors_titles)))
vectors_titles = [np.expand_dims(v,0) for v in vectors_titles]
vectors_bodies = [np.expand_dims(v,0) for v in vectors_bodies]
vectors_keywords = [np.expand_dims(v,0)  for v in vectors_keywords]

y_train = [np.ones_like(l) for l in vectors_keywords]

vectors_keywords, y_train = negative_sampling(vectors_keywords, y_train)

main_data = list(zip(vectors_titles,vectors_bodies, vectors_keywords, y_train))
print("Before Processing: ", len(main_data))

main_data = remove_nan(main_data); print(len(main_data))
main_data = replicate_lines(main_data); print(len(main_data))

import random
random.shuffle(main_data)

# main_data = main_data[:5000]

train_data = main_data[:int(p*len(main_data))]
x_train = [(a,b,c) for a,b,c,_ in train_data]
# print(train_data[0])
y_train = [d for _,_,_,d in train_data]

test_data = main_data[int(p*len(main_data)):]
x_test = [(a,b,c) for a,b,c,_ in test_data]
y_test = [d for _,_,_,d in test_data] 

# print(vectors_keywords)
print(type(x_train))
print(type(x_train[0]))

"""#Putting it all together"""

from keras.utils import Progbar

import tensorflow as tf
from time import time
from keras.utils import Progbar

class LipophRecommender(tf.keras.Model):
  def __init__(self, maindict):
      super(LipophRecommender, self).__init__()
      init_emb = tf.random.normal([len(maindict)+1,100], stddev=0.2, dtype=tf.float32)
      self.embedding = tfe.Variable(init_emb)
      # self.lstm_cell = tf.keras.layers.CuDNNLSTM(50)
      self.lstm_cell = tf.keras.layers.LSTM(50)
      self.red_dim = tf.keras.layers.Dense(50)
      self.second_dense = tf.keras.layers.Dense(1, activation='sigmoid')

  def form_lstm(self, x):
    return tf.expand_dims(self.lstm_cell(x)[-1],0)
      
  def embedding_lookup(self, x):
    return tf.nn.embedding_lookup(self.embedding, x)
  
  def compute_similarity(self, vt, va, vk):
    simtk = vt @ tf.transpose(vk)
    simta = va @ tf.transpose(vk)
    simvec = tf.transpose(tf.concat([simtk, simta], axis=0))
    return tf.transpose(self.second_dense(simvec))[0]
  
  def compute_loss(self, data, labels):
    loss = 0
    iters = 0
    logits = self.predict(data)
    for logit, label in list(zip(logits, labels)):
      iters+=1
#       print(iters, logit, label)
      loss += tf.losses.mean_squared_error(logit, label)
    final_loss = loss/iters
    print(final_loss)
    return final_loss
  
  def predict(self, data):
    iters = 0
    logits_all = []
    progbar = Progbar(len(data))
    for x_title_train, x_abstract_train, x_keywords_train in data:
      iters += 1
      title_h1 = self.embedding_lookup(x_title_train)
      abstract_h1 = self.embedding_lookup(x_abstract_train)
      keywords_h1 = self.embedding_lookup(x_keywords_train)
      title_h2 = self.form_lstm(title_h1)
      abstract_h2 = self.form_lstm(abstract_h1)
      keywords_h2 = self.red_dim(keywords_h1)
      logits = self.compute_similarity(title_h2, abstract_h2, keywords_h2)
      logits_all.append(tf.reshape([logits], [1,]))
      progbar.update(iters)
    return logits_all

  def grads_fn(self, data, labels):
    with tfe.GradientTape() as tape:
      loss = self.compute_loss(data, labels)
    return tape.gradient(loss, self.variables), loss

  def fit(self, data, target, optimizer,  num_epochs=2, verbose=2):
    
    for i in range(num_epochs):
      t0 = time()
      print("Calculating gradients for epoch {}\n".format(i+1))
      grads, loss_main = self.grads_fn(data, target)
#       optimizer=optimizer.minimize(loss_main)
      t1 = time()
      print("Time elapsed: {} h {} m {}s\n".format((t1-t0)//3600, ((t1-t0)%3600)//60, (t1-t0)%60))

      print("Applying gradients for epoch {}\n".format(i+1))
      optimizer.apply_gradients(zip(grads, self.variables))
      t2 = time()
      print("Time elapsed: {} h {} m {}s\n".format((t2-t1)//3600, ((t2-t1)%3600)//60, (t2-t1)%60))

      print('\nLoss at epoch %d: %f\n' %(i+1, loss_main))
      
#       callbacks.save("epoch_{}_checkpoint.ckpt".format(i+1))

print(x_train[0])

model = ""
model = LipophRecommender(maindict)
sgd_optimizer = tf.train.AdamOptimizer(0.05)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
model.compile(sgd_optimizer)

model = LipophRecommender(maindict)
sgd_optimizer = tf.train.AdamOptimizer(0.05)
model.compile(sgd_optimizer)
history = model.fit(data=x_train, target=y_train, optimizer=sgd_optimizer, num_epochs=20)

# https://toolbox.google.com/datasetsearch
import os

if not os.path.exists('models_checkpoints/'):
  os.makedirs('models_checkpoints/')
checkpoint_directory = 'models_checkpoints/Lipoph/'
# Create model checkpoint
checkpoint = tfe.Checkpoint(optimizer=sgd_optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())

from datetime import datetime 

current = datetime.now()

ckpt_filename = "model_{}_o{}_e{}_r{}.ckpt".format(current.strftime("%d%b"), 'Adam', 20, len(x_train))
print(ckpt_filename)

checkpoint.save(file_prefix=os.path.join(checkpoint_directory, ckpt_filename))

"""#Model Evaluation"""

checkpoint_directory = 'models_checkpoints/Lipoph/'
checkpoint = tfe.Checkpoint(optimizer=sgd_optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

model.compute_loss(x_test,y_test)
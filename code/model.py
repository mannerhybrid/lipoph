import os
import re
import sys
import gensim
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
from time import time
from keras.utils import Progbar
import os
from datetime import datetime
from preprocessing import *

tf.enable_eager_execution()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

maindict = corpora.Dictionary.load("..\\data\\maindict.dict")

class LipophRecommender(tf.keras.Model):
  def __init__(self, maindict):
    super(LipophRecommender, self).__init__()
    init_emb = tf.random.normal([len(maindict)+1,100], stddev=0.2, dtype=tf.float32)
    self.embedding = tfe.Variable(init_emb, name='embedding')
    self.lstm_title = tf.keras.layers.CuDNNLSTM(50, name='cudnn_title_lstm')
    self.lstm_abs = tf.keras.layers.CuDNNLSTM(50, name='cudnn_abstract_lstm')
    self.red_dim = tf.keras.layers.Dense(50, activation='relu', name='red_dim')
    self.second_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='second_dense')
    
  def title_lstm(self, x):
    return tf.expand_dims(self.lstm_title(x)[-1],0)

  def abstract_lstm(self, x):
    return tf.expand_dims(self.lstm_abs(x)[-1],0)
  
  def embedding_lookup(self, x):
    return tf.nn.embedding_lookup(self.embedding, x)
  
  def compute_similarity(self, vt, va, vk):
    vt_normalized = tf.nn.l2_normalize(vt,0)
    va_normalized = tf.nn.l2_normalize(va,0)
    vk_normalized = tf.nn.l2_normalize(vk,0)
    simtk = tf.reduce_sum(tf.multiply(vt_normalized, vk_normalized)).numpy()
    simak = tf.reduce_sum(tf.multiply(va_normalized, vk_normalized)).numpy()
    simvec = tf.expand_dims(tfe.Variable([simtk, simak]),0)
    return tf.transpose(self.second_dense(simvec))[0]
  
  def compute_loss(self, data, labels):
    loss = 0
    iters = 0
    logits = self.predict(data)
    final_loss = tf.sqrt(tf.losses.mean_squared_error(logits, labels))
    print('Loss: ', final_loss.numpy())
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
      title_h2 = self.title_lstm(title_h1)
      abstract_h2 = self.abstract_lstm(abstract_h1)
      keywords_h2 = self.red_dim(keywords_h1)
      logits = self.compute_similarity(title_h2, abstract_h2, keywords_h2)
      logits_all.append(tf.reshape([logits], [1,]))
    #   progbar.update(iters)
    return logits_all

  def grads_fn(self, data, labels):
    with tfe.GradientTape() as tape:
      loss = self.compute_loss(data, labels)

    return tape.gradient(loss, self.variables), loss

  def fit(self, data, target, optimizer, minibatch_size=5000, num_epochs=10, verbose=2):
    print("Creating generators")
    checkpoint_directory = '..\\models_checkpoints'
    checkpoint_num = 0
    
    p = 0.2
    cutoff = int(0.2*len(data))
    
    test_x_gen = batch_generator(data[-cutoff:], 100)
    test_y_gen = batch_generator(target[-cutoff:],100)

    if not os.path.exists(checkpoint_directory):
      os.makedirs(checkpoint_directory)
    print("Generators made")
    
    epoch_losses = []
    for i in range(num_epochs):
      print("Epoch {}".format(i+1))
      datagen = batch_generator(data, minibatch_size)
      targetgen = batch_generator(target, minibatch_size)
      t0 = time()
      total_mbs = 1 + len(data)//minibatch_size
      minibatch_losses = []
      for j in range(1 + len(data)//minibatch_size):
        
        global_step = tf.train.get_or_create_global_step()
        grads, loss_main = self.grads_fn(next(datagen), next(targetgen))
        optimizer.apply_gradients(zip(grads, self.variables))
        t1 = time()
        print("\nMinibatch {} time {}h,{}m,{}s, loss {}".format(j+1, (t1-t0)//3600, ((t1-t0)%3600)//60, (t1-t0)%60, loss_main))
        minibatch_losses.append(loss_main)
        
      optimizer.apply_gradients(zip(grads, self.variables))
      t2 = time()
      
      val_loss = self.compute_loss(next(test_x_gen), next(test_y_gen))
      epoch_losses.append(minibatch_losses)
      epoch_losses.append((loss_main, val_loss))
      
      checkpoint = tfe.Saver(self.variables)
      ckpt_filename = "model.ckpt"
      checkpoint_num += 1
      checkpoint.save(file_prefix=os.path.join(checkpoint_directory, ckpt_filename),
                     global_step=checkpoint_num)
      print('\nEpoch Saved')
    return epoch_losses

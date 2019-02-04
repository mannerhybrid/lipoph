import nltk
import gensim
from gensim import corpora
import re
import numpy as np
from nltk.tokenize import WordPunctTokenizer,  PunktSentenceTokenizer
from Bio import Entrez
from bs4 import BeautifulSoup as soup
from time import time
from keras.utils import Progbar
import os
import tensorflow as tf
from datetime import datetime

tf.enable_eager_execution()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
maindict = corpora.Dictionary.load("..\\data\\maindict.dict") 

stopwords = [word.replace('\n', '') for word in open("..\\data\\stopwords.txt").readlines()]
def remove_stopwords(text):
    text = [w for w in text if w not in stopwords and not len(w) == 1]
    return text

def preprocess(text):
    sent_tokenizer = PunktSentenceTokenizer()
    sentences = [sentence.lower() for sentence in sent_tokenizer.tokenize(text)]
    sentences = [re.sub(r'\([\W\w\d\D%=.,]+\)', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'\.', ' __END__ ', sentence) for sentence in sentences]
    sentences = [re.sub(r'-', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\d\D]\.[\d\D]+', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r' \d\D ', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[;,:-@#%&\"\'�]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\(\)\[\]\+\*\/]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[��]', ' ', sentence) for sentence in sentences]
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    texts = [gensim.parsing.preprocessing.remove_stopwords(sent) for sent in sentences]
    texts = [word_tokenizer.tokenize(sent) for sent in sentences]
    texts = [[lemmatizer.lemmatize(word) for word in text] for text in texts]
    texts = [[t for text in texts for t in text]]
    return texts

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names
    return []

# def extractNounPhrases(sample):
#   text = str(sample).lower()

#   pattern = "NP: {(<JJ>* <NN.*>+)?}"
#   st = nltk.tokenize.PunktSentenceTokenizer()
#   wt = nltk.tokenize.WordPunctTokenizer()

#   sentences = st.tokenize(text)
#   sentences = [re.sub(r'\([\W\w\d\D%=.,]+\)', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'\.', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'-', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'[\d\D]\.[\d\D]+', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r' \d\D ', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'[;,:-@#%&\"\'�]', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'[\(\)\[\]\+\*\/]', ' ', sentence) for sentence in sentences]
#   sentences = [re.sub(r'[��î±]', ' ', sentence) for sentence in sentences]
#   sentences = [wt.tokenize(sent) for sent in sentences]
#   sentences = [nltk.pos_tag(sent) for sent in sentences]
#   cp = RegexpParser(pattern)

#   nphrases_list = [[' '.join(leaf[0] for leaf in tree.leaves()) 
#                         for tree in cp.parse(sent).subtrees() 
#                         if tree.label()=='NP'] for sent in sentences]

#   return [phrase for item in nphrases_list for phrase in item]

def negative_sampling(vk, y, maindict=maindict):
  vkns = []; yns =[]
  keys = np.array(maindict.keys())
  for kwords, label in list(zip(vk, y)):
    absent_keys = np.setdiff1d(keys, kwords)
    kwords_noise = np.random.choice(absent_keys, [1,2])
    label_noise = np.zeros_like(kwords_noise)
    kwords_new = np.concatenate((kwords,kwords_noise), axis=1)
    labels_new = np.concatenate((label, label_noise), axis=1)
    vkns.append(np.array(kwords_new)); yns.append(labels_new)
  return vkns, yns

def remove_nan(train_data):
  error=1
  i = 0
  outdata = []
  for record in train_data:
    if not record[2] == []:
      outdata.append(record)
  return outdata

def replicate_lines(train_data):
  _data = []
  for record in train_data:
    numkw = len(record[-1])
    for i in range(numkw):
        _data.append((record[0], record[1], np.expand_dims(record[2][0][i],0), np.expand_dims(record[3][0][i],0)))
  return _data

def batch_generator(data, minibatch):
  num_batches = 1 + len(data)//minibatch
  for i in range(num_batches):
    start = i*minibatch
    if not (i+1)*minibatch > len(data):
      end = (i+1)*minibatch
    else:
      end = len(data)
    yield data[start:end]

def miner(pmid):
    Entrez.email = "md.nur.hakim.rosli@gmail.com"
    errant_ids = []
    disabled = 0
    passed = 0
    fetch_handle = Entrez.efetch("pubmed", id=pmid, rettype="xml", retmode="abstract")
    fetch_record = soup(fetch_handle.read(), "xml")
    article = fetch_record.PubmedArticleSet.PubmedArticle.MedlineCitation
    meshes = article.findAll("MeshHeading")
    authors = article.Article.AuthorList.findAll("Author")
    try:
        title = article.Article.ArticleTitle.text
        abstract = " ".join([section.text for section in article.Article.Abstract.findAll("AbstractText")])
    except:
        disabled += 1
        errant_ids.append(id)
        title = ""
        abstract = ""
    return (title, abstract)
import csv
import numpy as np
from gensim.models import KeyedVectors
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import string
import re
import itertools
import pickle
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

tf.config.list_physical_devices('CPU')

# function for loading in GloVe embeddings as a dictonary
def glove2dict(glove_filename):
    with open(glove_filename, encoding='UTF-8-sig') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:]))) 
                for line in reader}
    return embed

glove_path = "C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/multilingual_embeddings.es.txt" # get it from https://nlp.stanford.edu/projects/glove
pre_glove = glove2dict(glove_path)

data = pd.read_csv("C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/transformed-data/combined_transformated_data.csv", encoding="UTF-8")

string.punctuation
# defining the function to remove punctuation and numbers
def remove_punctuation_and_nums(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    numberfree=re.sub(r'\d+', '', text)
    return numberfree

# storing the puntuation free text
data['clean_text']= data['formatted_text'].apply(lambda x:remove_punctuation_and_nums(x))
data['lower_text']= data['clean_text'].apply(lambda x: x.lower())

#defining function for tokenization
def tokenization(text):
    tokens = re.findall('\w+', text)
    return tokens

#applying function to the column
data['tokenized_text']= data['lower_text'].apply(lambda x: tokenization(x))

#Stop words in spanish present in the nltk library
stopwords_es = nltk.corpus.stopwords.words('spanish')

#Stop words in portuguese present in the nltl library
stopwords_pt = nltk.corpus.stopwords.words('portuguese')

#defining the function to remove stopwords from tokenized text
def remove_stopwords_es(text):
    output = [i for i in text if i not in stopwords_es]
    return output
#applying the function
data['no_stopwords'] = data['tokenized_text'] #.apply(lambda x:remove_stopwords_es(x))

#defining the function to remove stopwords from tokenized text
def remove_stopwords_pt(text):
    output = [i for i in text if i not in stopwords_pt]
    return output
#applying the function
data['no_stopwords'] = data['no_stopwords'] #.apply(lambda x:remove_stopwords_pt(x))

def find_oov_words(text):
    output = [i for i in text if i not in pre_glove.keys()]
    return list(set(output))

# applying the function
oov_list = list(itertools.chain.from_iterable(data['no_stopwords'].apply(find_oov_words)))

# build a dictionary of word counts
oov_dict = {}
for word in oov_list:
    if word not in oov_dict:
        oov_dict[word] = 1
    else:
        oov_dict[word] += 1

def get_rareoov(xdict, val):
    return [k for (k,v) in xdict.items() if v <= val]

# convert oov back to a list with duplicate words
unique_text = []
for word in oov_dict.keys():
    unique_text.append(word)

# cleaning oov to only unique values with more than 2 instances included in glove
oov = unique_text
# test oov cleaning
print(len(oov))
print(oov[:10])

# selecting words with only 1 instance in pro-Russian corpus
oov_rare = get_rareoov(oov_dict,0)
oov_rare = set(oov_rare)
print(len(oov_rare))
corp_vocab = list(set(oov) - oov_rare)
# corp_vocab = list(set(oov))
print(len(corp_vocab))

data_doc = ""
for sentence in data['no_stopwords']:
    if isinstance(sentence, str):
        sentence_cleaned = [word for word in sentence.split() if word not in oov_rare]
        data_doc += ' '.join(sentence_cleaned) + "\n"
    elif isinstance(sentence, list):
        sentence_cleaned = [word for sublist in sentence for word in sublist.split() if word not in oov_rare]
        data_doc += ' '.join(sentence_cleaned) + "\n"

# Write data_doc to a .txt file; this only needs to be done once
with open('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/output.txt', 'w', encoding='utf-8') as file:
   file.write(data_doc)

# Load data_doc from a .txt file
with open('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/output.txt', 'r', encoding='utf-8') as file:
    data_doc = file.read()

data_doc = [data_doc]

cv = CountVectorizer(ngram_range=(1,5), vocabulary=corp_vocab)
X = cv.fit_transform(data_doc)
Xc = (X.T * X)
Xc.setdiag(0)
coocc_ar = Xc.toarray()

print(coocc_ar.shape)

mittens_model = Mittens(n=300, max_iter=1000, learning_rate=0.01)

new_embeddings = mittens_model.fit(
    coocc_ar,
    vocab = corp_vocab,
    initial_embedding_dict = pre_glove)

newglove = dict(zip(corp_vocab, new_embeddings))

with open("test_glove.txt", "w", encoding="utf-8") as f:
    for word in newglove.keys():
        embedding = " ".join(str(x) for x in newglove[word])
        f.write(f"{word} {embedding}\n")

#filename = 'glove2word2vec_model.sav'
#pickle.dump(newglove, open(filename, 'wb'))

#pre_glove_model = KeyedVectors.load_word2vec_format(glove_path, "/Users/dwaste/Desktop/Undergrad-Thesis-Repo/glove.twitter.27B.200d.word2vec.txt")

#with open(filename, 'a+') as fp:
    #pickle.dump(pre_glove_model, fp)



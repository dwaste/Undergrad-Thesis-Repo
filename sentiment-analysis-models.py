# this script is developed on top of tensorflow tutorials and techniques I 
# have learned from Santa Cruz Aftifical Intelligence (SCAI) meetings

import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle 
import keras
import numpy as np
import re
import string

# function to plot graphs for model evaluation
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# load the data
file = open('text.txt',"r",encoding = "utf8") #This is a file of your choice! Go Online and Find a pdf or book you want to use! Make sure it is a .txt file

# put file into array
lines = []
for i in file:
  lines.append(i) 

# join the list into a string
data = ""
for i in lines:
  data = "".join(lines)

data = data.replace('\n','').replace('\r','').replace('\ufeff','')

data =  data.split('deliminter')
data =  data.array(data) #join data
data[:500] #See the First 500 chars in your text file

# preprocessing the data 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

sequence_data = tokenizer.texts_to_sequences([data])[0] #convert string into numeric representation
sequence_data[:15] #first fifteen sequences

# defining GloVe function for multilanguage text embedding 
def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)



  return word_to_vec_map


import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd

def load_from_txt_file(path):
    df = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
    return {key: val.values for key, val in df.T.items()}

pretrained_embeddings = load_from_txt_file('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/test_glove.txt')

pretrained_stanford_embeddings = load_from_txt_file('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/multilingual_embeddings.es.txt')

with open("repo_glove.txt", "w", encoding="utf-8") as f:
    for word in pretrained_embeddings.keys():
       embedding = " ".join(str(x) for x in pretrained_embeddings[word])
       f.write(f"{word} {embedding}\n")

with open("repo_glove.txt", "a", encoding="utf-8") as f:
    for word in pretrained_stanford_embeddings.keys():
        embedding = " ".join(str(x) for x in pretrained_stanford_embeddings[word])
        f.write(f"{word} {embedding}\n")

glove_file = datapath('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/repo_glove.txt')
word2vec_glove_file = get_tmpfile("C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/repo_glove.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

filename = 'repo_glove2word2vec_model.sav'
pickle.dump(model, open(filename, 'wb'))      
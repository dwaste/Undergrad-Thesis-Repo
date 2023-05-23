# -*- coding: utf-8 -*-
# https://www.kaggle.com/code/khoulaalkharusi/semi-supervised-document-classification
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.semi_supervised import SelfTrainingClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# from fasttext import FastVector
from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/transformed-data/combined_transformated_data.csv", encoding="UTF-8")

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
data['no_stopwords'] = data['tokenized_text'].apply(lambda x:remove_stopwords_es(x))

#defining the function to remove stopwords from tokenized text
def remove_stopwords_pt(text):
    output = [i for i in text if i not in stopwords_pt]
    return output
#applying the function
data['no_stopwords'] = data['no_stopwords'].apply(lambda x:remove_stopwords_pt(x))

data['no_stopwords'] = data['no_stopwords'].apply(lambda x: ' '.join(x))

classesList = data['Economía'].unique()
print(classesList)
#data['Economía'] = pd.factorize(data['Economía'])[0]


df_s = data.sample(frac = 1, ignore_index = True)
# df_s['no_stopwords'] = [[word.lower() for word in line.split()] for line in data['no_stopwords']]
X = df_s['no_stopwords']
y = df_s['Economía'].astype('int64')

X_train = []
y_train = []
X_test = []
y_test = []
classTest = [0]*len(classesList)
for i in range (len(y)):
    if classTest[y[i]] < 100:
        X_test.append(X[i])
        y_test.append(y[i])
        classTest[y[i]] += 1
        
    else:
        X_train.append(X[i])
        y_train.append(y[i])
#for i in range(2):
    #print(X_train[i], y_train[i])

# vectorizer = MultiLabelBinarizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 0), max_features=10000)
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

y_train_mask = []
classMask = [0]*len(classesList)
for i in range (len(y_train)):
    if classMask[y_train[i]] >= 0:
        y_train_mask.append(y_train[i])
        classMask[y_train[i]] += 1
    else:
        y_train_mask.append(-1)

#for i in range(2):
    #print(X_train[i], y_train[i])

df_train = pd.DataFrame(list(zip(X_train, y_train_mask)),columns =['no_stopwords', 'Economía'])

y_train_mask = np.array(y_train_mask)

########## Step 1 - Data Prep ########## 
# Select only records with known labels
X_baseline = X_train_vec[df_train['Economía']!=-1]
y_baseline = y_train_mask[df_train['Economía']!=-1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.BuPu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

##########  Model Fitting ########## 
# Specify SVC model parameters
model = SVC(kernel='rbf', 
            probability=True, 
            C=1.0, # default = 1.0
            gamma='scale', # default = 'scale'
            random_state=0
           )

# Fit the model
clf = model.fit(X_baseline, y_baseline)
##########  Model Evaluation ########## 
# Use score method to get accuracy of the model
print('---------- SVC Baseline Model - Evaluation on Test Data ----------')
accuracy_score_B = model.score(X_test_vec, y_test)
print('Accuracy Score: ', accuracy_score_B)
# Look at classification report to evaluate the model
print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))

########## Model Fitting ########## 
# Specify SVC model parameters
model_svc = SVC(kernel='rbf', 
                probability=True, # Need to enable to be able to use predict_proba
                C=1.0, 
                gamma='scale',
                random_state=0
               )

# Specify Self-Training model parameters
self_training_model = SelfTrainingClassifier(base_estimator=model_svc, 
                                             threshold=0.98,
                                             criterion='threshold',
                                             #k_best=5,
                                             max_iter=100,
                                             verbose=True 
                                            )

# Fit the model
clf_ST = self_training_model.fit(X_train_vec, y_train_mask)

########## Model Evaluation ########## 
print('')
print('---------- Self Training Model - Summary ----------')
print('Base Estimator: ', clf_ST.base_estimator_)
print('Classes: ', clf_ST.classes_)
print('Transduction Labels: ', clf_ST.transduction_)
print('Iteration When Sample Was Labeled: ', clf_ST.labeled_iter_)
print('Number of Features: ', clf_ST.n_features_in_)
# print('Feature Names: ', clf_ST.feature_names_in_)
print('Number of Iterations: ', clf_ST.n_iter_)
print('Termination Condition: ', clf_ST.termination_condition_)
print('')


print('---------- Self Training Model - Evaluation on Test Data ----------')
accuracy_score_ST = clf_ST.score(X_test_vec, y_test)
print('Accuracy Score: ', accuracy_score_ST)
# Look at classification report to evaluate the model
y_pred = clf_ST.predict(X_test_vec)
print(classification_report(y_test, y_pred, zero_division=0))

cnf_matrix = confusion_matrix(y_test, y_pred,labels=list(range(2)))
# Plot non-normalized confusion matrix
plt.figure(figsize = (2, 2))
plot_confusion_matrix(cnf_matrix, classes=classesList,
                      title='Self-training SSL Confusion matrix | Base model: SVM')

# Specify LR model parameters
model = LogisticRegression(C = 10, penalty = 'l2', solver = 'newton-cg')

# Fit the model
clf = model.fit(X_baseline, y_baseline)

print('---------- Logistic Regression Baseline Model - Evaluation on Test Data ----------')
accuracy_score_B = model.score(X_test_vec, y_test)
print('Accuracy Score: ', accuracy_score_B)
# Look at classification report to evaluate the model
print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))

########## Model Fitting ########## 
# Specify LR model parameters
model_lr =  LogisticRegression(C = 10, penalty = 'l2', solver = 'newton-cg')

# Specify Self-Training model parameters
self_training_model = SelfTrainingClassifier(base_estimator=model_lr, 
                                             threshold=0.98,
                                             criterion='threshold',
                                             #k_best=5,
                                             max_iter=100,
                                             verbose=True 
                                            )

# Fit the model
clf_ST_LR = self_training_model.fit(X_train_vec, y_train_mask)

print('')
print('---------- Self Training Model - Summary ----------')
print('Base Estimator: ', clf_ST_LR.base_estimator_)
print('Classes: ', clf_ST_LR.classes_)
#print('Transduction Labels: ', clf_ST_LR.transduction_)
#print('Iteration When Sample Was Labeled: ', clf_ST_LR.labeled_iter_)
print('Number of Features: ', clf_ST_LR.n_features_in_)
#print('Feature Names: ', clf_ST_LR.feature_names_in_)
print('Number of Iterations: ', clf_ST_LR.n_iter_)
print('Termination Condition: ', clf_ST_LR.termination_condition_)
print('')

print('---------- Self Training Model - Evaluation on Test Data ----------')
accuracy_score_ST_LR = clf_ST_LR.score(X_test_vec, y_test)
print('Accuracy Score: ', accuracy_score_ST_LR)
# Look at classification report to evaluate the model
y_pred_LR = clf_ST_LR.predict(X_test_vec)
print(classification_report(y_test, y_pred_LR, zero_division=0))

cnf_matrix = confusion_matrix(y_test, y_pred_LR,labels=list(range(2)))
# Plot non-normalized confusion matrix
plt.figure(figsize = (2, 2))
print(plot_confusion_matrix(cnf_matrix, classes=classesList,
                      title='Self-training SSL Confusion matrix | Base model: Logistic Regression'))

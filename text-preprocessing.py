import matplotlib.pyplot as plt
import pickle 
import numpy as np
import re
import string

# load the data
file = open('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russian-ground-narrative-text-files-combined.txt',"r",encoding = "utf8")

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
data = np.asarray(data)

def clean_text(text):
    # Remove punctuation points that are floating in between whitespace
    text = re.sub(r'\s+\.\s+', ' ', text)
    
    # Remove any numbers in the text
    text = re.sub(r'\d+', '', text)
    
    # Remove instances of double whitespace and \t
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\t', '', text)

    return text.strip()

cleaned_text = clean_text(data)
print(cleaned_text)

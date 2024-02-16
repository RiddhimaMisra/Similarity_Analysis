#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
import re
from tqdm import tqdm

import collections

from sklearn.cluster import KMeans

from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens

import pickle
import sys

from gensim.models import word2vec # For represent words in vectors
import gensim


# In[5]:


#Read given data-set using pandas
text_data = pd.read_csv("DataNeuron_Text_Similarity.csv")
print("Shape of text_data : ", text_data.shape)
text_data.head(5)


# In[6]:


# Check if text data have any null values
text_data.isnull().sum() 


# Preprocessing of text1 & text2
# Convert phrases like won't to will not using function decontracted() below
# Remove Stopwords.
# Remove any special symbols and lower case all words
# lemmatizing words using WordNetLemmatizer define in function word_tokenizer below

# In[7]:


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[8]:


# Combining all the above stundents 

preprocessed_text1 = []

# tqdm is for printing the status bar

for sentance in tqdm(text_data['text1'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text1.append(sent.lower().strip())


# In[9]:


# Merging preprocessed_text1 in text_data

text_data['text1'] = preprocessed_text1
text_data.head(3)


# In[10]:


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_text2 = []

# tqdm is for printing the status bar
for sentance in tqdm(text_data['text2'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
   
    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text2.append(sent.lower().strip())


# In[11]:


# Merging preprocessed_text2 in text_data

text_data['text2'] = preprocessed_text2

text_data.head(3)


# In[12]:


def word_tokenizer(text):
            #tokenizes and stems the text
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer() 
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            return tokens


# In[17]:


# Load pre_trained Google News Vectors after download file

wordmodelfile="GoogleNews-vectors-negative300.bin.gz"
wordmodel= gensim.models.KeyedVectors.load_word2vec_format(wordmodelfile, binary=True)


# In[33]:


# This code check if word in text1 & text2 present in our google news vectors vocabalry.
# if not it removes that word and if present it compares similarity score between text1 and text2 words


similarity = [] # List to store similarity scores


for ind in text_data.index:
    s1 = text_data['text1'][ind]
    s2 = text_data['text2'][ind]
    
    
    if s1 == s2:
        similarity.append(1.0) # 1 means highly similar
        
        
    else:
        s1words = word_tokenizer(s1)
        s2words = word_tokenizer(s2)
        
        
        vocab = wordmodel.key_to_index # The vocabulary considered in the word embeddings
        
        
        if len(s1words and s2words) == 0:
            similarity.append(0.0) # 0 means highly dissimilar
            
            
        else:
            for word in s1words.copy(): # Remove sentence words not found in the vocab
                if word not in vocab:
                    s1words.remove(word)
                    
                    
            
            for word in s2words.copy(): # Remove sentence words not found in the vocab
                if word not in vocab:
                    s2words.remove(word)
                    
                    
                            
            similarity.append(wordmodel.n_similarity(s1words, s2words))


# In[31]:


# Get Unique_ID and similarity

final_score = pd.DataFrame({'Unique_ID':text_data.index,
                     'Similarity_score':similarity})
final_score.head(3) 


# In[32]:


# SAVE DF as CSV file 

final_score.to_csv('Submission.csv',index=False)


# In[37]:


#pip install streamlit


# In[41]:


#pip install --upgrade protobuf


# In[3]:


import streamlit as st

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0

# Streamlit UI
st.title('Text Similarity App')

text1 = st.text_input('Enter text 1:')
text2 = st.text_input('Enter text 2:')

if st.button('Get Similarity Score'):
    similarity_score = jaccard_similarity(text1, text2)
    st.write('Similarity Score:', similarity_score)


# In[ ]:





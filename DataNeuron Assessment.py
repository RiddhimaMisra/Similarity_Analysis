import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import gensim
import streamlit as st

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load pre-trained Google News Vectors
wordmodelfile = "GoogleNews-vectors-negative300.bin"
wordmodel = gensim.models.KeyedVectors.load_word2vec_format(wordmodelfile, binary=True,limit=100000)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords_set)
    return text.lower().strip()

# Jaccard similarity function
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0

# Load stopwords and convert to set for faster lookup
stopwords_set = set(stopwords.words('english'))

# Streamlit UI
st.title('Text Similarity App')
text1 = st.text_input('Enter text 1:')
text2 = st.text_input('Enter text 2:')

if st.button('Get Similarity Score'):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)
    similarity_score = jaccard_similarity(preprocessed_text1, preprocessed_text2)
    st.write('Similarity Score:', similarity_score)

# Import necessary libraries
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Lowercase and remove stopwords
    text = text.lower()
    # Apply stemming
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text




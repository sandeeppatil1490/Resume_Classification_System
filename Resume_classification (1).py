import pickle
import streamlit as st
import pandas as pd
import numpy as np
import re
import docx2txt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

svm_model_final = pickle.load(open('dt.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('word_vectorizer.pkl', 'rb'))


st.title('Resume classification')

uploaded = st.file_uploader('Upload your Resume', type = ["pdf", "docx", "doc", "txt"])


if uploaded is not None:
    data = []
# Reading file from user      
    if file_name.endswith('.docx'):
        text = docx2txt.process(uploaded)
        data.append(text)
    else:
        with open(uploaded, 'rb') as f:
            text = str(f.read())
            data.append(uploaded)
            
    df = pd.DataFrame({'Resume': data})
    

#    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    
# Data Cleaning    
    def clean_resume_text(resume_text):
        """
        This function takes in a string of text (resume) as input and returns a cleaned version of the text.
        """
        # Convert to lowercase
        resume_text = resume_text.lower()
    
        # Remove numbers and special characters
        resume_text = re.sub('[^a-zA-Z]', ' ', resume_text)
    
        # Remove punctuation
        resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))
    
        # Remove extra whitespaces
        resume_text = ' '.join(resume_text.split())
    
        # Remove words with two or one letter
        resume_text = ' '.join(word for word in resume_text.split() if len(word) > 2)

        # Remove stop words
        resume_text = ' '.join(word for word in resume_text.split() if word not in stopwords)

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        resume_text = ' '.join(lemmatizer.lemmatize(word) for word in resume_text.split())

        return resume_text

    df["clean_text"] = df["Resume"].apply(clean_resume_text)
 
    
    
    # Clean the text by removing short words and noise words
    noise_words = ['xff','xffcj', 'xbabp','xddn','xaek','xcdf','xedv','xfe', 'xfeoj', 'xbe', 'xed', 'xbf', 'xef',
                   "xcf","xfe",'xfd', 'xea', 'xdd', 'xde', 'xba', 'xdc', 'xae', 'xdf', 'xec', 'xeb', 'xbb', 'xca',
                   'xaf', 'xac', 'xaa', 'xcf', 'xda', 'xcd', 'xab', 'xfb', 'xce', 'xbd', 'xdb', 'xcc', 
                   'xbc', 'xfc', 'xfa', 'xee', 'xad', 'xcb','hxai','xban']

    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\b\w{{1,2}}\b|\b(?:{})\b'.format('|'.join(noise_words)), '', x))
    
    resume = df.loc[:, 'clean_text']
    
    X_train_tfidf = word_vectorizer.transform(resume)
    
    y = dt.predict(X_train_tfidf)
    
    if y == 0:
        st.subheader("Person's Resume Match's to SQL Developer")
    elif y == 1:
        st.subheader("Person's Resume Match's to People Soft")
    elif y == 2:
        st.subheader("Person's Resume Match's to React Developer")
    else:
        st.subheader("Person's Resume Match's to Workday")

        
        
       
   
    
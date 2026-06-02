from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os

import warnings
warnings.filterwarnings('ignore')

word_index = imdb.get_word_index()

model_path = os.path.join(os.path.dirname(__file__), "simple_rnn_imdb.h5")
model = load_model(model_path)

def preprocessing(text):
    text = text.lower().split()
    review_text = [word_index.get(word,2) + 3 for word in text]
    pad_text = sequence.pad_sequences([review_text],maxlen=500)
    return pad_text

def prediction(text):
    preproced_text = preprocessing(text)
    predicted_val = model.predict(preproced_text,verbose=0)
    statment = 'positive' if predicted_val[0][0] > 0.5 else 'negative'
    return predicted_val[0][0],statment


import streamlit as st
user_input = st.text_area("movie review")

if st.button('classify'):
    if user_input.strip():
        score,statment = prediction(user_input)
        st.write(f'score: {score}')
        st.write(f'sentiment: {statment}')

    else:
        st.write('Please enter a movie review')

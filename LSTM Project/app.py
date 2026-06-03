import pandas as pd
import numpy  as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('LSTM_model.h5',compile=False)

with open('tokenized.pickle','rb') as file:
    tokenizer = pickle.load(file)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Predixtion Using LSTM RNN")
input_text = st.text_input("Entre The Text")
if st.button("Predict The Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word: {next_word}")

else:
    st.write("Enter Some Message.")
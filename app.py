import pickle
from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf
import pandas as pd


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl','rb') as file:
    one_hot_encoder_geography = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


model = load_model('model.h5')

st.title('custmer chirn prediction')

geography = st.selectbox('Geograpgy', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age", 18,93)
balance = st.number_input('Balance')
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0,10)
num_of_products = st.slider("Number of Products", 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
geo_encoded = one_hot_encoder_geography.transform(pd.DataFrame({'Geography':[geography]})).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]


st.write(f'Predicted probability of churn: {pred_prob:.2f}')

if prediction[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')


import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

# Define the components of the path
base_dir = 'artifacts'
sub_dir = 'models'

with open(os.path.join(base_dir, sub_dir, 'LabelEncoder.pkl'), 'rb') as f:
  LE = pickle.load(f)
  f.close()

with open(os.path.join(base_dir, sub_dir, 'ColumnTransformer.pkl'), 'rb') as f:  
  ct = pickle.load(f)
  f.close()

with open(os.path.join(base_dir, sub_dir, 'StandardScaler.pkl'), 'rb') as f:  
  sc = pickle.load(f)
  f.close()
model = load_model(os.path.join(base_dir, sub_dir, 'model.h5'))

import streamlit as st
st.write("""
# End to End project of ANN
""")
st.title("Predict the churn of employee")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")

# st.number_input('CreditScore', label_visibility="visible")
with st.form("my_form"):
  CreditScore = st.number_input(
      "Insert a Credit Score", min_value=0, max_value=900, placeholder="Type a number..."
  )
  Geography=st.selectbox('Geography', ['France','Germany','Spain'], index=0, placeholder="Choose an option", label_visibility="visible")
  Gender=st.selectbox('Gender', ['Female', 'Male'], index=0, placeholder="Choose an option", label_visibility="visible")
  Age=st.slider('Age', min_value=0, max_value=100, value=0, label_visibility="visible")
  Tenure=st.slider('Tenure', min_value=0, max_value=100, value=0, label_visibility="visible")
  Balance = st.number_input('Balance',  label_visibility="visible")

  NumOfProducts = st.number_input(
      "Insert a Number of products", min_value=0, max_value=4, placeholder="Type a number..."
  )
  my_dict = {False:0, True:1}
  HasCrCard = st.toggle('Has Credit Card', value=False, label_visibility="visible")
  HasCrCard = my_dict[HasCrCard]
  IsActiveMember = st.toggle('Is Active Member', value=False, label_visibility="visible")
  IsActiveMember = my_dict[IsActiveMember]
  EstimatedSalary = st.number_input(
      "Insert a Estimated Salary", placeholder="Type a number..."
  )
  # Every form must have a submit button.
  submitted = st.form_submit_button("Submit")
  if submitted:
    input_data = {
    'CreditScore':CreditScore,
    'Geography': Geography,
    'Gender' : Gender,
    'Age':Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts' : NumOfProducts,
    'HasCrCard':HasCrCard,
    'IsActiveMember':IsActiveMember,
    'EstimatedSalary':EstimatedSalary
    }
    input_array = pd.DataFrame([input_data])
    input_array['Gender'] = LE.transform(input_array['Gender'])
    input_ct = ct.transform(input_array)
    scaled_input = sc.transform(input_ct)
    model.predict(scaled_input)
    chrun = round(model.predict(scaled_input)[0][0])
    if chrun == 0:
      st.write("Customer will retain")
    else:
      st.write("Customer will leave")

    # print(CreditScore, Geography,Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
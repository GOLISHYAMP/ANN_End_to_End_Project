import pickle
from tensorflow.keras.models import load_model


with open('artifacts\models\LabelEncoder.pkl', 'rb') as f:
  LE = pickle.load(f)
  f.close()

with open('artifacts\models\ColumnTransformer.pkl', 'rb') as f:
  ct = pickle.load(f)
  f.close()

with open('artifacts\models\StandardScaler.pkl', 'rb') as f:
  sc = pickle.load(f)
  f.close()
model = load_model('artifacts\models\model.h5')

import streamlit as st

st.title("Predict the churn of employee")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")

st.number_input('CreditScore', min_value=0, max_value=900, value="max", label_visibility="visible")
st.selectbox('Geography', ['France','Germany','Spain'], index=0, placeholder="Choose an option", label_visibility="visible")
st.selectbox('Gender', ['female', 'male'], index=0, placeholder="Choose an option", label_visibility="visible")
st.slider('Age', min_value=0, max_value=100, value=0, label_visibility="visible")
st.slider('Tenure', min_value=0, max_value=100, value=0, label_visibility="visible")
st.number_input('Balance', min_value=None, max_value=None, value="min",  label_visibility="visible")




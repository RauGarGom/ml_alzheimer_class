import sys
import os
import streamlit as st
import pickle
import numpy as np
sys.path.append(os.path.abspath(r'C:\Users\raulg\Documents\THEBRIDGE_DS\0.-Repo_Git\ml_alzheimer_class\src'))
import utils as ut ### TODO: Intentar que sea relative path

### Import of model
with open('../models/xgb_baseline.pkl', 'rb') as f:
    xgb_baseline = pickle.load(f)

### Streamlit
# Title of the app
st.title('Alzheimer form')
st.session_state['result'] = ''

# Text input
mmse = st.text_input("Patient MMSE scoring","10")
funct_assess = st.selectbox(
    "What's the patient's Functional Assessment Scoring?:",
    (0,1,2,3,4,5,6,7,8,9,10))
memory = st.selectbox(
    "Does the patient present memory inefficiencies?",
    ("Yes", "No"))
behav = st.selectbox(
    "Does the patient present behavioral issues?",
    ("Yes", "No"))
adl = st.selectbox(
    "Being 0 non-affected and 10 extremely affected, how does the patient sees their daily life affected?",
    (0,1,2,3,4,5,6,7,8,9,10))

def results():
    st.session_state['result'] = ut.model_prediction(mmse,funct_assess,memory,behav,adl)

if st.button('Run prediction'):
    resultado = ut.model_prediction(mmse, funct_assess, memory, behav, adl)
    st.write(resultado)


# pred = model.predict(user_input)
# Display the user input
# st.write(f'Hello potatoes, {st.session_state['result']}!')
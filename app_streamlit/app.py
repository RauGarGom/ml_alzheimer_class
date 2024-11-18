import sys
import os
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
# import numpy as np
#  sys.path.append(os.path.abspath(r'C:\Users\raulg\Documents\THEBRIDGE_DS\0.-Repo_Git\ml_alzheimer_class\src'))
sys.path.append(os.path.relpath('../src'))
from utils import model_prediction, img_model_prediction  ### TODO: Intentar que sea relative path

### Import of model
# with open('../models/class/xgb_baseline.pkl', 'rb') as f:
#     xgb_baseline = pickle.load(f)

### Streamlit
# Title of the app
st.title('Alzheimer form')
st.session_state['result'] = ''
resultado = [0,0,0,0]
mapping_class = {
        0: 'No Alzheimer',
        1: 'Alzheimer',
    }
mapping_img = {
        0: 'Non Demented',
        1: 'Very Mild Demented',
        2: 'Mild Demented',
        3: 'Moderate Demented'
    }
with st.sidebar:
    st.title("Charts of the predictions")

### CLASS - Inputs
funct_assess = st.slider("What's the patient's Functional Assessment Scoring?:",0,30)

col1,col2 = st.columns(2,gap='medium')
with col1:
    memory = st.pills(
        "Does the patient present memory inefficiencies?",
        ("Yes", "No"),key=0,default='No')
    mmse = st.slider("What's the patient's MMSE scoring",0,10)


with col2:
    behav = st.pills(
        "Does the patient present behavioral issues?",
        ("Yes", "No"),key=1,default='No')
    adl = st.slider(
    "How does the patient perceives their daily life?",0,10)


### CLASS : Model and charts
def results():
    st.session_state['result'] = model_prediction(mmse,funct_assess,memory,behav,adl)

if st.button('Run prediction'):
    resultado = model_prediction(mmse, funct_assess, memory, behav, adl)
    st.write(resultado)
    if resultado[3] == 1:
        st.error(resultado[1])
    elif resultado[3]==0:
        st.success(resultado[1])
    else:
        st.warning(resultado[1])
    with st.sidebar:    
        st.plotly_chart(px.pie(values=resultado[2].flatten(),names=mapping_class.values(), title='Probabilities of class prediction')) #,names=mapping.keys()


# if resultado[3] > 0:
uploaded_img = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if st.button('Run image prediction'):
    img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    result = img_model_prediction(img_bytes)
    st.write(f'The model predicts the brain in the image is {mapping_img[result[0]]}, with a certainty of {result[1].max()*100}%')
    with st.sidebar:
        st.plotly_chart(px.pie(values=result[1].flatten(),names=mapping_img.values(), title='Probabilities of the prediction')) #,names=mapping.keys()

# pred = model.predict(user_input)
# Display the user input
# st.write(f'Hello potatoes, {st.session_state['result']}!')
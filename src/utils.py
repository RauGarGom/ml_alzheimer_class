import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn import svm
from sklearn.metrics import auc, roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import pickle
from tensorflow.keras import layers, models
from matplotlib.image import imread
import cv2

### Loading of models
with open('../models/class/xgb_baseline.pkl', 'rb') as f:
    class_model = pickle.load(f)
img_model = models.load_model("../models/image/baseline_model.keras")

def feat_eng(df):
    df['BadSleep'] = np.where(df['SleepQuality']<5,1,0)
    df['Overweight'] = np.where(df['BMI']>25,1,0)
    df['mix'] = df['FunctionalAssessment'] * df['MMSE'] * df['ADL']
    df['mix2'] = df['MemoryComplaints'] + df['BehavioralProblems']
    return df
def train_test(df,test_size=.3):
    x1 = df.drop(columns=['Diagnosis'])
    y1 = df['Diagnosis']
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size=test_size,stratify=y1)
    return x1_train, x1_test, y1_train, y1_test



def model_prediction(mmse="1",funct_asses=1,memory="Yes",behav="Yes",adl=1):
    ### A first part cleans the input given by Streamlit
    ### TODO: limpiar modelo, hay cosas que no hacen mucho
    mmse = int(mmse)
    memory = np.where(memory=='Yes',1,0)
    behav = np.where(behav=='Yes',1,0)

    ### Prediction. Values which are hardcoded are mix
    result = class_model.predict(np.array([[mmse,funct_asses,memory,behav,adl,382]]))
    text_result = str(np.where(result == 1, "Patient presents signs of alzheimer","Patient shows no signs of alzehimer")[0])
    return text_result

def img_model_prediction(image_path):
    image = cv2.imdecode(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (32, 32)) ### 32x32 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ### Conversion to gray scale
    image = image.reshape(-1, 32, 32, 1)
    img_pred = img_model.predict(image)
    return img_pred.argmax()

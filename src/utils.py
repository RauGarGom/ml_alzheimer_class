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

with open('../models/xgb_baseline.pkl', 'rb') as f:
    model = pickle.load(f)

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
    result = model.predict(np.array([[mmse,funct_asses,memory,behav,adl,382]]))
    text_result = str(np.where(result == 1, "Patient presents signs of alzheimer","Patient shows no signs of alzehimer")[0])
    return text_result


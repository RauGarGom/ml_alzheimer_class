# import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn import svm
from sklearn.metrics import auc, roc_auc_score, accuracy_score, confusion_matrix
# from xgboost import XGBClassifier
import pickle
from tensorflow.keras import models
# from matplotlib.image import imread
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

def img_images_load(x1_train_path='../data/images/processed_train/x1.pkl',y1_train_path='../data/images/processed_train/y1.pkl',
                    x1_test_path='../data/images/processed_val/x2.pkl',y1_test_path='../data/images/processed_val/y2.pkl',
                    val_set = False, reshuffle = True,img_size=32):
    ''' Loads the resized and grayscaled versions of the arrays of the images, and transforms the arrays to 
    the convenience of the predictor model. Makes vectorization of the target, and scales the x values'''
    x1_train = pickle.load(open(x1_train_path,'rb'))
    y1_train = pickle.load(open(y1_train_path,'rb'))
    x1_test = pickle.load(open(x1_test_path,'rb'))
    y1_test = pickle.load(open(y1_test_path,'rb'))
    mapping = {
        'NonDemented': 0,
        'VeryMildDemented': 1,
        'MildDemented': 2,
        'ModerateDemented': 3
    }
    y1_train = np.vectorize(mapping.get)(y1_train)
    y1_test = np.vectorize(mapping.get)(y1_test)
    x1_train = x1_train.reshape(-1, 1)
    x1_test = x1_test.reshape(-1, 1)
    ### Reshape needed for scalling
    scal = StandardScaler()
    x1_train = scal.fit_transform(x1_train)
    x1_test = scal.transform(x1_test)
    ### And we reshape to the final shape needed for the model
    if img_size == 32:
        x1_train = x1_train.reshape(-1, 32, 32, 1)
        x1_test = x1_test.reshape(-1, 32, 32, 1)
    else:
        x1_train = x1_train.reshape(-1, 64, 64, 1)
        x1_test = x1_test.reshape(-1, 64, 64, 1)       
    ### If val_set == True, it makes a subset for manual validation
    if reshuffle == False:
        if val_set == False:
            return x1_train,x1_test,y1_train,y1_test
        else:
            x1_train,x1_val,y1_train,y1_val = train_test_split(x1_train,y1_train,test_size=.2,stratify=y1_train,random_state=42)
            return x1_train,x1_test,x1_val,y1_train,y1_test,y1_val
    else:
        x1 = np.concatenate((x1_train,x1_test))
        y1 = np.concatenate((y1_train,y1_test))
        x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=.15,stratify=y1,random_state=42)
        print('x1_train shape', x1_train.shape)
        print('x1_test shape', x1_test.shape)
        print('y1_train shape', y1_train.shape)
        print('y1_test shape', y1_test.shape)
        unique_train, counts_train = np.unique(y1_train, return_counts=True)
        unique_test, counts_test = np.unique(y1_test, return_counts=True)
        print('y1_train distribution: \n',np.asarray((unique_train, counts_train)).T)
        print('y1_test distribution: \n',np.asarray((unique_test, counts_test)).T)
        if val_set == False:
            return x1_train,x1_test,y1_train,y1_test
        else:
            x1_train,x1_val,y1_train,y1_val = train_test_split(x1_train,y1_train,test_size=.2,stratify=y1_train,random_state=42)
            print('x1_val shape', x1_val.shape)
            print('y1_val shape', y1_val.shape)
            unique_val, counts_val = np.unique(y1_val, return_counts=True)
            print('y1_val distribution: \n',np.asarray((unique_val, counts_val)).T)
            return x1_train,x1_test,x1_val,y1_train,y1_test,y1_val


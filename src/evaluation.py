import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn import svm
from sklearn.metrics import auc, roc_auc_score, accuracy_score, confusion_matrix, recall_score
# from xgboost import XGBClassifier
import pickle
from tensorflow.keras import models
# from matplotlib.image import imread
import cv2
import utils as ut

def class_eval():
    alz = pd.read_csv('../data/class/raw/alzheimers_disease_data.csv',index_col='PatientID')
    alz_2b = alz[['MMSE','FunctionalAssessment','MemoryComplaints','BehavioralProblems','ADL','Diagnosis']]
    x1_train, x1_test, y1_train, y1_test = ut.train_test(alz_2b,test_size=.2)
    model_2= pickle.load(open("../models/class/model_2.pkl",'rb'))
    y1_pred = model_2.predict(x1_test)
    print("Accuracy", accuracy_score(y1_test,y1_pred))
    print("Recall", recall_score(y1_test,y1_pred))
    print(confusion_matrix(y1_test,y1_pred))

def img_eval():
    print('Loading Test dataset...')
    print('='*50)
    x1_train,x1_test,x1_val,y1_train,y1_test,y1_val,scal = ut.img_images_load('../data/images/processed_train/x1_64.pkl','../data/images/processed_train/y1_64.pkl',
                                                                     '../data/images/processed_val/x2_64.pkl','../data/images/processed_val/y2_64.pkl',
                                                                     val_set = True,reshuffle=True,img_size=64)
    print('='*50)
    print('Evaluating model and printing metrics...')
    model4c = models.load_model("../models/image/model_4.keras")
    y1_pred = model4c.predict(x1_test).argmax(axis=1)
    print("Accuracy", accuracy_score(y1_test,y1_pred))
    print("Recall", recall_score(y1_test,y1_pred,average='weighted'))
    print(confusion_matrix(y1_test,y1_pred))
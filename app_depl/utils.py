# import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import auc, roc_auc_score, accuracy_score, confusion_matrix
# from xgboost import XGBClassifier
import pickle
from tensorflow.keras import models
# from matplotlib.image import imread
import cv2

### Loading of models
with open('./model_2.pkl', 'rb') as f:
    class_model = pickle.load(f)
img_model = models.load_model("./model_5.keras")
img_scal = pickle.load(open("./aux_scal.pkl",'rb'))


def feat_eng(df):
    df['BadSleep'] = np.where(df['SleepQuality']<5,1,0)
    df['Overweight'] = np.where(df['BMI']>25,1,0)
    df['mix'] = df['FunctionalAssessment'] * df['MMSE'] * df['ADL']
    df['mix2'] = df['MemoryComplaints'] + df['BehavioralProblems']
    df['Elder'] = np.where(df['Age']<df['Age'].mean(),1,0)
    df['medical_conditions']= -df['FamilyHistoryAlzheimers'] + df['CardiovascularDisease']-df['Diabetes']-df['Depression']-df['HeadInjury']+df['Hypertension']
    df['demographics'] = df['Elder'] + df['Gender'] + df['Ethnicity'] + df['EducationLevel']
    df['clinical'] = -df['SystolicBP'] + df['DiastolicBP'] + df['CholesterolTotal'] - df['CholesterolLDL'] -df['CholesterolLDL'] +df['CholesterolHDL'] +df['CholesterolTriglycerides']
    df['symptoms'] = -df['Confusion'] - df['Disorientation'] - df['PersonalityChanges'] + df['DifficultyCompletingTasks']
    return df
def train_test(df,test_size=.3,val_size=0.2,val_set=False):
    x1 = df.drop(columns=['Diagnosis'])
    y1 = df['Diagnosis']
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size=test_size,stratify=y1,random_state=42)
    if val_set==False:
        return x1_train, x1_test, y1_train, y1_test
    else:
        x1_train,x1_val,y1_train,y1_val = train_test_split(x1_train,y1_train, test_size=val_size,stratify=y1_train,random_state=42)
        return x1_train, x1_test,x1_val, y1_train, y1_test,y1_val


def model_prediction(mmse="1",funct_asses=1,memory="Yes",behav="Yes",adl=1):
    ### A first part cleans the input given by Streamlit
    ### TODO: limpiar modelo, hay cosas que no hacen mucho
    mmse = int(mmse)
    memory = np.where(memory=='Yes',1,0)
    behav = np.where(behav=='Yes',1,0)

    result = class_model.predict(np.array([[mmse,funct_asses,memory,behav,adl]]))
    text_result = str(np.where(result == 1, "Patient presents signs of alzheimer. Please confirm with MRI prediction","Patient shows no signs of alzehimer")[0])
    result_proba = class_model.predict_proba(np.array([[mmse,funct_asses,memory,behav,adl]]))
    result_stream = result
    if (result_proba.max() < 0.9) and (result==0):
        text_result = str('Patient shows no signs of alzheimer, but the model is unsure. MRI model prediction is advised')
        result_stream = 2
    if  (result_proba.max() < 0.9) and (result==1):
        text_result = str('Patient shows signs of alzheimer, but the model is unsure. MRI model prediction is anyway advised')
        result_stream = 3
    return result, text_result, result_proba, result_stream

def img_model_prediction(image_path,img_size=32):
    '''Img_size must be the same as the one used by the training of the model.
    Model 4 (used in the demo) is made with 64x64'''
    image = cv2.imdecode(image_path, cv2.IMREAD_COLOR)
    image = recortar_centro_relativo(1,0.5)
    # mapping = {
    #     0: 'Non Demented',
    #     1: 'Very Mild Demented',
    #     2: 'Mild Demented',
    #     3: 'Moderate Demented'
    # }
    if img_size == 32:
        image = cv2.resize(image, (32, 32)) ### 32x32 pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ### Conversion to gray scale
        image = image.reshape(-1,1)
        image = img_scal.transform(image)
        image = image.reshape(-1, 32, 32, 1)
    else:
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ### Conversion to gray scale
        image = image.reshape(-1,1)
        image = img_scal.transform(image)      
        image = image.reshape(-1, 64, 64, 1)        
    img_pred = img_model.predict(image)
    return img_pred.argmax(),img_pred.round(4)

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
            return x1_train,x1_test,y1_train,y1_test,scal
        else:
            x1_train,x1_val,y1_train,y1_val = train_test_split(x1_train,y1_train,test_size=.2,stratify=y1_train,random_state=42)
            print('x1_val shape', x1_val.shape)
            print('y1_val shape', y1_val.shape)
            unique_val, counts_val = np.unique(y1_val, return_counts=True)
            print('y1_val distribution: \n',np.asarray((unique_val, counts_val)).T)
            return x1_train,x1_test,x1_val,y1_train,y1_test,y1_val,scal

def recortar_centro_relativo(imagen, porcentaje_ancho=1, porcentaje_alto=0.5):
    alto, ancho, _ = imagen.shape  # Dimensiones de la imagen
    
    # Calcular dimensiones del recorte
    ancho_corte = int(ancho * porcentaje_ancho)
    alto_corte = int(alto * porcentaje_alto)
    
    # Coordenadas centrales
    centro_x, centro_y = ancho // 2, alto // 2
    
    # Coordenadas del recorte
    x_inicio = max(centro_x - ancho_corte // 2, 0)
    x_fin = min(centro_x + ancho_corte // 2, ancho)
    y_inicio = max(centro_y - alto_corte // 2, 0)
    y_fin = min(centro_y + alto_corte // 2, alto)
    
    # Recortar la imagen
    recorte = imagen[y_inicio:y_fin, x_inicio:x_fin]
    return recorte
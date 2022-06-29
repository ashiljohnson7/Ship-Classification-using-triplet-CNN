from tensorflow import keras
import numpy as np
import cv2
from mean import mean1

import os
import pandas as pd
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
#from process import lbp2
from sklearn.model_selection import train_test_split

def compute_dist(a,b):
    return np.sum(np.square(a-b))

def load():
    import pickle
    

    f1=open('pro2.pkl','rb')
    df=pickle.load(f1)
    f1.close()
    # print(df.labels.value_counts())
    train, test = train_test_split(df, test_size=0.2)
    x_train=train['images']
    x_train=np.vstack(x_train)
    print(x_train.shape)
    x_train=x_train.reshape(-1,28,28,1)
    y_train=train['labels']
    y_train=np.array(y_train)
    print(y_train.shape)
    print(x_train.shape)
    x_test=test['images']
    x_test=np.vstack(x_test)
    x_test=x_test.reshape(-1,28,28,1)
    print(x_test.shape)
    y_test=test['labels']
    y_test=np.array(y_test)
    #print(x_train[0])

    return (x_train,y_train),(x_test,y_test)


def predict_type(img):
        global loaded_model,m0,m1
        
        
        
        m0=mean1(0)
        m1=mean1(1)
        m2=mean1(2)
        m3=mean1(3)
        
        li=[]

        preds = loaded_model.predict(img)[0]
        d0=compute_dist(m0,preds)
        print(d0)
        li.append(d0)
        d1=compute_dist(m1,preds)
        print(d1)
        li.append(d1)
        d2=compute_dist(m2,preds)
        print(d2)
        li.append(d2)
        d3=compute_dist(m3,preds)
        print(d3)
        li.append(d3)
        
        return np.argmin(li)

from sklearn.metrics import accuracy_score
def test():
    global loaded_model,m1,m0
    m0=mean1(0)
    m1=mean1(1)
    loaded_model = keras.models.load_model('model')
    (x_train,y_train),(x_test,y_test)=load()
    pred=[]
    l1=x_test.shape[0]
    print(x_train[0].shape)
    
    # t=np.array([x_train[0]]).reshape(1,28,28,1)
    # print(t.shape)
    for i in range(50):
        print(i,'****************************************')
        img=x_test[i]
        img=np.array(img).reshape(1,28,28,1)
        t=predict_type(img)
        pred.append(t)
    pred=np.array(pred)  
    print(y_test)
    print(pred)
    ac=accuracy_score(pred,y_test[:50])
    print(ac)
     


if __name__=='__main__':
    test()    


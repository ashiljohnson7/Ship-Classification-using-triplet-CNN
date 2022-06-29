from calibrate import lbp2
import pickle
from tensorflow import keras
import numpy as np

def mean1(n):
    model=keras.models.load_model('model')
    if n==0:
        f1=open('df0.pkl','rb')
        df=pickle.load(f1)
        f1.close()
        preds=model.predict(df)
        m=np.mean(preds,axis=0)
        print(m)
        return m
    if n==1:
        f1=open('df1.pkl','rb')
        df=pickle.load(f1)
        f1.close()
        preds=model.predict(df)
        m=np.mean(preds,axis=0)
        print(m)
        return m
    if n==2:
        f1=open('df2.pkl','rb')
        df=pickle.load(f1)
        f1.close()
        preds=model.predict(df)
        m=np.mean(preds,axis=0)
        print(m)
        return m 

    if n==3:
        f1=open('df3.pkl','rb')
        df=pickle.load(f1)
        f1.close()
        preds=model.predict(df)
        m=np.mean(preds,axis=0)
        print(m)
        return m        



if __name__ == '__main__':
    
   mean1(1)
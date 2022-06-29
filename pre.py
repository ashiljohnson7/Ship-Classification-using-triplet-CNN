import os
import pandas as pd
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential, Model #model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from model import main1
from calibrate import lbp2


def read_from_folder(path):
    print("==================================================")
    print("Scanning Folders....")
    images=[]
    labels=[]
    subfolders= [f for f in os.scandir(path) if f.is_dir()]
    # c=0
    for sf1 in subfolders:
        sf=[f for f in os.scandir(sf1.path) if f.is_dir()]
        for sf2 in sf: 

            imgs1=[f for f in os.scandir(sf1.path+'/Patch_Uint8')]
            for img in imgs1:
                try:
                    path1=path+'/'+sf1.name+'/Patch_Uint8'+'/'+img.name
                    label=img.name.split('_')[1]
                    if label=='Cargo':
                        c=0
                    elif label=='Dredging or underwater ops':
                        c=1
                    elif label=='Passenger' :
                        c=2
                    elif label=='Tanker':
                        c=3
                    if c in [0,1,2,3] :       
                        img1=load_process_image(path1)
                        images.append(img1)
                        labels.append(c)
                except Exception as e:
                    print(e)
                    #print(path1)
                    continue    
        # c=c+1              
    print('found ',len(images),' images belonging to 4 classes')
    print("converting to dataframe...")
    image_dict={'images':images,'labels':labels}
    image_df=pd.DataFrame(image_dict) #use to covert into table format
    image_df= image_df.sample(frac=1).reset_index(drop=True)   #data shffle table reurn cheyum
    print("dataframe ready") 
    return image_df

def load_process_image(path):
    #img=image.load_img(path,target_size=(48,48))
    l=lbp2(path)

    # norm = np.linalg.norm(img)
    # img = img/norm
    # img/=255.0
    return l
      


from sklearn.model_selection import train_test_split

def load():
    import pickle
    # df=read_from_folder('data')

    # f1=open('pro2.pkl','wb')
    # pickle.dump(df,f1)
    # f1.close()

    f1=open('pro2.pkl','rb') #read
    df=pickle.load(f1)
    f1.close()
    print(df.labels.value_counts())

    df0=df[df.labels==0]
    df1=df[df.labels==1]
    df2=df[df.labels==2]
    df3=df[df.labels==3]

    x0=df0['images']
    x0=np.vstack(x0)
    x0=x0.reshape(-1,28,28,1)
    f1=open('df0.pkl','wb') #write
    pickle.dump(x0,f1) #save
    f1.close()

    x1=df1['images']
    x1=np.vstack(x1)
    x1=x1.reshape(-1,28,28,1)
    f1=open('df1.pkl','wb')
    pickle.dump(x1,f1)
    f1.close()


    x2=df2['images']
    x2=np.vstack(x2)
    x2=x2.reshape(-1,28,28,1)
    f1=open('df2.pkl','wb')
    pickle.dump(x2,f1)
    f1.close()


    x3=df3['images']
    x3=np.vstack(x3)
    x3=x3.reshape(-1,28,28,1)
    f1=open('df3.pkl','wb')
    pickle.dump(x3,f1)
    f1.close()

   
    

    
    

    train, test = train_test_split(df, test_size=0.2)
    x_train=train['images']
    x_train=np.vstack(x_train)  #single array
    x_train=x_train.reshape(-1,28,28)
    y_train=train['labels']
    y_train=np.array(y_train)
    print(y_train.shape)
    print(x_train.shape)
    x_test=test['images']
    x_test=np.vstack(x_test)
    x_test=x_test.reshape(-1,28,28)
    print(x_test.shape)
    y_test=test['labels']
    y_test=np.array(y_test)
    #print(x_train[0])

    return (x_train,y_train),(x_test,y_test)

if __name__=="__main__":

    load()    


   
    







#n=10/0

# # df['images'] = df['images'].apply(lambda im: np.array(im))

# x_train = np.reshape(x_train,(len(df['images']),128,128, 1))
# x_test = np.reshape(x_test,(len(test['images']),128,128, 1))
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# model=main1((128,128,1))
# model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

# checkpointer = ModelCheckpoint(filepath='spoof_model1.h5', verbose=1, save_best_only=True ,monitor='val_accuracy')

# model.fit(x_train, y_train, epochs=20,
#         shuffle=True,
#         batch_size=32, validation_data=(x_test,y_test),
#         callbacks=[checkpointer], verbose=1)

# model_json = model.to_json()
# with open("spoof_model1.json", "w") as json_file:
#     json_file.write(model_json)    




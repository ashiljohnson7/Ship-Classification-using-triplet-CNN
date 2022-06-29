from tensorflow import keras
import numpy as np
import cv2
from mean import mean1

def compute_dist(a,b):
    return np.sum(np.square(a-b))

class ShipModel(object):
    ship_list=[
        'Cargo','Dredging','Passenger','Tanker'
    ]
    def __init__(self, model_json_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            self.loaded_model = keras.models.load_model('model')

            

       

    def predict_type(self, img):
        # img1=lbp2('Data/d1/Patch_Uint8/Visual_Tanker_x24918_y9426_hv.tif')
        # img1=np.reshape([img],(1,28,28,1))
        m0=mean1(0)
        m1=mean1(1)
        m2=mean1(2)
        m3=mean1(3)
        
        li=[]

        self.preds = self.loaded_model.predict(img)[0]
        d0=compute_dist(m0,self.preds)
        print(d0)
        li.append(d0)
        d1=compute_dist(m1,self.preds)
        print(d1)
        li.append(d1)
        d2=compute_dist(m2,self.preds)
        print(d2)
        li.append(d2)
        d3=compute_dist(m3,self.preds)
        print(d3)
        li.append(d3)
        
        return ShipModel.ship_list[np.argmin(li)]
        # return self.preds


from calibrate import lbp2

if __name__ == '__main__':
    model=ShipModel("model")
    img=lbp2('Data/d1/Patch_Uint8/Visual_Cargo_x23832_y8345_hh.tif')
    
    img=np.reshape([img],(1,28,28,1))
    pred=model.predict_type(img)
    print(pred)





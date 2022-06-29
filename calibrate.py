import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
   
      
def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        pass
      
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
    center = img[x][y] 
    val_ar = [] 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 

    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 



def lbp2(path):

    img_bgr = cv2.imread(path, 1) 
    #print(img_bgr)
    img_bgr=cv2.resize(img_bgr,(28,28))
    height, width, _ = img_bgr.shape  
    img_gray = cv2.cvtColor(img_bgr, 
                            cv2.COLOR_BGR2GRAY) 
    img_lbp = np.zeros((height, width), 
                    np.uint8) 
    
    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 
    
    return img_lbp



if __name__=='__main__':
    LBP('Data\d2\Patch_Uint8\Patch_Uint8\Visual_Tanker_x31058_y6071.tif')

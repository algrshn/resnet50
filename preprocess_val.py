import numpy as np
import skimage.io
import skimage.transform
import skimage.util
import pandas as pd
import sys
import configparser

#------start reading from config.txt----------------------

config = configparser.ConfigParser()
config.read('config.txt')

try:
    ImageNet_folder=config.get('preprocess_val','ImageNet_folder')
except:
    sys.exit("Check configuration file config.txt. Option ImageNet_folder does not exist in section [preprocess_val].")
    
try:
    path_to_save=config.get('preprocess_val','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [preprocess_val].")
 
   
mu_global=np.zeros((3))
sigma_global=np.ones((3))

try:
    mu_global[0]=config.get('utils','mu_0')
except:
    sys.exit("Check configuration file config.txt. Option mu_0 does not exist in section [utils].")
try:
    mu_global[1]=config.get('utils','mu_1')
except:
    sys.exit("Check configuration file config.txt. Option mu_1 does not exist in section [utils].")
try:
    mu_global[2]=config.get('utils','mu_2')
except:
    sys.exit("Check configuration file config.txt. Option mu_2 does not exist in section [utils].")
try:
    sigma_global[0]=config.get('utils','sigma_0')
except:
    sys.exit("Check configuration file config.txt. Option sigma_0 does not exist in section [utils].")
try:
    sigma_global[1]=config.get('utils','sigma_1')
except:
    sys.exit("Check configuration file config.txt. Option sigma_1 does not exist in section [utils].")
try:
    sigma_global[2]=config.get('utils','sigma_2')
except:
    sys.exit("Check configuration file config.txt. Option sigma_2 does not exist in section [utils].")
    

#-----finish reading from config.txt------------------------

mu_shaped=np.zeros((1,1,3))
mu_shaped[0,0,:]=mu_global[:]
sigma_shaped=np.ones((1,1,3))
sigma_shaped[0,0,:]=sigma_global[:]

df=pd.read_csv(ImageNet_folder + 'val.txt',sep=' ', header=None)
src_folder=ImageNet_folder + "2012_val/"    
    
df.columns=["filename","label"]
filename_list=df['filename'].tolist()
label=df['label'].to_numpy(dtype=np.int64)

for i in range(len(filename_list)):
    
    X_batch=np.zeros((50,224,224,3),dtype=np.float32)
    y_batch=np.zeros((1),dtype=np.int64)
    
    y_batch[0]=label[i]
    
    imgfilename=src_folder + filename_list[i]
    
    img=skimage.io.imread(imgfilename)
    
    h=img.shape[0]
    w=img.shape[1]
    
    j=0
    for shorter_side in [224,256,384,480,640]:
        
        if(h<w):
            h_new=shorter_side
            w_new=(h_new*w)//h
        elif(w<h):
            w_new=shorter_side
            h_new=(w_new*h)//w
        else:
            w_new=shorter_side
            h_new=shorter_side
            
        img_resized=skimage.transform.resize(img,output_shape=[h_new,w_new,3])
        
        
        
        ulc_max_x=w_new-224
        ulc_max_y=h_new-224
       
        for pos in ['upper_left', 'upper_right', 'bottom_left', 'bottom_right', 'center']:
            
            img224=np.zeros((224,224,3),dtype=np.float32)
            
            if(pos=="upper_left"):
                ulc_x=0
                ulc_y=0
            elif(pos=="upper_right"):
                ulc_x=ulc_max_x
                ulc_y=0
            elif(pos=="bottom_left"):
                ulc_x=0
                ulc_y=ulc_max_y
            elif(pos=="bottom_right"):
                ulc_x=ulc_max_x
                ulc_y=ulc_max_y
            elif(pos=="center"):
                ulc_x=ulc_max_x//2
                ulc_y=ulc_max_y//2
            
            img224[:,:,:]=img_resized[ulc_y:(ulc_y+224),ulc_x:(ulc_x+224),:]
            
            X224=(img224-mu_shaped)/sigma_shaped
            
            
                        
            for flip in ['noflip','flip']:
                
                if(flip=="flip"):
                    
                    X224_f=np.flip(X224,axis=1)
                
                elif(flip=="noflip"):
                    
                    X224_f=X224
                    
                X_batch[j,:,:,:]=X224_f[:,:,:]
                    
                j+=1
                
               
    np.save(path_to_save + "X50_" +str(i) +".npy",np.asarray(X_batch,dtype=np.float16))
    np.save(path_to_save + "y_" + str(i) + ".npy",y_batch)               
    
    if((i+1) % 100 == 0):           
        print("Number of processed images: {}/50000".format(i+1))            
               
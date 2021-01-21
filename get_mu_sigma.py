import numpy as np
import skimage.io
import skimage.transform
import time
import argparse
import sys
import os
import configparser 

#------start reading from config.txt----------------------

config = configparser.ConfigParser()
config.read('config.txt')

try:
    preprocessed_images_folder=config.get('preprocess','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [preprocess].")
    
try:
    path_to_save=config.get('get_mu_sigma','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [get_mu_sigma].")

try:
    N=int(config.get('train','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [train].")
    
#-----finish reading from config.txt------------------------



#------start reading command line arguments----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--shorter_side',type=int)
args = parser.parse_args()

if(not args.shorter_side):
    sys.exit("Must specify shorter_side")
    
if not os.path.exists(preprocessed_images_folder + str(args.shorter_side)):
    sys.exit("Folder " + preprocessed_images_folder + str(args.shorter_side) + " doesn't exist")
    
#------finish reading command line arguments----------------------

sum_of_pixels=np.zeros((3),dtype=np.float64)
total_num_pixels=np.zeros((1),dtype=np.float64)

start=time.time()

for i in range(N):
    
    if(i % 1000 == 0):
        print("shorter_side={}px | processing mu | image #{}".format(args.shorter_side,i))
       
    imgfilename=preprocessed_images_folder + str(args.shorter_side) + "/img_" +str(i)+".jpg"
    
    img255=skimage.io.imread(imgfilename)

    img=img255/255.0
    
    
    img224=np.zeros((224,224,3),dtype=np.float16)
    
    h=img.shape[0]
    w=img.shape[1]
    
    blc_max_x=w-224
    blc_max_y=h-224
    
    blc_x=np.random.randint(low=0,high=blc_max_x+1)
    blc_y=np.random.randint(low=0,high=blc_max_y+1)
    
    img224[:,:,:]=img[blc_y:(blc_y+224),blc_x:(blc_x+224),:]
    
    np.save(path_to_save + "X_" + str(i) + ".npy",img224)
    
    
    total_num_pixels[0]+=img224.shape[0]*img224.shape[1]
    sum_of_pixels+=np.sum(img224,axis=(0,1),dtype=np.float64)
    
mu=sum_of_pixels/total_num_pixels[0]

mu_shaped=np.zeros((1,1,3))
mu_shaped[0,0,:]=mu[:]

sum_of_squares=np.zeros((3),dtype=np.float64)


for i in range(N):
    
    if(i % 1000 == 0):
        print("shorter_side={}px | processing sigma | image #{}".format(args.shorter_side,i))
       
    
    img224=np.load(path_to_save + "X_" + str(i) + ".npy")
    

       
    img_centered=img224-mu_shaped    
    img_squared=img_centered*img_centered
    sum_of_squares+=np.sum(img_squared,axis=(0,1),dtype=np.float64)


    
sigma=np.sqrt(sum_of_squares/total_num_pixels[0])

print("mu = {}".format(mu))
print("sigma = {}".format(sigma))

end=time.time()
print("Execution time: {} h {} min".format(int((end-start)//3600),int(((end-start) % 3600)//60)))  
    
    
    
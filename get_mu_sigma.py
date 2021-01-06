import numpy as np
import skimage.io
import skimage.transform
import time
import argparse
import sys
import os 

#----------hard coded data-------------------
preprocessed_images_folder="/media/alex/data2/"
N=1281167
#--------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--shorter_side',type=int)
args = parser.parse_args()

if(not args.shorter_side):
    sys.exit("Must specify shorter_side")
    
if not os.path.exists(preprocessed_images_folder + str(args.shorter_side)):
    sys.exit("Folder " + preprocessed_images_folder + str(args.shorter_side) + " doesn't exist")

sum_of_pixels=np.zeros((3),dtype=np.float64)
total_num_pixels=np.zeros((1),dtype=np.float64)

start=time.time()

for i in range(N):
    
    if(i % 1000 == 0):
        print("shorter_side={}px | processing mu | image #{}".format(args.shorter_side,i))
       
    imgfilename=preprocessed_images_folder + str(args.shorter_side) + "/img_" +str(i)+".jpg"
    
    img255=skimage.io.imread(imgfilename)

    img=img255/255.0
    
    total_num_pixels[0]+=img.shape[0]*img.shape[1]
    sum_of_pixels+=np.sum(img,axis=(0,1),dtype=np.float64)
    
mu=sum_of_pixels/total_num_pixels[0]

mu_shaped=np.zeros((1,1,3))
mu_shaped[0,0,:]=mu[:]

sum_of_squares=np.zeros((3),dtype=np.float64)


for i in range(N):
    
    if(i % 1000 == 0):
        print("shorter_side={}px | processing sigma | image #{}".format(args.shorter_side,i))
       
    imgfilename=preprocessed_images_folder + str(args.shorter_side) + "/img_" +str(i)+".jpg"
    
    img255=skimage.io.imread(imgfilename)
    
    img=img255/255.0
       
    img_centered=img-mu_shaped    
    img_squared=img_centered*img_centered
    sum_of_squares+=np.sum(img_squared,axis=(0,1),dtype=np.float64)


    
sigma=np.sqrt(sum_of_squares/total_num_pixels[0])

print("mu = {}".format(mu))
print("sigma = {}".format(sigma))

end=time.time()
print("Execution time: {} h {} min".format(int((end-start)//3600),int(((end-start) % 3600)//60)))  
    
    
    
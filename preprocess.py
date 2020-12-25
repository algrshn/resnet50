import numpy as np
import skimage.io
import skimage.transform
import skimage.util
import pandas as pd
import argparse
import sys
import time
import os
import math

#---------hard coded paths----------------
ImageNet_folder='/media/alex/data1/ImageNet/'
path_to_save="/media/alex/data2/"
#------------------------------------------

df=pd.read_csv(ImageNet_folder + 'train.txt',sep=' ', header=None)
src_folder=ImageNet_folder + "2012_train/"


parser = argparse.ArgumentParser()
parser.add_argument('--shorter_side',type=int)
args = parser.parse_args()

if(not args.shorter_side):
    sys.exit("Must specify shorter_side")
else:
    shorter_side=args.shorter_side
    
if not os.path.exists(path_to_save + str(shorter_side)):
    os.makedirs(path_to_save + str(shorter_side))

directory= os.listdir(path_to_save + str(shorter_side) + "/")
if(len(directory)!=0):
    ans = input("The target directory is not empty. Are you sure you are OK with that? (y/n): ")
    if(ans!='y' and ans!='yes' and ans!="Yes" and ans!="Y"):
        sys.exit("Program terminated")

df.columns=["filename","label"]
filename_list=df['filename'].tolist()
label=df['label'].to_numpy(dtype=np.int64)

N=len(filename_list)

shuffled_indices=np.random.permutation(N)

start=time.time()

print("Processing shorter_side={}px".format(shorter_side))

for i in range(N):
        
    print("Processing shorter_side={0}px | image# {1}/{2} | {3:3.1f}%".format(shorter_side,i,N-1,math.floor(1000*i/(N-1))/10))
    
    orig_index=shuffled_indices[i]
    
    y=np.zeros((1),dtype=np.int64)
    y[0]=label[orig_index]
    
    imgfilename=src_folder + filename_list[orig_index]
    
    img=skimage.io.imread(imgfilename)
    
    h=img.shape[0]
    w=img.shape[1]
        
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
    np.save(path_to_save + str(shorter_side) + "/y_" +str(i)+".npy",y)    
    skimage.io.imsave(path_to_save + str(shorter_side) + "/img_" +str(i)+".jpg", skimage.util.img_as_ubyte(img_resized))
    
    
    
end=time.time()
print("Execution time: {} h {} min".format(int((end-start)//3600),int(((end-start) % 3600)//60)))  
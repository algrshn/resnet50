import numpy as np
import skimage
import pandas as pd


df=pd.read_csv('/media/alex/data/ImageNet/train.txt',sep=' ', header=None)
img_folder="/media/alex/data/ImageNet/2012_train/"
path_to_save="/media/alex/data/npy/"

df.columns=["filename","label"]
filename_list=df['filename'].tolist()
y=df['label'].to_numpy(dtype=np.int64)

N=len(filename_list)

shuffled_indices=np.random.permutation(N)

for i in range(N):
    
    print(i)
    
    orig_index=shuffled_indices[i]
    
    Y=np.zeros((1,1000))
    Y[0,y[orig_index]]=1
    
    imgfilename=img_folder + filename_list[orig_index]
    
    img=skimage.io.imread(imgfilename)
    
    h=img.shape[0]
    w=img.shape[1]
    
#---------------------------------------------------------------------
    
    if(h<w):
        h_new=256
        w_new=(h_new*w)//h
    elif(w<h):
        w_new=256
        h_new=(w_new*h)//w
    else:
        w_new=256
        h_new=256
        
    img_resized=skimage.transform.resize(img,output_shape=[h_new,w_new,3])
    
    np.save(path_to_save + "X16_" +str(i),np.asarray(img_resized,dtype=np.float16))
    np.save(path_to_save + "Y_" +str(i),Y)
    
    
    
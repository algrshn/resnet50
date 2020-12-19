import numpy as np
import skimage.io
import skimage.transform
import pandas as pd
import time
from skimage.color import gray2rgb 



ImageNet_folder="/media/alex/data1/ImageNet/"

df=pd.read_csv(ImageNet_folder + 'train.txt',sep=' ', header=None)
img_folder=ImageNet_folder + '2012_train/'

df.columns=["filename","label"]
filename_list=df['filename'].tolist()


N=len(filename_list)


sum_of_pixels=np.zeros((3),dtype=np.float64)
total_num_pixels=np.zeros((1),dtype=np.float64)

start=time.time()

for i in range(294566,294567):
    
    if(i % 1000 == 0):
        print("Processing mu | image #{}".format(i))
       
    imgfilename=img_folder + filename_list[i]
    
    img255=skimage.io.imread(imgfilename)
    
    print(imgfilename)
    
    print(img255.shape)
       
    if(len(img255.shape)!=3):
        img255=gray2rgb(img255)

    img=img255/255.0
    
    total_num_pixels[0]+=img.shape[0]*img.shape[1]
    print(i)
    print(img.shape)
    sum_of_pixels+=np.sum(img,axis=(0,1),dtype=np.float64)
    
mu=sum_of_pixels/total_num_pixels[0]

# mu_shaped=np.zeros((1,1,3))
# mu_shaped[0,0,:]=mu[:]

# sum_of_squares=np.zeros((3),dtype=np.float64)


# for i in range(N):
    
#     if(i % 1000 == 0):
#         print("Processing sigma | image #{}".format(i))
       
#     imgfilename=img_folder + filename_list[i]
    
#     img255=skimage.io.imread(imgfilename)
    
#     if(len(img255.shape)!=3):
#         img255=gray2rgb(img255)

#     img=img255/255.0
       
#     img_centered=img-mu_shaped    
#     img_squared=img_centered*img_centered
#     sum_of_squares+=np.sum(img_squared,axis=(0,1),dtype=np.float64)


    
# sigma=np.sqrt(sum_of_squares/total_num_pixels[0])

print("mu = {}".format(mu))
#print("sigma = {}".format(sigma))

end=time.time()
print(end-start)   
    
    
    
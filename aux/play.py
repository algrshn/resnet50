import numpy as np
import skimage
import pandas as pd


mu=np.array([0.47703891, 0.45346393, 0.40395429])  #actual values
sigma=np.array([0.27914875, 0.27181716, 0.28552585])  #actual values
mu_113=np.zeros((1,1,3))
sigma_113=np.zeros((1,1,3))
mu_113[0,0,:]=mu[:]
sigma_113[0,0,:]=sigma[:]

n=np.random.randint(1281000)
#n=np.random.randint(49990)
for i in range(n,n+30):
    
    X16=np.load("/media/alex/data/npy/X16_" + str(i) + ".npy")
    Y=np.load("/media/alex/data/npy/Y_" + str(i) + ".npy")
    
    X16_img=sigma_113*X16+mu_113
    
    df=pd.read_csv('/media/alex/data/ImageNet/synsets.txt',sep=' ', header=None)
    df.columns=["id"]
    
    id_list=df['id'].tolist()
    
    print(id_list[np.argmax(Y)])
    
    print(np.amax(X16_img))
    print(np.amin(X16_img))

    skimage.io.imshow(np.asarray(X16_img,dtype=np.float32))
    skimage.io.show()


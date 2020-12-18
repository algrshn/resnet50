import numpy as np
import math
import sys
import os
import glob



def progress(count, total):
    bar_len = 100
    filled_len = int(math.floor(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()
    
def process256(X256):
    
    #crop
    img224=np.zeros((224,224,3),dtype=np.float32)
    
    h=X256.shape[0]
    w=X256.shape[1]
    
    blc_max_x=w-224
    blc_max_y=h-224
    
    blc_x=np.random.randint(low=0,high=blc_max_x+1)
    blc_y=np.random.randint(low=0,high=blc_max_y+1)
    
    img224[:,:,:]=X256[blc_y:(blc_y+224),blc_x:(blc_x+224),:]
    
    #flip
    if(np.random.randint(2)==1):
        img224_f=np.flip(img224,axis=1)
    else:
        img224_f=img224

    #color augmentation
    I=np.reshape(img224_f,newshape=(50176,3))
    mu=np.mean(I,axis=0,keepdims=True)
    
    I_c=I-mu
    cov_matrix=(1/50176)*np.matmul(np.transpose(I_c),I_c)
    
    w,v=np.linalg.eigh(cov_matrix)
    
    color_augmentation=np.zeros((1,1,3))
    
    v0=np.zeros((1,1,3))
    v1=np.zeros((1,1,3))
    v2=np.zeros((1,1,3))
    v0[0,0,:]=v[:,0]
    v1[0,0,:]=v[:,1]
    v2[0,0,:]=v[:,2]
    
    alpha0=0.1*np.random.randn()
    alpha1=0.1*np.random.randn()
    alpha2=0.1*np.random.randn()
    
    color_augmentation=alpha0*w[0]*v0+alpha1*w[1]*v1+alpha2*w[2]*v2
    
    img_augmented=img224_f+color_augmentation
    
    return img_augmented


def process256c(X256):
    
    #center crop
    img224=np.zeros((224,224,3),dtype=np.float32)
    
    h=X256.shape[0]
    w=X256.shape[1]
    
    blc_max_x=w-224
    blc_max_y=h-224
    
    blc_x=blc_max_x//2
    blc_y=blc_max_y//2
    
    img224[:,:,:]=X256[blc_y:(blc_y+224),blc_x:(blc_x+224),:]
        
    return img224

def run_preprocessing(ds_folder, buffer_folder, shuffled_indices, epoch, batch_num, batch_size):
    
    X_batch=np.zeros((batch_size,224,224,3),dtype=np.float32)
    Y_batch=np.zeros((batch_size,1000),dtype=np.int8)
    
    for j in range(batch_size):
        
        orig_index=shuffled_indices[batch_num*batch_size+j]
        
        X256=np.load(ds_folder + "X16_" + str(orig_index) + ".npy")
        Y=np.load(ds_folder + "Y_" + str(orig_index) + ".npy") 
        
        Y_batch[j,:]=Y[0,:]
        
        X224=process256(X256)
                   
        X_batch[j,:,:,:]=X224[:,:,:]
    
    print("Epoch: {} | batch_num: {}".format(epoch, batch_num))
    
    np.save(buffer_folder + "X16_" +str(epoch)+"_"+str(batch_num)+".npy",np.asarray(X_batch,dtype=np.float16))
    np.save(buffer_folder + "Y_" +str(epoch)+"_"+str(batch_num)+".npy",Y_batch)

    
    
def delete_from_buffer(buffer_folder, training_epoch, training_batch_num):
    
    for epoch in range(0,training_epoch):
    
        fileListX=glob.glob(buffer_folder+"X16_" + str(epoch) + "_*.npy")
        fileListY=glob.glob(buffer_folder+"Y_" + str(epoch) + "_*.npy")
    
        for filePath in fileListX:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file {}".format(filePath))
                
        for filePath in fileListY:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file {}".format(filePath))
                
    for batch_num in range(0,training_batch_num):
        
        fileListX=glob.glob(buffer_folder + "X16_" +str(training_epoch)+"_"+str(batch_num)+".npy")
        fileListY=glob.glob(buffer_folder + "Y_" +str(training_epoch)+"_"+str(batch_num)+".npy")
        
        for filePath in fileListX:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file {}".format(filePath))
                
        for filePath in fileListY:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file {}".format(filePath))
        
import numpy as np
import math
import sys
import os
import glob
import configparser
import skimage.io



def progress(count, total):
    bar_len = 100
    filled_len = int(math.floor(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()
    
def img2X224(imgfilename, mode):
    
    img_uint8=skimage.io.imread(imgfilename)
    img=img_uint8/255.0
        
    #crop
    img224=np.zeros((224,224,3),dtype=np.float32)
    
    h=img.shape[0]
    w=img.shape[1]
    
    blc_max_x=w-224
    blc_max_y=h-224
    
    blc_x=np.random.randint(low=0,high=blc_max_x+1)
    blc_y=np.random.randint(low=0,high=blc_max_y+1)
    
    img224[:,:,:]=img[blc_y:(blc_y+224),blc_x:(blc_x+224),:]
    
    #linear transformation with mu and sigma
    
    mu_global=np.zeros((3))
    sigma_global=np.ones((3))
    
    config = configparser.ConfigParser()
    config.read('config.txt')
    
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
    
    
    
    mu_shaped=np.zeros((1,1,3))
    mu_shaped[0,0,:]=mu_global[:]
    sigma_shaped=np.ones((1,1,3))
    sigma_shaped[0,0,:]=sigma_global[:]
    
    X224=(img224-mu_shaped)/sigma_shaped
    
    
    #flip
    if(np.random.randint(2)==1):
        X224_f=np.flip(X224,axis=1)
    else:
        X224_f=X224

    #color augmentation
    I=np.reshape(X224_f,newshape=(50176,3))
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
    
    X224_augmented=X224_f+color_augmentation
    
    return X224_augmented


class Run_preprocessing(object):
    def __init__(self, ds_location, buffer_folder, shuffled_indices, epoch, batch_num_start, batch_num_end, batch_size, num_of_threads, mode, iter_num=0):
        self.ds_location = ds_location
        self.buffer_folder = buffer_folder
        self.shuffled_indices = shuffled_indices
        self.epoch = epoch
        self.batch_num_start = batch_num_start
        self.batch_num_end = batch_num_end
        self.batch_size = batch_size
        self.num_of_threads = num_of_threads
        self.mode = mode
        self.iter_num = iter_num
    def __call__(self, thread_num):
        
        for batch_num in range(self.batch_num_start,self.batch_num_end):
            
            if(batch_num % self.num_of_threads == thread_num):
            
                X_batch=np.zeros((self.batch_size,224,224,3),dtype=np.float32)
                if(self.mode=="train"):
                    Y_batch=np.zeros((self.batch_size,1000),dtype=np.int8)
            
                for j in range(self.batch_size):
                    
                    orig_index=self.shuffled_indices[batch_num*self.batch_size+j]
                    
                    shorter_side=256+7*np.random.randint(33)
                    imgfilename=self.ds_location + str(shorter_side) + "/img_" +str(orig_index)+".jpg"
                    
                    if(self.mode=="train"):
                        y=np.load(self.ds_location + str(shorter_side) + "/y_" +str(orig_index)+".npy")
                        Y=np.zeros((1,1000),dtype=np.int64)
                        Y[0,int(y)]=1
                        Y_batch[j,:]=Y[0,:]
                    
                    X224=img2X224(imgfilename,self.mode)
                               
                    X_batch[j,:,:,:]=X224[:,:,:]
                
                if(self.mode=="train"):
                    print("Epoch: {} | batch_num: {}".format(self.epoch, batch_num))
                    np.save(self.buffer_folder + "X16_" +str(self.epoch)+"_"+str(batch_num)+".npy",np.asarray(X_batch,dtype=np.float16))
                    np.save(self.buffer_folder + "Y_" +str(self.epoch)+"_"+str(batch_num)+".npy",Y_batch)
                elif(self.mode=="calc_bn_avgs"):
                    print("Epoch: {} | iter_num: {} | batch_num: {}".format(self.epoch, self.iter_num, batch_num))
                    np.save(self.buffer_folder + "X16_" +str(self.epoch)+"_" +str(self.iter_num)+ "_" +str(batch_num)+".npy",np.asarray(X_batch,dtype=np.float16))

    
    
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
        
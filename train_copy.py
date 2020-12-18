import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import argparse
import sys

import model as mdl
import utils

#mu=np.array([0.47703891, 0.45346393, 0.40395429])  #actual values
#sigma=np.array([0.27914875, 0.27181716, 0.28552585])  #actual values

#-------------hard coded data----------------------------
ds_folder="/media/alex/data/npy/"
path_for_saving="/home/alex/ResNet/saved_models/"
N=1281167
#--------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_folder',type=str)
parser.add_argument('--epoch_start',type=int)
parser.add_argument('--epoch_end',type=int)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--wd',type=float)
args = parser.parse_args()

if(not args.run_folder):
    sys.exit("Must specify run_folder")

if(not args.epoch_start and args.epoch_start!=0):
    sys.exit("Must specify epoch_start")
    
if(not args.epoch_end and args.epoch_end!=0):
    sys.exit("Must specify epoch_end")
    
if(not args.batch_size):
    sys.exit("Must specify batch_size")
    
if(not args.learning_rate):
    sys.exit("Must specify learning_rate")

if(not args.wd):
    sys.exit("Must specify wd (weight_decay)")

batch_size=args.batch_size
 

X_batch=np.zeros((batch_size,224,224,3),dtype=np.float32)
Y_batch=np.zeros((batch_size,1000),dtype=np.int8)

num_of_batches=N//batch_size

model=mdl.Model()


if(args.epoch_start>0):
    path_to_saved_model=path_for_saving + args.run_folder + '/epoch' + str(args.epoch_start-1) + '/'
    loaded = tf.saved_model.load(path_to_saved_model)

    model.b_start=loaded.b_start
    model.w_start=loaded.w_start
    model.beta_start=loaded.beta_start
    model.gamma_start=loaded.gamma_start
    
    model.b=loaded.b
    model.w=loaded.w
    model.beta=loaded.beta
    model.gamma=loaded.gamma
    
    model.dense_b=loaded.dense_b
    model.dense_w=loaded.dense_w

#opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
opt = tfa.optimizers.SGDW(learning_rate=args.learning_rate, weight_decay=args.wd, momentum=0.9)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

for epoch in range(args.epoch_start,args.epoch_end):
    e_start=time.time()
    print("\n\n\nTraining epoch# {}".format(epoch))    
    
    shuffled_indices=np.random.permutation(N)
    
    #for batch_num in range(10):
    for batch_num in range(num_of_batches):

        utils.progress(batch_num,num_of_batches-1)
               
        for j in range(batch_size):
            
            orig_index=shuffled_indices[batch_num*batch_size+j]
            
            X256=np.load(ds_folder + "X16_" + str(orig_index) + ".npy")
            Y=np.load(ds_folder + "Y_" + str(orig_index) + ".npy") 
            
            Y_batch[j,:]=Y[0,:]
            
            X224=utils.process256(X256)
                       
            X_batch[j,:,:,:]=X224[:,:,:]
        
        
        Xtf=tf.convert_to_tensor(X_batch,dtype=tf.dtypes.float32)
        Ytf=tf.convert_to_tensor(Y_batch,dtype=tf.dtypes.float32)
        
        model.train_step(opt,Xtf,Ytf)

        
    save_folder=path_for_saving + args.run_folder + "/epoch" + str(epoch) +"/"      
    tf.saved_model.save(model, save_folder)
           
    e_end=time.time()
    print("\nEpoch execution time: {0:4.0f}min".format((e_end-e_start)/60))





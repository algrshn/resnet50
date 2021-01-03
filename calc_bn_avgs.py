import tensorflow as tf
import numpy as np
import time
import argparse
import sys

import model as mdl
import utils2

#mu=np.array([0.47703891, 0.45346393, 0.40395429])  #actual values
#sigma=np.array([0.27914875, 0.27181716, 0.28552585])  #actual values

#-------------hard coded data----------------------------
ds_folder="/media/alex/data/npy/"
ds_folder_val="/media/alex/data/npy_val/"
path_to_saved_models="/home/alex/ResNet/saved_models/"
N=1281167
N_val=50000
#--------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_folder',type=str)
parser.add_argument('--epoch_start',type=int)
parser.add_argument('--epoch_end',type=int)
parser.add_argument('--batch_size',type=int)
args = parser.parse_args()

if(not args.run_folder):
    sys.exit("Must specify run_folder")

if(not args.epoch_start and args.epoch_start!=0):
    sys.exit("Must specify epoch_start")
    
if(not args.epoch_end and args.epoch_end!=0):
    sys.exit("Must specify epoch_end")
    
if(not args.batch_size):
    sys.exit("Must specify batch_size")
    

batch_size=args.batch_size
 

X_batch=np.zeros((batch_size,224,224,3),dtype=np.float32)

num_of_batches=N//batch_size
num_of_batches_val=N_val//batch_size


for epoch in range(args.epoch_start,args.epoch_end):
    e_start=time.time()
    
          
    model=mdl.Model()

    opt = tf.keras.optimizers.Adam(learning_rate=0.0)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    

    success_loading_model=0
    while (success_loading_model==0):
        try:
            loaded = tf.saved_model.load(path_to_saved_models + args.run_folder + '/epoch' + str(epoch) + '/')
            success_loading_model=1            
        except:
            print("Model is not yet available, will try again in 10 mins...")
            time.sleep(600)

    print("\n\n\nModel loaded successfully")
    print("Processing epoch# {}".format(epoch))  
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


    #set all mu and V to zero
    for i in range(len(model.mu)):
        model.mu[i].assign(0.0*model.mu[i])
        model.V[i].assign(0.0*model.V[i])
    model.mu_start.assign(0.0*model.mu_start)
    model.V_start.assign(0.0*model.V_start)
    model.curr_num_of_record_steps.assign(0.0)
    
    
    
    shuffled_indices=np.random.permutation(N)
    
    for batch_num in range(num_of_batches):

        utils2.progress(batch_num,num_of_batches-1)
               
        for j in range(batch_size):
            
            orig_index=shuffled_indices[batch_num*batch_size+j]
            
            X256=np.load(ds_folder + "X16_" + str(orig_index) + ".npy")
            
            X224=utils2.process256c(X256)
                       
            X_batch[j,:,:,:]=X224[:,:,:]
        
        
        Xtf=tf.convert_to_tensor(X_batch,dtype=tf.dtypes.float32)
        
        model(Xtf,mode='calc_bn_avgs')

        
    save_folder=path_to_saved_models + args.run_folder + "/epoch" + str(epoch) +"_/"      
    tf.saved_model.save(model, save_folder)
              
    e_end=time.time()
    print("\nEpoch processing time: {0:4.0f}min".format((e_end-e_start)/60))





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
ds_folder="/media/alex/data/npy_val/"
path_to_saved_models="/home/alex/ResNet/saved_models/"
N=50000
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
Y_batch=np.zeros((batch_size,1000),dtype=np.int8)

num_of_batches=N//batch_size


for epoch in range(args.epoch_start,args.epoch_end):
    e_start=time.time()
    print("\n\n\nProcessing epoch# {}".format(epoch))
          
    model=mdl.Model()

    opt = tf.keras.optimizers.Adam(learning_rate=0.0)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
          
    loaded = tf.saved_model.load(path_to_saved_models + args.run_folder + '/epoch' + str(epoch) + '_/')      
    model.b_start=loaded.b_start
    model.w_start=loaded.w_start
    model.beta_start=loaded.beta_start
    model.gamma_start=loaded.gamma_start
    model.mu_start=loaded.mu_start
    model.V_start=loaded.V_start
    
    model.b=loaded.b
    model.w=loaded.w
    model.beta=loaded.beta
    model.gamma=loaded.gamma
    model.mu=loaded.mu
    model.V=loaded.V
    
    model.dense_b=loaded.dense_b
    model.dense_w=loaded.dense_w 
    
    
    shuffled_indices=np.random.permutation(N)
    
    
    corr1=tf.Variable(0)
    corr5=tf.Variable(0)
    for batch_num in range(num_of_batches):

        utils2.progress(batch_num,num_of_batches-1)
               
        for j in range(batch_size):
                    
            orig_index=shuffled_indices[batch_num*batch_size+j]
            
            X256=np.load(ds_folder + "X16_" + str(orig_index) + ".npy")
            Y=np.load(ds_folder + "Y_" + str(orig_index) + ".npy") 
            
            Y_batch[j,:]=Y[0,:]
            
            X224=utils2.process256c(X256)
                       
            X_batch[j,:,:,:]=X224[:,:,:]
        
        
        Xtf=tf.convert_to_tensor(X_batch,dtype=tf.dtypes.float32)
        Ytf=tf.convert_to_tensor(Y_batch,dtype=tf.dtypes.float32)
        
        y=tf.math.argmax(Ytf,axis=1,output_type=tf.dtypes.int32)
        y1=tf.reshape(y,shape=[batch_size,1])
        
        
        A=model(Xtf,mode='inference')
      
        p1=tf.math.argmax(A,axis=1,output_type=tf.dtypes.int32)
        P5=tf.math.top_k(A,k=5)[1]
    
        corr1.assign_add(tf.reduce_sum(tf.cast(tf.equal(p1,y),dtype=tf.dtypes.int32)))
        corr5.assign_add(tf.reduce_sum(tf.cast(tf.equal(P5,y1),dtype=tf.dtypes.int32)))

    acc1=np.round(100*(corr1/(batch_size*num_of_batches)),decimals=2)
    acc5=np.round(100*(corr5/(batch_size*num_of_batches)),decimals=2)
    print('\nEpoch# {}'.format(epoch))
    print('Top 1 accuracy: {}%'.format(acc1))
    print('Top 5 accuracy: {}%'.format(acc5))
    print("-------------------")



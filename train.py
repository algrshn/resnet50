import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
import time
import argparse
import sys
import configparser

import model as mdl

#------start reading from config.txt----------------------

config = configparser.ConfigParser()
config.read('config.txt')

try:
    buffer_folder=config.get('train','buffer_folder')
except:
    sys.exit("Check configuration file config.txt. Option buffer_folder does not exist in section [train].")
    
try:
    path_for_saving=config.get('train','path_for_saving')
except:
    sys.exit("Check configuration file config.txt. Option path_for_saving does not exist in section [train].")

try:
    N=int(config.get('train','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [train].")
    
try:
    buffer_size=int(config.get('train','buffer_size'))
except:
    sys.exit("Check configuration file config.txt. Option buffer_size does not exist in section [train].")
    
try:
    sleep_time=int(config.get('train','sleep_time'))
except:
    sys.exit("Check configuration file config.txt. Option sleep_time does not exist in section [train].")
    
try:
    sgd_momentum=float(config.get('train','sgd_momentum'))
except:
    sys.exit("Check configuration file config.txt. Option sgd_momentum does not exist in section [train].")

#-----finish reading from config.txt------------------------


#------start reading command line arguments----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_folder',type=str)
parser.add_argument('--epoch_start',type=int)
parser.add_argument('--epoch_end',type=int)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--learning_rate',type=float)
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

#------finish reading command line arguments----------------------

batch_size=args.batch_size
num_of_batches=N//batch_size

model=mdl.Model()

if(args.epoch_start>0):
    path_to_saved_model=path_for_saving + args.run_folder + '/epoch' + str(args.epoch_start-1) + '/'
    loaded = tf.saved_model.load(path_to_saved_model)
    
    model.b=loaded.b
    model.w=loaded.w
    model.beta=loaded.beta
    model.gamma=loaded.gamma
    
    model.b_start=loaded.b_start
    model.w_start=loaded.w_start
    model.beta_start=loaded.beta_start
    model.gamma_start=loaded.gamma_start
            
    model.dense_b=loaded.dense_b
    model.dense_w=loaded.dense_w
            
    model.mu_start=loaded.mu_start
    model.sigma_start=loaded.sigma_start
    
    model.mu=loaded.mu
    model.sigma=loaded.sigma
    
    model.train_step_num=loaded.train_step_num
    model.rmax=loaded.rmax
    model.dmax=loaded.dmax
       

#learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([400180,800360,1400630,2000900], [0.1,0.01,0.001,0.0001,0.00001])

opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=sgd_momentum)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

for epoch in range(args.epoch_start,args.epoch_end):
    e_start=time.time()
    print("\n\n\nTraining epoch# {}".format(epoch))    
    

    for batch_num in range(num_of_batches):

        # utils.progress(batch_num,num_of_batches-1)
        
        keeptrying=1
        while keeptrying:
            try:
                keeptrying=0
                X_batch=np.load(buffer_folder + "X16_" +str(epoch)+"_"+str(batch_num)+".npy")
                Y_batch=np.load(buffer_folder + "Y_" +str(epoch)+"_"+str(batch_num)+".npy")
            except:
                keeptrying=1
                print("Batch not yet available, will wait for " + str(sleep_time) + "sec")
                time.sleep(sleep_time)
        
        
        Xtf=tf.convert_to_tensor(X_batch,dtype=tf.dtypes.float32)
        Ytf=tf.convert_to_tensor(Y_batch,dtype=tf.dtypes.float32)
        
        model.train_step(opt,Xtf,Ytf)
        
        if(batch_num % (buffer_size//5) ==0):
            with open("training_progress.txt",'w') as fo:
                fo.write(str(epoch))
                fo.write("\n")
                fo.write(str(batch_num))
            print("Epoch: {} | batch_num: {}".format(epoch, batch_num)) 

        
    save_folder=path_for_saving + args.run_folder + "/epoch" + str(epoch) +"/"      
    tf.saved_model.save(model, save_folder)
           
    e_end=time.time()
    print("\nEpoch execution time: {0:4.0f}min".format((e_end-e_start)/60))





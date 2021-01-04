import tensorflow as tf
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
    buffer_folder=config.get('calc_bn_avgs','buffer_folder')
except:
    sys.exit("Check configuration file config.txt. Option buffer_folder does not exist in section [calc_bn_avgs].")
    
try:
    path_to_saved_models=config.get('train','path_for_saving')
except:
    sys.exit("Check configuration file config.txt. Option path_for_saving does not exist in section [train].")

try:
    N=int(config.get('train','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [train].")
    
try:
    buffer_size=int(config.get('calc_bn_avgs','buffer_size'))
except:
    sys.exit("Check configuration file config.txt. Option buffer_size does not exist in section [calc_bn_avgs].")
    
try:
    sleep_time=int(config.get('calc_bn_avgs','sleep_time'))
except:
    sys.exit("Check configuration file config.txt. Option sleep_time does not exist in section [calc_bn_avgs].")
    
try:
    num_of_iterations=int(config.get('calc_bn_avgs','num_of_iterations'))
except:
    sys.exit("Check configuration file config.txt. Option num_of_iterations does not exist in section [calc_bn_avgs].")

#-----finish reading from config.txt------------------------


#------start reading command line arguments----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run_folder',type=str)
parser.add_argument('--epoch',type=int)
parser.add_argument('--batch_size',type=int)
args = parser.parse_args()

if(not args.run_folder):
    sys.exit("Must specify run_folder")

if(not args.epoch and args.epoch!=0):
    sys.exit("Must specify epoch")
        
if(not args.batch_size):
    sys.exit("Must specify batch_size")
#------finish reading command line arguments----------------------    

batch_size=args.batch_size
num_of_batches=N//batch_size


model=mdl.Model()

opt = tf.keras.optimizers.Adam(learning_rate=0.0)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
 
loaded = tf.saved_model.load(path_to_saved_models + args.run_folder + '/epoch' + str(args.epoch) + '/')

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


for iter_num in range(num_of_iterations):
    
    e_start=time.time()
           
    for batch_num in range(num_of_batches):

        keeptrying=1
        while keeptrying:
            try:
                keeptrying=0
                X_batch=np.load(buffer_folder + "X16_" + str(args.epoch) +"_" + str(iter_num) + "_" + str(batch_num)+".npy")
            except:
                keeptrying=1
                print("Batch not yet available, will wait for " + str(sleep_time) + "sec")
                time.sleep(sleep_time)
         
        Xtf=tf.convert_to_tensor(X_batch,dtype=tf.dtypes.float32)
        
        model(Xtf,mode='calc_bn_avgs')
        
        if(batch_num % (buffer_size//5) ==0):
            with open("bn_avgs_progress.txt",'w') as fo:
                fo.write(str(iter_num))
                fo.write("\n")
                fo.write(str(batch_num))
            print("Iteration num: {} | batch_num: {}".format(iter_num, batch_num))


    e_end=time.time()
    print("\nIteration {0} processing time: {1:4.0f}min".format(iter_num,(e_end-e_start)/60))
        
save_folder=path_to_saved_models + args.run_folder + "/epoch" + str(args.epoch) +"_/"      
tf.saved_model.save(model, save_folder)
              






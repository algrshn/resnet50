import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import configparser

import model as mdl
import utils

#------start reading from config.txt----------------------

config = configparser.ConfigParser()
config.read('config.txt')
   
try:
    ds_folder=config.get('preprocess_val','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [preprocess_val].")

try:
    path_to_saved_models=config.get('train','path_for_saving')
except:
    sys.exit("Check configuration file config.txt. Option path_for_saving does not exist in section [train].")

try:
    N=int(config.get('preprocess_val','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [preprocess_val].")

#-----finish reading from config.txt------------------------



#------start reading command line arguments----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_folder',type=str)
parser.add_argument('--epoch',type=int)
args = parser.parse_args()

if(not args.run_folder):
    sys.exit("Must specify run_folder")

if(not args.epoch and args.epoch!=0):
    sys.exit("Must specify epoch")
       
#------finish reading command line arguments----------------------


e_start=time.time()

model=mdl.Model()

opt = tf.keras.optimizers.Adam(learning_rate=0.0)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

try:      
    loaded = tf.saved_model.load(path_to_saved_models + args.run_folder + '/epoch' + str(args.epoch) + '_/')
except:
    sys.exit("Can't load the model from epoch" + str(args.epoch) + "_ folder. Most likely you haven't run calc_bn_avgs.py for epoch " + str(args.epoch) + ".")

      
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


corr1=tf.Variable(0)
corr5=tf.Variable(0)
for i in range(N):

    utils.progress(i,N-1)
                             
    Xnp=np.load(ds_folder + "X50_" +str(i) +".npy")
    ynp=np.load(ds_folder + "y_" + str(i) + ".npy")
    y=ynp[0]
    
    Xtf=tf.convert_to_tensor(Xnp,dtype=tf.dtypes.float32)
    
    A=model(Xtf,mode='inference')
    
    proto_tensor = tf.make_tensor_proto(A)
    B=tf.make_ndarray(proto_tensor)
    
    A_avg=np.mean(B, axis=0, keepdims=False)
    
    b=np.argsort(A_avg, axis=0)
    c=np.flip(b, axis=0)
    
    if(y==c[0]):
        corr1+=1
    if(y==c[0] or y==c[1] or y==c[2] or y==c[3] or y==c[4]):
        corr5+=1


acc1=np.round(100*(corr1/N),decimals=2)
acc5=np.round(100*(corr5/N),decimals=2)
print('\nEpoch# {}'.format(args.epoch))
print('Top 1 accuracy: {}%'.format(acc1))
print('Top 5 accuracy: {}%'.format(acc5))
print("-------------------")



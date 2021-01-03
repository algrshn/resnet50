import numpy as np
import argparse
import sys
import time
import configparser
from multiprocessing import Pool

import utils

#------start reading from config.txt----------------------

config = configparser.ConfigParser()
config.read('config.txt')

try:
    buffer_folder=config.get('train','buffer_folder')
except:
    sys.exit("Check configuration file config.txt. Option buffer_folder does not exist in section [train].")
    
try:
    ds_location=config.get('preprocess','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [preprocess].")

try:
    N=int(config.get('train','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [train].")
    
try:
    buffer_size=int(config.get('train','buffer_size'))
except:
    sys.exit("Check configuration file config.txt. Option buffer_size does not exist in section [train].")
    
try:
    sleep_time=int(config.get('runtime_preprocessing','sleep_time'))
except:
    sys.exit("Check configuration file config.txt. Option sleep_time does not exist in section [runtime_preprocessing].")
    
try:
    num_of_threads=int(config.get('runtime_preprocessing','num_of_threads'))
except:
    sys.exit("Check configuration file config.txt. Option num_of_threads does not exist in section [runtime_preprocessing].")

#-----finish reading from config.txt------------------------


#------start reading command line arguments----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epoch_start',type=int)
parser.add_argument('--epoch_end',type=int)
parser.add_argument('--batch_size',type=int)
args = parser.parse_args()

if(not args.epoch_start and args.epoch_start!=0):
    sys.exit("Must specify epoch_start")
    
if(not args.epoch_end and args.epoch_end!=0):
    sys.exit("Must specify epoch_end")
    
if(not args.batch_size):
    sys.exit("Must specify batch_size")
    
#------finish reading command line arguments----------------------
    
batch_size=args.batch_size
num_of_batches=N//batch_size

keepgoing=1

shuffled_indices=np.random.permutation(N)
preprocessing_epoch=args.epoch_start
preprocessing_batch_num=0

pool_input_list=[]
for i in range(num_of_threads):
    pool_input_list.append(i)

while keepgoing:
    
    line_num=0
    with open("training_progress.txt") as fo:
        for line in fo:
            if(line_num==0):
                training_epoch=int(line)
            elif(line_num==1):
                training_batch_num=int(line)
            line_num+=1
    
    
    utils.delete_from_buffer(buffer_folder, training_epoch, training_batch_num)


    
    current_buffer=(preprocessing_epoch-training_epoch)*num_of_batches+preprocessing_batch_num-training_batch_num
    if(current_buffer<0.8*buffer_size):

        if((training_batch_num+buffer_size)>num_of_batches):
            target_preprocessing_epoch=training_epoch+1
            target_preprocessing_batch_num=training_batch_num+buffer_size-num_of_batches
        else:
            target_preprocessing_epoch=training_epoch
            target_preprocessing_batch_num=training_batch_num+buffer_size
            
        if(target_preprocessing_epoch==preprocessing_epoch):

            # utils.run_preprocessing(ds_location,buffer_folder, shuffled_indices, preprocessing_epoch, preprocessing_batch_num, target_preprocessing_batch_num, batch_size, num_of_threads)
            
            try:
                pool = Pool(num_of_threads) 
                run_preprocessing = utils.Run_preprocessing(ds_location,buffer_folder, shuffled_indices, preprocessing_epoch, preprocessing_batch_num, target_preprocessing_batch_num, batch_size, num_of_threads, mode='train')
                
                
                
                pool.map(run_preprocessing, pool_input_list)
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
            
            
            
                
        elif(target_preprocessing_epoch==(preprocessing_epoch+1)):
            
            # utils.run_preprocessing(ds_location,buffer_folder, shuffled_indices, preprocessing_epoch, preprocessing_batch_num, num_of_batches, batch_size, num_of_threads)
            
            
            try:
                pool = Pool(num_of_threads) 
                run_preprocessing = utils.Run_preprocessing(ds_location,buffer_folder, shuffled_indices, preprocessing_epoch, preprocessing_batch_num, num_of_batches, batch_size, num_of_threads, mode='train')
                pool.map(run_preprocessing, pool_input_list)
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
            
            
                
            shuffled_indices=np.random.permutation(N)
            print("Performed re-shuffling")
            
            if(target_preprocessing_epoch==args.epoch_end):
                keepgoing=0
            else:
                #utils.run_preprocessing(ds_location, buffer_folder, shuffled_indices, target_preprocessing_epoch, 0, target_preprocessing_batch_num, batch_size, num_of_threads)
                
                try:
                    pool = Pool(num_of_threads) 
                    run_preprocessing = utils.Run_preprocessing(ds_location, buffer_folder, shuffled_indices, target_preprocessing_epoch, 0, target_preprocessing_batch_num, batch_size, num_of_threads, mode='train')
                    pool.map(run_preprocessing, pool_input_list)
                finally: # To make sure processes are closed in the end, even if errors happen
                    pool.close()
                    pool.join()
                
        preprocessing_batch_num=target_preprocessing_batch_num
        preprocessing_epoch=target_preprocessing_epoch
        
    else:
        
        print("Waiting...")
        time.sleep(sleep_time)
            
        
    



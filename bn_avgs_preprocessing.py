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
    buffer_folder=config.get('calc_bn_avgs','buffer_folder')
except:
    sys.exit("Check configuration file config.txt. Option buffer_folder does not exist in section [calc_bn_avgs].")
    
try:
    ds_location=config.get('preprocess','path_to_save')
except:
    sys.exit("Check configuration file config.txt. Option path_to_save does not exist in section [preprocess].")

try:
    N=int(config.get('train','N'))
except:
    sys.exit("Check configuration file config.txt. Option N does not exist in section [train].")
    
try:
    buffer_size=int(config.get('calc_bn_avgs','buffer_size'))
except:
    sys.exit("Check configuration file config.txt. Option buffer_size does not exist in section [calc_bn_avgs].")
    
try:
    num_of_iterations=int(config.get('calc_bn_avgs','num_of_iterations'))
except:
    sys.exit("Check configuration file config.txt. Option num_of_iterations does not exist in section [calc_bn_avgs].")
    
try:
    sleep_time=int(config.get('bn_avgs_preprocessing','sleep_time'))
except:
    sys.exit("Check configuration file config.txt. Option sleep_time does not exist in section [bn_avgs_preprocessing].")
    
try:
    num_of_threads=int(config.get('bn_avgs_preprocessing','num_of_threads'))
except:
    sys.exit("Check configuration file config.txt. Option num_of_threads does not exist in section [bn_avgs_preprocessing].")

#-----finish reading from config.txt------------------------


#------start reading command line arguments----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int)
parser.add_argument('--batch_size',type=int)
args = parser.parse_args()

if(not args.epoch and args.epoch!=0):
    sys.exit("Must specify epoch")
        
if(not args.batch_size):
    sys.exit("Must specify batch_size")
    
#------finish reading command line arguments----------------------
    
batch_size=args.batch_size
num_of_batches=N//batch_size

keepgoing=1

shuffled_indices=np.random.permutation(N)
preprocessing_iter_num=0
preprocessing_batch_num=0

pool_input_list=[]
for i in range(num_of_threads):
    pool_input_list.append(i)

while keepgoing:
    
    line_num=0
    with open("bn_avgs_progress.txt") as fo:
        for line in fo:
            if(line_num==0):
                calc_iter_num=int(line)
            elif(line_num==1):
                calc_batch_num=int(line)
            line_num+=1
    
    
    utils.delete_from_buffer_bn_avgs(buffer_folder, args.epoch, calc_iter_num, calc_batch_num)


    
    current_buffer=(preprocessing_iter_num-calc_iter_num)*num_of_batches+preprocessing_batch_num-calc_batch_num
    if(current_buffer<0.8*buffer_size):

        if((calc_batch_num+buffer_size)>num_of_batches):
            
            target_preprocessing_iter_num=calc_iter_num+1                        
            target_preprocessing_batch_num=calc_batch_num+buffer_size-num_of_batches
            
        else:
            target_preprocessing_iter_num=calc_iter_num
            target_preprocessing_batch_num=calc_batch_num+buffer_size
            
        if(target_preprocessing_iter_num==preprocessing_iter_num):

            
            try:
                pool = Pool(num_of_threads) 
                run_preprocessing = utils.Run_preprocessing(ds_location,buffer_folder, shuffled_indices, args.epoch, preprocessing_batch_num, target_preprocessing_batch_num, batch_size, num_of_threads, mode='calc_bn_avgs', iter_num=preprocessing_iter_num)
                pool.map(run_preprocessing, pool_input_list)
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
            
            
            
                
        elif(target_preprocessing_iter_num==(preprocessing_iter_num+1)):        
           
            
            try:
                pool = Pool(num_of_threads) 
                run_preprocessing = utils.Run_preprocessing(ds_location,buffer_folder, shuffled_indices, args.epoch, preprocessing_batch_num, num_of_batches, batch_size, num_of_threads, mode='calc_bn_avgs', iter_num=preprocessing_iter_num)
                pool.map(run_preprocessing, pool_input_list)
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
            
            
                
            shuffled_indices=np.random.permutation(N)
            print("Performed re-shuffling")
            
            if(target_preprocessing_iter_num==num_of_iterations):
                keepgoing=0
            else:
                
                try:
                    pool = Pool(num_of_threads) 
                    run_preprocessing = utils.Run_preprocessing(ds_location, buffer_folder, shuffled_indices, args.epoch, 0, target_preprocessing_batch_num, batch_size, num_of_threads, mode='calc_bn_avgs', iter_num=target_preprocessing_iter_num)
                    pool.map(run_preprocessing, pool_input_list)
                finally: # To make sure processes are closed in the end, even if errors happen
                    pool.close()
                    pool.join()
                
        preprocessing_batch_num=target_preprocessing_batch_num
        preprocessing_iter_num=target_preprocessing_iter_num
        
    else:
        
        print("Waiting...")
        time.sleep(sleep_time)
            
        
    



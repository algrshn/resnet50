import numpy as np
import argparse
import sys
import time

import utils

#mu=np.array([0.47703891, 0.45346393, 0.40395429])  #actual values
#sigma=np.array([0.27914875, 0.27181716, 0.28552585])  #actual values

#-------------hard coded data----------------------------
ds_folder="/media/alex/data/npy/"
buffer_folder="/mnt/ramdisk/buffer/"
N=1281167
buffer_size=1000
#buffer_size is specified in units of batch_size (e.g. with batch_size=64
#and buffer_size=1000 the buffer will hold 64000 images)
sleeptime=10 
#--------------------------------------------------------

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
    
batch_size=args.batch_size

num_of_batches=N//batch_size

epoch=args.epoch_start
batch_num=0
keepgoing=1

shuffled_indices=np.random.permutation(N)
preprocessing_epoch=args.epoch_start
preprocessing_batch_num=0

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
            for batch_num in range(preprocessing_batch_num,target_preprocessing_batch_num):

                utils.run_preprocessing(ds_folder,buffer_folder, shuffled_indices, preprocessing_epoch, batch_num, batch_size)
                
        elif(target_preprocessing_epoch==(preprocessing_epoch+1)):
            
            for batch_num in range(preprocessing_batch_num,num_of_batches):
                
                utils.run_preprocessing(ds_folder,buffer_folder, shuffled_indices, preprocessing_epoch, batch_num, batch_size)
                
            shuffled_indices=np.random.permutation(N)
            print("Performed re-shuffling")
            
            if(target_preprocessing_epoch==args.epoch_end):
                keepgoing=0
            else:
                for batch_num in range(0,target_preprocessing_batch_num):
                
                    utils.run_preprocessing(ds_folder, buffer_folder, shuffled_indices, target_preprocessing_epoch, batch_num, batch_size)
                
        preprocessing_batch_num=target_preprocessing_batch_num
        preprocessing_epoch=target_preprocessing_epoch
        
    else:
        
        print("Waiting...")
        time.sleep(sleeptime)
            
        
    



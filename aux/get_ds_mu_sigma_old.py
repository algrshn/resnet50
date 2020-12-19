import numpy as np

ds_folder="/media/alex/data1/npy/"
#save_folder="/media/alex/data/npy_val/"
N=1281167
N#=50000


#Step1
sum_of_pixels=np.zeros((3))
total_num_pixels=0
for i in range(N):
    print(i)
    X=np.load(ds_folder+"X16_"+str(i)+".npy")
    h=X.shape[0]
    w=X.shape[1]
    total_num_pixels+=h*w
    sum_of_pixels+=np.sum(X,axis=(0,1),dtype=np.float64)
  
print(sum_of_pixels/total_num_pixels)

#mu=np.array([0.47703891, 0.45346393, 0.40395429])  #actual values


#Step2

#mu_shaped=np.zeros((1,1,3))
#mu_shaped[0,0,:]=mu[:]
#
#sum_of_squares=np.zeros((3))
#total_num_pixels=0
#for i in range(N):
#    print(i)
#    X=np.load(ds_folder+'X16_'+str(i)+'.npy')
#    total_num_pixels+=X.shape[0]*X.shape[1]
#    #X_centered=np.zeros_like(X)
#    #X_squared=np.zeros_like(X)
#
#    X_centered=X-mu_shaped    
#    X_squared=X_centered*X_centered
#    sum_of_squares+=np.sum(X_squared,axis=(0,1),dtype=np.float64)
#
#
#    
#print(np.sqrt(sum_of_squares/total_num_pixels))

#sigma=np.array([0.27914875, 0.27181716, 0.28552585])  #actual values
#
##Step3
#mu_113=np.zeros((1,1,3))
#mu_113[0,0,:]=mu[:]
#sigma_113=np.zeros((1,1,3))
#sigma_113[0,0,:]=sigma[:]
#
#
#for i in range(N):
#    
#    print(i)
#    
#    X=np.load(ds_folder+"X16_"+str(i)+".npy")
#    X_n=(X-mu_113)/sigma_113
#    X_n16=X_n.astype(dtype=np.float16)
#    
#    np.save(save_folder+"X16_"+str(i)+".npy",X_n16)
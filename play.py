import numpy as np
from matplotlib import pyplot as plt
import configparser
import sys


mu_global=np.zeros((3))
sigma_global=np.ones((3))

config = configparser.ConfigParser()
config.read('config.txt')

try:
    mu_global[0]=config.get('utils','mu_0')
except:
    sys.exit("Check configuration file config.txt. Option mu_0 does not exist in section [utils].")
try:
    mu_global[1]=config.get('utils','mu_1')
except:
    sys.exit("Check configuration file config.txt. Option mu_1 does not exist in section [utils].")
try:
    mu_global[2]=config.get('utils','mu_2')
except:
    sys.exit("Check configuration file config.txt. Option mu_2 does not exist in section [utils].")
try:
    sigma_global[0]=config.get('utils','sigma_0')
except:
    sys.exit("Check configuration file config.txt. Option sigma_0 does not exist in section [utils].")
try:
    sigma_global[1]=config.get('utils','sigma_1')
except:
    sys.exit("Check configuration file config.txt. Option sigma_1 does not exist in section [utils].")
try:
    sigma_global[2]=config.get('utils','sigma_2')
except:
    sys.exit("Check configuration file config.txt. Option sigma_2 does not exist in section [utils].")

mu_shaped=np.zeros((1,1,3))
mu_shaped[0,0,:]=mu_global[:]
sigma_shaped=np.ones((1,1,3))
sigma_shaped[0,0,:]=sigma_global[:]

n=70
k=0

X=np.zeros((224,224,3))

X50=np.load("/media/alex/data1/npy_val2/X50_" + str(n) + ".npy")
y=np.load("/media/alex/data1/npy_val/y_" + str(n) + ".npy")

X[:,:,:]=X50[k,:,:,:]

img=X*sigma_shaped+mu_shaped


print(y[0])

plt.imshow(img)
plt.show()



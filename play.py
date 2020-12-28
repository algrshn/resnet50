import configparser
import sys
import numpy as np
import skimage.io

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
    
imgfilename="/media/alex/data2/403/img_13844.jpg"

img_uint8=skimage.io.imread(imgfilename)
img=img_uint8/255.0
    
#crop
img224=np.zeros((224,224,3),dtype=np.float32)

h=img.shape[0]
w=img.shape[1]

blc_max_x=w-224
blc_max_y=h-224

blc_x=np.random.randint(low=0,high=blc_max_x+1)
blc_y=np.random.randint(low=0,high=blc_max_y+1)

img224[:,:,:]=img[blc_y:(blc_y+224),blc_x:(blc_x+224),:]

mu_shaped=np.zeros((1,1,3))
mu_shaped[0,0,:]=mu_global[:]
sigma_shaped=np.ones((1,1,3))
sigma_shaped[0,0,:]=sigma_global[:]

X224=(img224-mu_shaped)/sigma_shaped

print(X224)
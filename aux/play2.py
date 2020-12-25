import numpy as np
import skimage.io
from matplotlib import pyplot as plt

n=751

y=np.load("/media/alex/data2/256/y_" + str(n) + ".npy")
print(y[0])
img=skimage.io.imread("/media/alex/data2/256/img_" + str(n) +".jpg")
skimage.io.imshow(img)
plt.show()




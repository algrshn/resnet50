# import numpy as np

# A=np.array([0.123, 0.03, 0.56, 0.1, 0.78, 0.002, 0.11, 0.55, 0.98, 0.01])

# b=np.argsort(A, axis=0)

# c=np.flip(b, axis=0)

# print(c)

import numpy as np
import tensorflow as tf
a = np.array([[1,2,3],[4,5,6]])

x=tf.convert_to_tensor(a,dtype=tf.dtypes.float32)

proto_tensor = tf.make_tensor_proto(x)
b=tf.make_ndarray(proto_tensor)

print(b)
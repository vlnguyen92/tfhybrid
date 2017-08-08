import tensorflow as tf
import time
import numpy as np
import sys

dim = int(sys.argv[1])
n = 100 
c1 = tf.Variable([])
c2 = tf.Variable([])
def matpow(M, n):
    if n < 1: 
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

with tf.device("/job:worker/task:0/cpu:0"):
    A = np.random.rand(dim, dim).astype('float32')

with tf.device("/job:worker/task:1/cpu:0"):
    t1 = time.time()
    id = tf.identity(A)  
    t2 = time.time()
    idTime = t2-t1

with tf.Session("grpc://localhost:2222") as sess:        
#    matpow(B,2)
    t1 = time.time()
    sess.run(id)
    t2 = time.time()
    print (str(dim) + " " + str(t2-t1-idTime))

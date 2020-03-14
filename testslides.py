import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

Weight = [[0.01,0.7,0.0],[-0.05,0.2,-0.45],[0.1,0.05,-0.2],[0.05,0.16,0.03]]
bias = [0.0,0.2,-0.3]
data = [-15.0,22.0,-44.0,56.0]

Weight = tf.reshape(Weight,[4,3])
bias = tf.reshape(bias,[1,3])
data=tf.reshape(data,[1,4])

c = tf.matmul(data, Weight)
z = c+bias
prediction =tf.nn.softmax(z)
with tf.Session() as sess:
    s=0
    z = c+bias

   # print (sess.run([data,Weight,bias,c,z,prediction]))
    p = sess.run(prediction)
    print p



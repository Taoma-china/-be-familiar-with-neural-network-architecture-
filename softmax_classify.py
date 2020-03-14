import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
def softmax_fun(z):
    prediction=[]
    sum_z = 0
    for j in range(20):
        sum_z = sum_z +pow((math.e),(z[0][j]))
    log_sum = math.log(sum_z,2)
    for i in range(20):

        prediction.append(pow((math.e),(z[0][i]))/sum_z)
    return prediction,log_sum

data_weight=('weight.mat')
Weight = scio.loadmat(data_weight)

data_bias = ('bias.mat')
bias = scio.loadmat(data_bias)

data_sample = ('data.mat')
data = scio.loadmat(data_sample)

Weight= (Weight["softmax_weight"])
bias = (bias['softmax_bias'])
data = (data['asample'])



Weight = tf.reshape(Weight,[100,20])
bias = tf.reshape(bias,[1,20])
data=tf.reshape(data,[1,100])

sess = tf.Session()



c = tf.matmul(data, Weight)
z = c+bias
z = sess.run(z)


log_sum=[]
probability, log_sum= softmax_fun(z)
#loss = z[0][0]-log_sum
#print probability
#print z
#print probability-z



#gradient descent
#with tf.Session() as sess:
 #   s=0
  #  z = c+bias
   # print (sess.run([data,Weight,bias,c,z,prediction]))
   # p = sess.run(prediction)
    #print p

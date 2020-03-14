import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import tensorflow as tf
import numpy as np


data_weight=('weight.mat')
Weight = scio.loadmat(data_weight)

data_bias = ('bias.mat')
bias = scio.loadmat(data_bias)

data_sample = ('data.mat')
data = scio.loadmat(data_sample)

Weight= (Weight["softmax_weight"])
bias = (bias['softmax_bias'])
data = (data['asample'])


bias = np.array(bias)
Weight = np.array(Weight)
scores = np.array(data)
scores -=np.max(scores)
p=np.exp(scores)/np.sum(np.exp(scores))
def softmax_loss_vectorized(W, X, y, reg):
     loss = 0.0
     dW = np.zeros_like(W)

     num_train = X.shape[0]
     num_classes = W.shape[1]
     scores = X.dot(W)
     scores_shift = scores - np.max(scores, axis = 1).reshape(-1,1)
     softmax_output = np.exp(scores_shift) / np.sum(np.exp(scores_shift), axis=1).reshape(-1,1)
     loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
     loss /= float(num_train)
     loss += 0.5 * reg * np.sum(W * W)

     dS = softmax_output.copy()
     dS[range(num_train), list(y)] += -1
     dW = (X.T).dot(dS)
     dW = dW / num_train + reg * W  

     return loss, dW
loss_value,dw_value = softmax_loss_vectorized(Weight,data,p,0.005)
print (loss_value)
